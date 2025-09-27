#!/usr/bin/env python3
"""
Convert trained transformer model to GGUF format for llama.cpp with ROCm support
"""

import struct
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

class GGUFWriter:
    """Write model in GGUF format for llama.cpp"""
    
    def __init__(self, path: str):
        self.path = path
        self.fout = open(path, 'wb')
        self.tensors = []
        self.kv_pairs = []
        
    def write_header(self):
        """Write GGUF header"""
        # Magic
        self.fout.write(struct.pack('<I', GGUF_MAGIC))
        # Version
        self.fout.write(struct.pack('<I', GGUF_VERSION))
        # Placeholder for tensor count
        self.tensor_count_offset = self.fout.tell()
        self.fout.write(struct.pack('<Q', 0))
        # Placeholder for metadata count
        self.kv_count_offset = self.fout.tell()
        self.fout.write(struct.pack('<Q', 0))
        
    def add_architecture(self):
        """Add model architecture metadata"""
        self.add_string("general.architecture", "relentless")
        self.add_string("general.name", "relentless-financial-transformer")
        self.add_uint32("general.quantization_version", 2)
        
    def add_model_params(self, config: dict):
        """Add model hyperparameters"""
        # Model dimensions
        self.add_uint32("relentless.context_length", config['max_seq_length'])
        self.add_uint32("relentless.embedding_length", config['embed_dim'])
        self.add_uint32("relentless.block_count", config['num_layers'])
        self.add_uint32("relentless.attention.head_count", config['num_heads'])
        self.add_uint32("relentless.feed_forward_length", config['ff_dim'])
        
        # Matryoshka dimensions
        self.add_array("relentless.matryoshka_dims", [64, 128, 256, 512, 768, 1024, 1536])
        
        # Positional encoding
        self.add_string("relentless.position_encoding", "rope")
        self.add_uint32("relentless.rope_dims", config['embed_dim'])
        
    def add_tokenizer_params(self, vocab_size: int):
        """Add tokenizer configuration"""
        self.add_uint32("tokenizer.ggml.model", 2)  # WordPiece
        self.add_uint32("tokenizer.ggml.vocab_size", vocab_size)
        self.add_string("tokenizer.ggml.bos_token", "<bos>")
        self.add_string("tokenizer.ggml.eos_token", "<eos>")
        self.add_string("tokenizer.ggml.pad_token", "<pad>")
        self.add_string("tokenizer.ggml.unk_token", "<unk>")
        self.add_uint32("tokenizer.ggml.bos_token_id", 2)
        self.add_uint32("tokenizer.ggml.eos_token_id", 3)
        self.add_uint32("tokenizer.ggml.pad_token_id", 0)
        self.add_uint32("tokenizer.ggml.unk_token_id", 1)
        
    def add_tensor(self, name: str, data: np.ndarray, quantize: str = "f32"):
        """Add a tensor to the model"""
        # Store tensor info for later writing
        self.tensors.append({
            'name': name,
            'data': data,
            'shape': data.shape,
            'dtype': quantize
        })
        
    def add_string(self, key: str, value: str):
        """Add string key-value pair"""
        self.kv_pairs.append(('string', key, value))
        
    def add_uint32(self, key: str, value: int):
        """Add uint32 key-value pair"""
        self.kv_pairs.append(('uint32', key, value))
        
    def add_array(self, key: str, value: list):
        """Add array key-value pair"""
        self.kv_pairs.append(('array', key, value))
        
    def write_metadata(self):
        """Write all metadata key-value pairs"""
        # Update KV count
        current_pos = self.fout.tell()
        self.fout.seek(self.kv_count_offset)
        self.fout.write(struct.pack('<Q', len(self.kv_pairs)))
        self.fout.seek(current_pos)
        
        # Write each KV pair
        for kv_type, key, value in self.kv_pairs:
            # Write key
            key_bytes = key.encode('utf-8')
            self.fout.write(struct.pack('<Q', len(key_bytes)))
            self.fout.write(key_bytes)
            
            # Write value based on type
            if kv_type == 'string':
                self.fout.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING
                value_bytes = value.encode('utf-8')
                self.fout.write(struct.pack('<Q', len(value_bytes)))
                self.fout.write(value_bytes)
            elif kv_type == 'uint32':
                self.fout.write(struct.pack('<I', 4))  # GGUF_TYPE_UINT32
                self.fout.write(struct.pack('<I', value))
            elif kv_type == 'array':
                self.fout.write(struct.pack('<I', 9))  # GGUF_TYPE_ARRAY
                self.fout.write(struct.pack('<I', 4))  # array of uint32
                self.fout.write(struct.pack('<Q', len(value)))
                for v in value:
                    self.fout.write(struct.pack('<I', v))
                    
    def write_tensor_info(self):
        """Write tensor metadata"""
        # Update tensor count
        current_pos = self.fout.tell()
        self.fout.seek(self.tensor_count_offset)
        self.fout.write(struct.pack('<Q', len(self.tensors)))
        self.fout.seek(current_pos)
        
        # Calculate offsets
        data_offset = self.fout.tell()
        for tensor in self.tensors:
            # Tensor name
            name_bytes = tensor['name'].encode('utf-8')
            data_offset += 8 + len(name_bytes)  # name length + name
            data_offset += 4  # n_dims
            data_offset += 8 * len(tensor['shape'])  # dimensions
            data_offset += 4  # dtype
            data_offset += 8  # offset
            
        # Align to 256 bytes
        data_offset = (data_offset + 255) & ~255
        
        # Write tensor info
        for tensor in self.tensors:
            # Name
            name_bytes = tensor['name'].encode('utf-8')
            self.fout.write(struct.pack('<Q', len(name_bytes)))
            self.fout.write(name_bytes)
            
            # Dimensions
            self.fout.write(struct.pack('<I', len(tensor['shape'])))
            for dim in tensor['shape']:
                self.fout.write(struct.pack('<Q', dim))
                
            # Data type (0 = f32)
            dtype_map = {'f32': 0, 'f16': 1, 'q4_0': 2, 'q8_0': 8}
            self.fout.write(struct.pack('<I', dtype_map.get(tensor['dtype'], 0)))
            
            # Offset
            self.fout.write(struct.pack('<Q', data_offset))
            tensor['offset'] = data_offset
            
            # Calculate next offset
            data_size = np.prod(tensor['shape']) * 4  # f32 = 4 bytes
            data_offset += data_size
            data_offset = (data_offset + 31) & ~31  # Align to 32 bytes
            
    def write_tensor_data(self):
        """Write actual tensor data"""
        # Align to 256 bytes for data section
        current = self.fout.tell()
        padding = (256 - (current % 256)) % 256
        if padding:
            self.fout.write(b'\x00' * padding)
            
        # Write each tensor's data
        for tensor in self.tensors:
            data = tensor['data'].astype(np.float32)
            self.fout.write(data.tobytes())
            
            # Align to 32 bytes
            current = self.fout.tell()
            padding = (32 - (current % 32)) % 32
            if padding:
                self.fout.write(b'\x00' * padding)
                
    def finalize(self):
        """Complete the GGUF file"""
        self.fout.close()
        print(f"GGUF model saved to: {self.path}")


def convert_checkpoint_to_gguf(checkpoint_path: str, output_path: str, 
                              config_path: str = None):
    """Convert a model checkpoint to GGUF format"""
    
    print(f"Converting {checkpoint_path} to GGUF format...")
    
    # Load config
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'vocab_size': 50000,
            'max_seq_length': 8191,
            'embed_dim': 1536,
            'num_layers': 6,
            'num_heads': 12,
            'ff_dim': 4096
        }
    
    # Create GGUF writer
    writer = GGUFWriter(output_path)
    writer.write_header()
    
    # Add metadata
    writer.add_architecture()
    writer.add_model_params(config)
    writer.add_tokenizer_params(config['vocab_size'])
    
    # Load checkpoint weights
    print("Loading checkpoint weights...")
    weights = load_checkpoint_weights(checkpoint_path)
    
    # Convert weight names to GGUF convention
    for pytorch_name, tensor_data in weights.items():
        gguf_name = convert_weight_name(pytorch_name)
        writer.add_tensor(gguf_name, tensor_data)
    
    # Write everything
    writer.write_metadata()
    writer.write_tensor_info()
    writer.write_tensor_data()
    writer.finalize()
    
    print(f"Conversion complete! Output: {output_path}")
    return output_path


def load_checkpoint_weights(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """Load weights from checkpoint file"""
    # This is a placeholder - implement based on your checkpoint format
    # For now, create dummy weights for demonstration
    
    weights = {}
    
    # Token embeddings
    weights['token_embeddings.weight'] = np.random.randn(50000, 1536).astype(np.float32) * 0.02
    
    # Each transformer layer
    for layer in range(6):
        prefix = f'layers.{layer}'
        
        # Attention weights
        weights[f'{prefix}.attention.q_proj.weight'] = np.random.randn(1536, 1536).astype(np.float32) * 0.02
        weights[f'{prefix}.attention.k_proj.weight'] = np.random.randn(1536, 1536).astype(np.float32) * 0.02
        weights[f'{prefix}.attention.v_proj.weight'] = np.random.randn(1536, 1536).astype(np.float32) * 0.02
        weights[f'{prefix}.attention.o_proj.weight'] = np.random.randn(1536, 1536).astype(np.float32) * 0.02
        
        # FFN weights
        weights[f'{prefix}.ffn.fc1.weight'] = np.random.randn(4096, 1536).astype(np.float32) * 0.02
        weights[f'{prefix}.ffn.fc2.weight'] = np.random.randn(1536, 4096).astype(np.float32) * 0.02
        
        # Layer norms
        weights[f'{prefix}.ln1.weight'] = np.ones(1536, dtype=np.float32)
        weights[f'{prefix}.ln1.bias'] = np.zeros(1536, dtype=np.float32)
        weights[f'{prefix}.ln2.weight'] = np.ones(1536, dtype=np.float32)
        weights[f'{prefix}.ln2.bias'] = np.zeros(1536, dtype=np.float32)
    
    # Final layer norm
    weights['final_ln.weight'] = np.ones(1536, dtype=np.float32)
    weights['final_ln.bias'] = np.zeros(1536, dtype=np.float32)
    
    # Output projection
    weights['output.weight'] = np.random.randn(1536, 1536).astype(np.float32) * 0.02
    
    return weights


def convert_weight_name(pytorch_name: str) -> str:
    """Convert PyTorch weight names to GGUF naming convention"""
    # Map from PyTorch style to GGUF style
    name_map = {
        'token_embeddings.weight': 'token_embd.weight',
        'final_ln.weight': 'output_norm.weight',
        'final_ln.bias': 'output_norm.bias',
        'output.weight': 'output.weight',
    }
    
    # Handle layer-specific weights
    if 'layers.' in pytorch_name:
        # Extract layer number
        parts = pytorch_name.split('.')
        layer_num = parts[1]
        remainder = '.'.join(parts[2:])
        
        # Convert component names
        conversions = {
            'attention.q_proj': 'attn.q',
            'attention.k_proj': 'attn.k',
            'attention.v_proj': 'attn.v',
            'attention.o_proj': 'attn.o',
            'ffn.fc1': 'ffn.gate',
            'ffn.fc2': 'ffn.down',
            'ln1': 'attn_norm',
            'ln2': 'ffn_norm',
        }
        
        for old, new in conversions.items():
            if remainder.startswith(old):
                remainder = remainder.replace(old, new, 1)
                break
                
        return f'blk.{layer_num}.{remainder}'
    
    return name_map.get(pytorch_name, pytorch_name)


def quantize_gguf(input_path: str, output_path: str, quant_type: str = "Q8_0"):
    """Quantize a GGUF model using llama.cpp quantize tool"""
    import subprocess
    
    print(f"Quantizing model to {quant_type}...")
    
    # Use llama.cpp's quantize tool
    cmd = [
        "llama-quantize",
        input_path,
        output_path,
        quant_type
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"Quantized model saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Quantization failed: {e}")
        print(e.stderr)
    except FileNotFoundError:
        print("llama-quantize not found. Please install llama.cpp first.")
        print("Manual quantization command:")
        print(f"  llama-quantize {input_path} {output_path} {quant_type}")


def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF for llama.cpp")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("output", help="Output GGUF file path")
    parser.add_argument("--config", help="Model config JSON file")
    parser.add_argument("--quantize", choices=["Q4_0", "Q4_K", "Q5_K", "Q8_0", "F16"],
                       help="Quantization type")
    
    args = parser.parse_args()
    
    # Convert to GGUF
    gguf_path = convert_checkpoint_to_gguf(args.checkpoint, args.output, args.config)
    
    # Optionally quantize
    if args.quantize:
        quant_path = args.output.replace('.gguf', f'.{args.quantize}.gguf')
        quantize_gguf(gguf_path, quant_path, args.quantize)


if __name__ == "__main__":
    main()