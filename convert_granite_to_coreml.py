#!/usr/bin/env python3
"""
Convert IBM Granite 4.0 model to CoreML format for native macOS acceleration
"""

import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import os
from pathlib import Path

class GraniteCoreMLConverter:
    """Convert IBM Granite model to CoreML format"""

    def __init__(self, model_name="ibm-granite/granite-4.0-h-micro"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.output_dir = Path("coreml_models")
        self.output_dir.mkdir(exist_ok=True)

    def load_model(self):
        """Load the IBM Granite model and tokenizer"""
        print(f"📥 Loading {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model with appropriate settings for CoreML conversion
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CoreML prefers float32
            low_cpu_mem_usage=True
        )

        print("✅ Model loaded successfully")
        print(f"   Model parameters: {self.model.num_parameters():,}")
        print(f"   Model size: {self.get_model_size_mb():.1f} MB")

    def get_model_size_mb(self):
        """Calculate model size in MB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        # Assuming float32, each parameter is 4 bytes
        return (total_params * 4) / (1024 * 1024)

    def prepare_for_conversion(self):
        """Prepare model for CoreML conversion"""
        print("🔧 Preparing model for CoreML conversion...")

        # Set model to evaluation mode
        self.model.eval()

        # Create example input for tracing
        example_text = "Hello, how are you?"
        inputs = self.tokenizer(
            example_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        print(f"   Example input shape: {inputs['input_ids'].shape}")
        return inputs

    def convert_to_coreml(self, inputs, quantize=False):
        """Convert PyTorch model to CoreML format"""
        print("🔄 Converting to CoreML format...")

        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                self.model,
                (inputs['input_ids'], inputs['attention_mask'])
            )

        # Define input types for CoreML
        input_ids = ct.TensorType(
            name="input_ids",
            shape=inputs['input_ids'].shape,
            dtype=np.int32
        )

        attention_mask = ct.TensorType(
            name="attention_mask",
            shape=inputs['attention_mask'].shape,
            dtype=np.int32
        )

        # Convert to CoreML
        if quantize:
            print("   Applying quantization for smaller model size...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[input_ids, attention_mask],
                outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                minimum_deployment_target=ct.target.macOS13,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.ALL
            )
        else:
            mlmodel = ct.convert(
                traced_model,
                inputs=[input_ids, attention_mask],
                outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                minimum_deployment_target=ct.target.macOS13,
                compute_units=ct.ComputeUnit.ALL
            )

        return mlmodel

    def save_model(self, mlmodel, model_name="granite-4.0-h-micro"):
        """Save CoreML model and metadata"""
        print(f"💾 Saving CoreML model as {model_name}...")

        # Save CoreML model
        model_path = self.output_dir / f"{model_name}.mlpackage"
        mlmodel.save(str(model_path))

        # Save tokenizer config
        tokenizer_config = {
            "model_name": self.model_name,
            "vocab_size": len(self.tokenizer),
            "max_position_embeddings": self.model.config.max_position_embeddings,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
        }

        config_path = self.output_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)

        # Save tokenizer files
        self.tokenizer.save_pretrained(self.output_dir / f"{model_name}_tokenizer")

        print(f"✅ Model saved to {model_path}")
        print(f"   Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Config saved to {config_path}")

    def run_conversion(self, quantize=False):
        """Run the complete conversion process"""
        print("🚀 Starting IBM Granite to CoreML conversion")
        print("=" * 60)

        try:
            # Load model
            self.load_model()

            # Prepare for conversion
            inputs = self.prepare_for_conversion()

            # Convert to CoreML
            mlmodel = self.convert_to_coreml(inputs, quantize)

            # Save model
            model_name = "granite-4.0-h-micro-quantized" if quantize else "granite-4.0-h-micro"
            self.save_model(mlmodel, model_name)

            print("\n✅ Conversion completed successfully!")
            print(f"   CoreML model saved in: {self.output_dir}")
            print("   Ready for native macOS inference with CoreML acceleration")
            return True

        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False

def main():
    """Main conversion function"""
    print("🍎 IBM Granite 4.0 CoreML Converter for macOS")
    print("=" * 60)

    # Check system compatibility
    print("System Information:")
    print(f"   macOS: {torch.backends.mps.is_available()}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CoreML tools version: {ct.__version__}")

    # Initialize converter
    converter = GraniteCoreMLConverter()

    # Run conversion (without quantization first)
    success = converter.run_conversion(quantize=False)

    if success:
        print("\n🎉 Ready to use IBM Granite 4.0 with CoreML!")
        print("   The model is now optimized for native macOS performance.")
        print("   Use the coreml_inference.py script for inference.")
    else:
        print("\n❌ Conversion failed. Check the error messages above.")
        print("   You may need to adjust model settings or system configuration.")

if __name__ == "__main__":
    main()