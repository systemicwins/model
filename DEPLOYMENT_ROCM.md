# Deploying Relentless Model with llama.cpp and ROCm

## Overview
This guide explains how to export, convert, and deploy the trained transformer model using llama.cpp with ROCm acceleration.

## Pipeline Steps

### 1. Train and Save Model

```cpp
// In your training code
transformer::TransformerModel model(config);
// ... training loop ...

// Save checkpoint
model.save_weights("checkpoints/model_epoch_100.bin");
```

### 2. Export to GGUF Format

```bash
# Convert checkpoint to GGUF
python scripts/convert_to_gguf.py \
    checkpoints/model_epoch_100.bin \
    models/relentless-financial.gguf \
    --config config.json
```

### 3. Quantize for Efficiency

```bash
# Quantize to Q8_0 (8-bit, good quality/speed balance)
python scripts/convert_to_gguf.py \
    checkpoints/model_epoch_100.bin \
    models/relentless-financial.Q8_0.gguf \
    --quantize Q8_0

# Or Q4_K for smaller size (4-bit with k-means)
python scripts/convert_to_gguf.py \
    checkpoints/model_epoch_100.bin \
    models/relentless-financial.Q4_K.gguf \
    --quantize Q4_K
```

### 4. Deploy with llama.cpp + ROCm

```bash
# Copy model to server
scp models/relentless-financial.Q8_0.gguf olympus:/opt/models/

# Run with ROCm acceleration
ssh olympus
cd /opt/llama.cpp

# Start server with ROCm/HIP
sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    HIP_VISIBLE_DEVICES=0 \
    ./build/bin/llama-server \
    --model /opt/models/relentless-financial.Q8_0.gguf \
    --host 0.0.0.0 \
    --port 11434 \
    --n-gpu-layers 999 \
    --ctx-size 8191 \
    --threads 8 \
    --mlock
```

## Model Format Details

### GGUF Structure
```
relentless-financial.gguf
├── Header
│   ├── Magic: GGUF
│   ├── Version: 3
│   └── Counts: tensors, metadata
├── Metadata
│   ├── Architecture: relentless
│   ├── Parameters
│   │   ├── context_length: 8191
│   │   ├── embedding_length: 1536
│   │   ├── block_count: 6
│   │   ├── attention.head_count: 12
│   │   └── feed_forward_length: 4096
│   ├── Matryoshka
│   │   └── dimensions: [64, 128, 256, 512, 768, 1024, 1536]
│   └── Tokenizer
│       ├── vocab_size: 50000
│       └── special_tokens: <bos>, <eos>, <pad>, <unk>
└── Tensors
    ├── token_embd.weight [50000, 1536]
    ├── blk.0.attn.q.weight [1536, 1536]
    ├── blk.0.attn.k.weight [1536, 1536]
    ├── blk.0.attn.v.weight [1536, 1536]
    ├── blk.0.attn.o.weight [1536, 1536]
    ├── blk.0.ffn.gate.weight [4096, 1536]
    ├── blk.0.ffn.down.weight [1536, 4096]
    └── ... (repeat for 6 layers)
```

### Quantization Options

| Type | Bits | Size Reduction | Quality | Speed | Use Case |
|------|------|---------------|---------|-------|----------|
| F32  | 32   | 1x (baseline) | Perfect | Slow  | Development |
| F16  | 16   | 2x            | Excellent | Good | Production |
| Q8_0 | 8    | 4x            | Very Good | Fast | **Recommended** |
| Q5_K | 5    | 6.4x          | Good | Fast | Memory-constrained |
| Q4_K | 4    | 8x            | Acceptable | Very Fast | Edge deployment |
| Q4_0 | 4    | 8x            | Basic | Very Fast | Testing |

### Memory Requirements

**Base Model (F32)**: ~850MB
- Embeddings: 300MB
- Transformer layers: 504MB  
- Matryoshka: 50MB

**Quantized (Q8_0)**: ~213MB
- 4x reduction from F32
- Minimal quality loss
- ROCm-optimized kernels

**Quantized (Q4_K)**: ~106MB
- 8x reduction from F32
- Good for inference
- Fastest on GPU

## ROCm Optimization

### GPU Layer Offloading
```bash
# Offload all layers to GPU
--n-gpu-layers 999

# Partial offloading (if VRAM limited)
--n-gpu-layers 24  # Offload 24 layers

# CPU-only fallback
--n-gpu-layers 0
```

### Performance Tuning
```bash
# Optimal settings for AMD RX 7900 XTX
HSA_OVERRIDE_GFX_VERSION=11.0.0  # GFX11 architecture
HIP_VISIBLE_DEVICES=0             # Use first GPU
--ctx-size 8191                   # Max context
--batch-size 512                  # Batch processing
--threads 8                        # CPU threads
--mlock                           # Lock memory
--no-mmap                         # Disable mmap for GPU
```

## Integration Examples

### Python Client
```python
import requests

# Query the model
response = requests.post('http://olympus:11434/completion', 
    json={
        'prompt': 'Analyze AAPL Q3 earnings: ',
        'n_predict': 200,
        'temperature': 0.7,
        'top_p': 0.9
    })

print(response.json()['content'])
```

### Streaming Response
```python
import requests
import json

response = requests.post('http://olympus:11434/completion',
    json={'prompt': 'Market analysis:', 'stream': True},
    stream=True)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8'))
        print(data['content'], end='', flush=True)
```

## Benchmarking

### Test inference speed
```bash
./build/bin/llama-bench \
    -m /opt/models/relentless-financial.Q8_0.gguf \
    -n 128 \
    -p "Financial analysis of" \
    --n-gpu-layers 999
```

Expected performance (RX 7900 XTX):
- Prompt eval: ~2000 tokens/sec
- Generation: ~150 tokens/sec
- Memory usage: ~2GB VRAM

## Docker Deployment

```dockerfile
FROM rocm/pytorch:latest

# Install llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    make LLAMA_HIPBLAS=1

# Copy model
COPY models/relentless-financial.Q8_0.gguf /models/

# Start server
CMD ["llama.cpp/build/bin/llama-server", \
     "--model", "/models/relentless-financial.Q8_0.gguf", \
     "--host", "0.0.0.0", \
     "--n-gpu-layers", "999"]
```

## Troubleshooting

### ROCm Detection Issues
```bash
# Check ROCm installation
rocminfo

# Verify GPU is detected
rocm-smi

# Test HIP
hipcc --version
```

### Memory Issues
```bash
# Reduce context size
--ctx-size 2048

# Use smaller quantization
--model relentless-financial.Q4_0.gguf

# Offload fewer layers
--n-gpu-layers 12
```

### Performance Issues
```bash
# Enable flash attention
--flash-attn

# Adjust batch size
--batch-size 256

# Use tensor cores
--tensor-split 1.0
```

## Next Steps

1. **Fine-tuning**: Continue training on financial data
2. **LoRA Adapters**: Add domain-specific adapters
3. **Multimodal**: Add chart/image understanding
4. **Distributed**: Multi-GPU inference setup
5. **Production**: Kubernetes deployment with autoscaling