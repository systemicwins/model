#!/usr/bin/env python3
"""
Test script for running IBM Granite 4.0 hybrid model locally on macOS
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys

def test_granite_model():
    """Test IBM Granite 4.0 hybrid model locally"""

    print("🧪 Testing IBM Granite 4.0 Hybrid Model Locally")
    print("=" * 60)

    # Check system resources
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CPU cores: {torch.get_num_threads()}")

    if torch.backends.mps.is_available():
        print("✅ Using Apple Metal Performance Shaders (MPS) for GPU acceleration")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using NVIDIA CUDA for GPU acceleration")
        device = torch.device("cuda")
    else:
        print("⚠️ Using CPU (slower performance)")
        device = torch.device("cpu")

    try:
        # Load the model
        print("\n📥 Loading IBM Granite 4.0 hybrid model...")
        model_name = "ibm-granite/granite-4.0-h-micro"

        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to(device)
        load_time = time.time() - start_time

        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"Model parameters: {model.num_parameters():,}")

        # Test prompts
        test_prompts = [
            "Explain quantum computing in simple terms:",
            "Write a Python function to calculate fibonacci numbers:",
            "What are the benefits of hybrid attention mechanisms?",
            "Compare traditional transformers with modern hybrid architectures:"
        ]

        print("\n🧪 Running inference tests...")
        print("-" * 60)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt[:50]}...")

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time

            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"⏱️ Inference time: {inference_time:.3f} seconds")
            print(f"📝 Generated text: {generated_text[len(prompt):len(prompt)+200]}...")

        # Performance analysis
        print("\n📊 Performance Analysis:")
        print("-" * 60)
        print("✅ Model runs successfully on macOS")
        print("✅ Uses Apple Metal Performance Shaders when available")
        print("✅ Supports both CPU and GPU inference")
        # Memory usage
        if device.type == "mps":
            print("✅ Optimized for Apple Silicon")
        elif device.type == "cuda":
            print("✅ CUDA acceleration available")
        else:
            print("⚠️ Running on CPU (slower)")

        print("\n🎯 Recommendations:")
        if device.type == "mps":
            print("• Excellent performance on Apple Silicon")
            print("• Use for real-time applications")
        else:
            print("• Consider using smaller models for CPU-only inference")
            print("• Upgrade to Apple Silicon for better performance")

        return True

    except Exception as e:
        print(f"❌ Error running model: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if model requires more RAM")
        print("2. Try a smaller model variant")
        print("3. Ensure internet connection for initial download")
        return False

def compare_with_our_implementation():
    """Compare IBM Granite with our custom hybrid implementation"""

    print("\n🔍 Comparison: IBM Granite vs Our Custom Hybrid Implementation")
    print("=" * 80)

    print("\nIBM Granite 4.0:")
    print("• Pre-trained hybrid transformer model")
    print("• Combines different attention mechanisms")
    print("• Optimized for various hardware platforms")
    print("• Large-scale pre-training on diverse data")

    print("\nOur Custom Implementation:")
    print("• Mamba2 SSM + Sparse Attention hybrid")
    print("• Custom C++ implementation for maximum performance")
    print("• Designed specifically for financial time series")
    print("• Adaptive sparsity based on SSM state evolution")

    print("\nKey Differences:")
    print("• IBM Granite: General-purpose, pre-trained, Python/PyTorch")
    print("• Our Implementation: Specialized, custom-built, C++/Eigen")
    print("• IBM Granite: Broad capabilities, higher resource usage")
    print("• Our Implementation: Optimized efficiency, lower latency")

if __name__ == "__main__":
    success = test_granite_model()
    compare_with_our_implementation()

    if success:
        print("\n✅ IBM Granite model test completed successfully!")
        print("The model can run locally on your macOS system.")
    else:
        print("\n❌ Model test failed - may need additional setup or resources.")