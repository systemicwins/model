#!/usr/bin/env python3
"""
CoreML inference script for IBM Granite 4.0 model on macOS
Provides native macOS acceleration using Apple's CoreML framework
"""

import coremltools as ct
import numpy as np
import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer
import sys

class GraniteCoreMLInference:
    """Native macOS inference using CoreML for IBM Granite model"""

    def __init__(self, model_path="coreml_models/granite-4.0-h-micro.mlpackage"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.config = None

        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"CoreML model not found at {self.model_path}")

    def load_model(self):
        """Load CoreML model and tokenizer"""
        print("🚀 Loading IBM Granite CoreML model for native macOS inference")
        print("=" * 70)

        # Load CoreML model
        print(f"📦 Loading model from {self.model_path}...")
        start_time = time.time()

        self.model = ct.models.MLModel(str(self.model_path))
        load_time = time.time() - start_time

        print(f"✅ CoreML model loaded in {load_time:.2f} seconds")

        # Load tokenizer
        tokenizer_path = self.model_path.parent / "granite-4.0-h-micro_tokenizer"
        print(f"🔤 Loading tokenizer from {tokenizer_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load config
        config_path = self.model_path.parent / "granite-4.0-h-micro_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        print("✅ Tokenizer and config loaded")
        # Display model info
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        print("\n📊 Model Information:")
        print(f"   Model size: {model_size_mb:.1f} MB")
        print(f"   Vocabulary size: {self.config['vocab_size']:,}")
        print(f"   Max sequence length: {self.config['max_position_embeddings']}")
        print("   Backend: CoreML (Native macOS acceleration)")
    def preprocess_text(self, text, max_length=128):
        """Preprocess text for CoreML model input"""
        # Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        # Convert to correct format for CoreML
        input_ids = tokens["input_ids"].astype(np.int32)
        attention_mask = tokens["attention_mask"].astype(np.int32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "original_text": text
        }

    def run_inference(self, inputs):
        """Run inference using CoreML"""
        # Run prediction
        start_time = time.time()
        prediction = self.model.predict(inputs)
        inference_time = time.time() - start_time

        return prediction, inference_time

    def postprocess_output(self, prediction, original_text):
        """Postprocess CoreML output"""
        # Get logits from prediction
        logits = prediction["logits"]

        # Convert to token probabilities
        probabilities = torch.softmax(torch.tensor(logits), dim=-1)

        # Get predicted token (greedy decoding)
        predicted_token = torch.argmax(probabilities[0, -1]).item()

        # Decode token
        generated_text = self.tokenizer.decode([predicted_token], skip_special_tokens=True)

        return {
            "generated_token": predicted_token,
            "generated_text": generated_text,
            "probabilities": probabilities.numpy(),
            "confidence": probabilities[0, -1, predicted_token].item()
        }

    def generate_text(self, prompt, max_tokens=50, temperature=0.7):
        """Generate text using CoreML model"""
        print(f"\n🧠 Generating text for: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print("-" * 70)

        current_text = prompt
        tokens_generated = 0

        while tokens_generated < max_tokens:
            # Preprocess
            inputs = self.preprocess_text(current_text)

            # Run inference
            prediction, inference_time = self.run_inference({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            })

            # Postprocess
            output = self.postprocess_output(prediction, current_text)

            # Check if we should stop (EOS token or low confidence)
            if (output["generated_token"] == self.config["eos_token_id"] or
                output["confidence"] < 0.1):
                break

            # Add generated text
            current_text += output["generated_text"]
            tokens_generated += 1

            print(f"⏱️  Inference time: {inference_time:.3f}s | "
                  f"Confidence: {output['confidence']:.3f} | "
                  f"Generated: '{output['generated_text']}'")

        return current_text

    def benchmark_performance(self, test_prompts=None):
        """Benchmark CoreML model performance"""
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "Explain quantum computing in simple terms:",
                "Write a Python function to calculate fibonacci numbers:",
                "What are the benefits of machine learning?"
            ]

        print("\n🏃 Performance Benchmark")
        print("=" * 50)

        total_time = 0
        num_tests = len(test_prompts)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt[:40]}{'...' if len(prompt) > 40 else ''}")

            # Measure inference time
            inputs = self.preprocess_text(prompt)

            start_time = time.time()
            prediction, _ = self.run_inference({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            })
            inference_time = time.time() - start_time

            total_time += inference_time

            print(f"   ⏱️  Inference time: {inference_time:.3f} seconds")
            print(f"   🚀 Throughput: {len(prompt.split()) / inference_time:.1f} tokens/sec")

        avg_time = total_time / num_tests
        print("\n📊 Results:")
        print(f"   Average inference time: {avg_time:.3f} seconds")
        print(f"   Total tests: {num_tests}")
        print("   Backend: CoreML (Apple Silicon optimized)")
        return avg_time

def main():
    """Main inference demo"""
    print("🍎 IBM Granite 4.0 CoreML Inference Demo")
    print("Native macOS acceleration with CoreML framework")
    print("=" * 60)

    try:
        # Initialize inference engine
        inference = GraniteCoreMLInference()
        inference.load_model()

        # Demo prompts
        demo_prompts = [
            "The future of artificial intelligence is",
            "Explain machine learning in simple terms:",
            "Write a haiku about coding:",
            "What are the advantages of hybrid attention mechanisms?"
        ]

        print("\n🎯 Running inference demos...")
        print("=" * 50)

        for prompt in demo_prompts:
            result = inference.generate_text(prompt, max_tokens=30)
            print(f"\n📝 Result: {result}")
            print("-" * 50)

        # Performance benchmark
        inference.benchmark_performance()

        print("\n✅ CoreML inference completed successfully!")
        print("🎉 IBM Granite 4.0 is now running with native macOS acceleration!")

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure the CoreML model was converted successfully")
        print("2. Check that all dependencies are installed")
        print("3. Verify macOS compatibility (macOS 13+)")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)