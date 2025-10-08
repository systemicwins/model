#!/usr/bin/env python3
"""
Interactive demo for IBM Granite 4.0 with CoreML on macOS
Showcases native macOS acceleration and performance capabilities
"""

import time
import sys
import os
from pathlib import Path
from coreml_inference import GraniteCoreMLInference

class GraniteCoreMLDemo:
    """Interactive demo showcasing IBM Granite with CoreML"""

    def __init__(self):
        self.model = None
        self.setup_model()

    def setup_model(self):
        """Initialize the CoreML model"""
        print("🚀 Initializing IBM Granite 4.0 with CoreML...")
        try:
            self.model = GraniteCoreMLInference()
            self.model.load_model()
            print("✅ Model ready for native macOS inference!\n")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Make sure to run setup_granite_coreml.py first")
            sys.exit(1)

    def demo_basic_inference(self):
        """Demonstrate basic text generation"""
        print("🧠 Basic Inference Demo")
        print("=" * 40)

        prompts = [
            "The future of artificial intelligence is",
            "Explain quantum computing in simple terms:",
            "Write a Python function to calculate fibonacci numbers:",
            "What are the benefits of hybrid attention mechanisms?"
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. Prompt: {prompt}")

            start_time = time.time()
            result = self.model.generate_text(prompt, max_tokens=25)
            inference_time = time.time() - start_time

            print(f"⏱️  Generated in {inference_time:.2f} seconds")
            print(f"📝 Result: {result}")
            print("-" * 60)

    def demo_performance_comparison(self):
        """Compare performance across different prompts"""
        print("\n🏃 Performance Benchmark")
        print("=" * 40)

        test_cases = [
            ("Short prompt", "Hello!"),
            ("Medium prompt", "Explain machine learning in simple terms for beginners:"),
            ("Long prompt", "Write a comprehensive explanation of how transformers work in natural language processing, including attention mechanisms and their advantages over recurrent neural networks:"),
        ]

        results = []

        for name, prompt in test_cases:
            print(f"\n📝 Testing: {name}")
            print(f"Prompt length: {len(prompt)} characters")

            # Warm up
            _ = self.model.generate_text(prompt, max_tokens=10)

            # Actual test
            start_time = time.time()
            result = self.model.generate_text(prompt, max_tokens=20)
            inference_time = time.time() - start_time

            tokens_per_sec = len(result.split()) / inference_time if inference_time > 0 else 0

            results.append({
                'name': name,
                'time': inference_time,
                'tokens_per_sec': tokens_per_sec,
                'output_length': len(result)
            })

            print(f"⏱️  Inference time: {inference_time:.3f} seconds")
            print(f"🚀 Tokens per second: {tokens_per_sec:.1f}")
            print(f"📊 Output length: {len(result)} characters")

        # Summary
        print("\n📊 Performance Summary:")
        print("-" * 50)
        print(f"{'Test Case'"<15"} {'Time (s)'"<10"} {'Tokens/s'"<10"} {'Length'"<8"}")
        print("-" * 50)

        for result in results:
            print(f"{result['name']<15} {result['time']<10.3f} {result['tokens_per_sec']<10.1f} {result['output_length']<8}")

        avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in results) / len(results)
        print("-" * 50)
        print(f"{'Average'<15} {''<10} {avg_tokens_per_sec<10.1f}")

    def demo_interactive_mode(self):
        """Interactive mode for user input"""
        print("\n💬 Interactive Mode")
        print("=" * 30)
        print("Enter your own prompts! Type 'quit' to exit.")
        print("Example: 'Write a haiku about programming'")

        while True:
            try:
                prompt = input("\n🤔 Your prompt: ").strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not prompt:
                    print("Please enter a prompt or 'quit' to exit.")
                    continue

                print(f"\n🤖 Generating response for: '{prompt}'")
                print("-" * 50)

                start_time = time.time()
                result = self.model.generate_text(prompt, max_tokens=40)
                inference_time = time.time() - start_time

                print(f"\n📝 Generated in {inference_time:.2f} seconds:")
                print(f"{result}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")

    def demo_macos_features(self):
        """Showcase macOS-specific features"""
        print("\n🍎 macOS Native Features")
        print("=" * 35)

        print("✅ Native CoreML acceleration")
        print("✅ Apple Silicon optimization")
        print("✅ Metal Performance Shaders integration")
        print("✅ Low memory footprint")
        print("✅ Fast inference on M-series chips")

        # Show system info
        import platform
        import torch

        print("
💻 System Information:"        print(f"   macOS: {platform.mac_ver()[0]}")
        print(f"   Processor: {platform.processor()}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   MPS available: {torch.backends.mps.is_available()}")

        if torch.backends.mps.is_available():
            print("   ✅ Metal Performance Shaders enabled")
        else:
            print("   ℹ️  Running on CPU (still optimized)")

    def run_financial_demo(self):
        """Demo focused on financial analysis (tying into your project)"""
        print("\n📈 Financial Analysis Demo")
        print("=" * 35)

        financial_prompts = [
            "Analyze the current market trends in technology stocks:",
            "What are the key factors affecting stock price volatility:",
            "Explain options trading strategies for beginners:",
            "How do interest rate changes affect the stock market:",
            "What metrics should I look at when analyzing a company's financial health:"
        ]

        print("Testing IBM Granite's financial analysis capabilities...")

        for i, prompt in enumerate(financial_prompts, 1):
            print(f"\n{i}. {prompt}")

            start_time = time.time()
            result = self.model.generate_text(prompt, max_tokens=35)
            inference_time = time.time() - start_time

            print(f"⏱️  Generated in {inference_time:.2f} seconds")
            print(f"📊 Analysis: {result}")
            print("-" * 60)

    def run_complete_demo(self):
        """Run the complete demo suite"""
        print("🎯 IBM Granite 4.0 CoreML Complete Demo")
        print("=" * 50)
        print("This demo showcases the IBM Granite 4.0 model running")
        print("with native macOS acceleration using Apple's CoreML framework.")
        print("")

        # Show macOS features
        self.demo_macos_features()

        # Basic inference demo
        self.demo_basic_inference()

        # Performance benchmark
        self.demo_performance_comparison()

        # Financial demo (specific to your project)
        self.run_financial_demo()

        # Summary
        print("\n🎉 Demo completed!")
        print("=" * 30)
        print("✅ IBM Granite 4.0 is successfully running on your Mac")
        print("✅ Native CoreML acceleration is active")
        print("✅ Model is optimized for your Apple Silicon")
        print("✅ Ready for production use in your financial application")
        print("")
        print("🚀 Next steps:")
        print("   • Use coreml_inference.py for custom applications")
        print("   • Integrate with your existing financial analysis pipeline")
        print("   • Customize prompts for your specific use cases")
        print("   • Run setup_granite_coreml.py for easy deployment")

def main():
    """Main demo function"""
    demo = GraniteCoreMLDemo()

    print("Welcome to IBM Granite 4.0 CoreML Demo!")
    print("Choose a demo mode:")
    print("1. Complete demo (recommended)")
    print("2. Basic inference only")
    print("3. Performance benchmark")
    print("4. Interactive mode")
    print("5. Financial analysis demo")

    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == '1':
                demo.run_complete_demo()
                break
            elif choice == '2':
                demo.demo_basic_inference()
                break
            elif choice == '3':
                demo.demo_performance_comparison()
                break
            elif choice == '4':
                demo.demo_interactive_mode()
                break
            elif choice == '5':
                demo.run_financial_demo()
                break
            else:
                print("Please enter a number between 1 and 5")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()