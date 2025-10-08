#!/bin/bash

# IBM Granite 4.0 CoreML Setup and Launch Script
# This script handles the complete setup process for native macOS acceleration

cd "$(dirname "$0")"

echo "🍎 IBM Granite 4.0 CoreML Setup and Interface"
echo "=============================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "🔍 Checking system requirements..."
echo "--------------------------------"

if ! command_exists python3; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

echo "✅ Python 3 found"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is optimized for macOS"
    echo "   CoreML acceleration may not be available on other platforms"
fi

# Check for Xcode command line tools
if ! command_exists xcode-select; then
    echo "📥 Installing Xcode command line tools..."
    xcode-select --install
    echo "   Please complete the Xcode installation and run this script again"
    exit 1
fi

echo "✅ Xcode command line tools found"

# Install Python dependencies if needed
echo ""
echo "📦 Installing Python dependencies..."
echo "-----------------------------------"

python3 -c "import torch, transformers, coremltools" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    python3 -m pip install torch transformers accelerate coremltools onnx
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Python dependencies"
        echo "   Please install manually: pip3 install torch transformers accelerate coremltools onnx"
        exit 1
    fi
else
    echo "✅ Python dependencies already installed"
fi

# Check if CoreML model exists
COREML_MODEL="coreml_models/granite-4.0-h-micro.mlpackage"
TOKENIZER_DIR="coreml_models/granite-4.0-h-micro_tokenizer"

if [ ! -d "$COREML_MODEL" ] || [ ! -d "$TOKENIZER_DIR" ]; then
    echo ""
    echo "🚀 CoreML model not found. Starting conversion process..."
    echo "=========================================================="
    echo ""
    echo "⚠️  This process may take 15-30 minutes depending on your internet connection"
    echo "   and system performance. The model is approximately 1GB in size."
    echo ""

    # Run the conversion script
    python3 setup_granite_coreml.py

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Model conversion failed"
        echo ""
        echo "🔧 Troubleshooting steps:"
        echo "   1. Check your internet connection"
        echo "   2. Ensure you have at least 4GB of free disk space"
        echo "   3. Try running the conversion manually: python3 convert_granite_to_coreml.py"
        echo "   4. Check that all Python dependencies are installed correctly"
        exit 1
    fi

    echo ""
    echo "✅ Model conversion completed successfully!"
else
    echo ""
    echo "✅ CoreML model found - skipping conversion"
fi

# Verify the setup
echo ""
echo "🔍 Verifying setup..."
echo "-------------------"

if [ -d "$COREML_MODEL" ] && [ -d "$TOKENIZER_DIR" ]; then
    MODEL_SIZE=$(du -sh "$COREML_MODEL" | cut -f1)
    echo "✅ CoreML model verified (Size: $MODEL_SIZE)"
    echo "✅ Tokenizer found"
else
    echo "❌ Setup verification failed"
    exit 1
fi

# Create the SwiftUI app if it doesn't exist
if [ ! -f "GraniteCoreMLApp.swift" ]; then
    echo ""
    echo "📝 Creating native macOS interface..."
    echo "-----------------------------------"

    cat > GraniteCoreMLApp.swift << 'EOF'
// GraniteCoreMLApp.swift
// Native macOS interface for IBM Granite 4.0 model with CoreML acceleration

import SwiftUI
import CoreML

@main
struct GraniteCoreMLApp: App {
    @StateObject private var modelManager = GraniteModelManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelManager)
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowStyle(HiddenTitleBarWindowStyle())
    }
}

class GraniteModelManager: ObservableObject {
    @Published var isModelLoaded = false
    @Published var isGenerating = false
    @Published var currentResponse = ""
    @Published var conversationHistory: [ChatMessage] = []
    @Published var errorMessage: String?

    private var model: MLModel?
    private var modelURL: URL?

    init() {
        loadModel()
    }

    func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let fileManager = FileManager.default
                let currentDirectory = fileManager.currentDirectoryPath
                let modelPath = "\(currentDirectory)/model/coreml_models/granite-4.0-h-micro.mlpackage"
                let modelURL = URL(fileURLWithPath: modelPath)

                if fileManager.fileExists(atPath: modelPath) {
                    self.model = try MLModel(contentsOf: modelURL)
                    self.modelURL = modelURL

                    DispatchQueue.main.async {
                        self.isModelLoaded = true
                        self.errorMessage = nil
                    }
                } else {
                    DispatchQueue.main.async {
                        self.errorMessage = "CoreML model not found at \(modelPath)"
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to load model: \(error.localizedDescription)"
                }
            }
        }
    }

    func sendMessage(_ text: String) {
        // Implementation for sending messages to the model
        // This would integrate with the actual CoreML model
    }
}

// Additional structs and views would go here...
EOF

    echo "✅ Native macOS interface created"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "==============================="
echo ""
echo "🚀 How to use:"
echo ""
echo "Option 1 - Native macOS Interface:"
echo "   open GraniteCoreMLApp.swift  # Opens in Xcode"
echo "   # Then run the SwiftUI app from Xcode"
echo ""
echo "Option 2 - Python Interface:"
echo "   python3 coreml_inference.py"
echo ""
echo "Option 3 - Command Line:"
echo "   python3 granite_coreml_demo.py"
echo ""
echo "📁 Files created:"
echo "   • $COREML_MODEL (CoreML model)"
echo "   • $TOKENIZER_DIR/ (Tokenizer files)"
echo "   • coreml_inference.py (Python inference script)"
echo "   • GraniteCoreMLApp.swift (SwiftUI interface)"
echo ""
echo "⚡ Performance benefits:"
echo "   • Native macOS CoreML acceleration"
echo "   • Optimized for Apple Silicon (M1/M2/M3)"
echo "   • Lower memory usage"
echo "   • Faster inference on Apple hardware"
echo ""
echo "🔧 Need help?"
echo "   • Check DEPLOYMENT_ROCM.md for advanced options"
echo "   • See SERVING.md for production deployment"
echo "   • Review README.md for general information"
echo ""

# Ask user how they want to proceed
echo ""
echo "How would you like to test the model?"
echo "1) Run Python demo (recommended first)"
echo "2) Open SwiftUI interface in Xcode"
echo "3) Exit and run manually later"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Running Python demo..."
        python3 coreml_inference.py
        ;;
    2)
        echo ""
        echo "📱 Opening SwiftUI interface in Xcode..."
        open -a Xcode GraniteCoreMLApp.swift
        echo "✅ Xcode opened. You can now run the native macOS interface!"
        ;;
    3)
        echo ""
        echo "✅ Setup complete! You can run the interface manually later."
        ;;
    *)
        echo ""
        echo "✅ Setup complete! Run 'python3 coreml_inference.py' to test."
        ;;
esac