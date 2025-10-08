# IBM Granite 4.0 CoreML - Native macOS Interface

This directory contains a complete setup for running IBM Granite 4.0 with native macOS acceleration using Apple's CoreML framework.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
cd model
./run_granite_coreml.sh
```

This script will:
- Check system requirements
- Install Python dependencies
- Convert the model to CoreML format (if needed)
- Provide options to launch the interface

### Option 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip3 install torch transformers accelerate coremltools onnx
   ```

2. **Convert Model to CoreML**
   ```bash
   python3 setup_granite_coreml.py
   ```

3. **Launch Interface**
   ```bash
   # Option A: Python interface
   python3 coreml_inference.py

   # Option B: Native SwiftUI interface
   open GraniteCoreMLApp.swift  # Opens in Xcode
   ```

## 📁 Project Structure

```
model/
├── GraniteCoreMLApp.swift      # Native SwiftUI macOS interface
├── run_granite_coreml.sh       # Automated setup script
├── setup_granite_coreml.py     # Model conversion script
├── convert_granite_to_coreml.py # CoreML conversion utilities
├── coreml_inference.py         # Python inference interface
├── coreml_models/              # Generated CoreML models (after conversion)
│   ├── granite-4.0-h-micro.mlpackage
│   ├── granite-4.0-h-micro_tokenizer/
│   └── granite-4.0-h-micro_config.json
└── requirements.txt            # Python dependencies
```

## 🍎 Native macOS Interface

The `GraniteCoreMLApp.swift` file provides a beautiful, native SwiftUI interface with:

### Features
- **Native Performance**: Uses Apple's CoreML for optimal macOS acceleration
- **Apple Silicon Optimized**: Maximum performance on M1/M2/M3 chips
- **Modern UI**: Clean, responsive SwiftUI interface
- **Real-time Chat**: Interactive conversation interface
- **Model Status**: Visual indicators for model loading and generation
- **Error Handling**: Clear error messages and troubleshooting

### Interface Components
- **Header Bar**: Shows model status and provides clear conversation button
- **Chat Area**: Displays conversation history with message bubbles
- **Input Field**: Text input with send button and keyboard shortcuts
- **Status Indicators**: Visual feedback for model state and generation

## 🔧 Technical Details

### CoreML Benefits
- **Native Acceleration**: Direct integration with macOS hardware
- **Memory Efficiency**: Lower memory usage compared to PyTorch
- **Performance**: Optimized for Apple Neural Engine
- **Privacy**: All processing happens locally on your Mac

### Model Specifications
- **Model**: IBM Granite 4.0 Hybrid Micro
- **Parameters**: ~1.2B parameters
- **Size**: ~1GB on disk (CoreML format)
- **Backend**: CoreML with Metal Performance Shaders

## 🛠️ Development

### Requirements
- **macOS**: 13.0+ (Ventura or later)
- **Python**: 3.8+
- **Xcode**: 14.0+ (for SwiftUI interface)
- **Dependencies**: See `requirements.txt`

### Building the SwiftUI Interface

1. Open `GraniteCoreMLApp.swift` in Xcode
2. Select your development team (or use automatic signing)
3. Build and run the project

### Customization

The SwiftUI interface can be customized by modifying `GraniteCoreMLApp.swift`:

```swift
// Adjust window size
.frame(minWidth: 800, minHeight: 600)

// Modify colors and styling
.background(Color(NSColor.windowBackgroundColor))

// Add new features to GraniteModelManager
class GraniteModelManager: ObservableObject {
    // Add your custom functionality here
}
```

## 🚨 Troubleshooting

### Common Issues

**Model Conversion Fails**
- Check internet connection (model download required)
- Ensure 4GB+ free disk space
- Verify Python dependencies are installed

**SwiftUI Interface Won't Open**
- Install Xcode command line tools: `xcode-select --install`
- Check that you're running on macOS 13.0+
- Verify Xcode is properly installed

**Performance Issues**
- Ensure you're running on Apple Silicon (M1/M2/M3) for best performance
- Close other applications to free up memory
- Check that CoreML model loaded successfully

**Import Errors**
- Run: `pip3 install torch transformers accelerate coremltools onnx`
- Restart terminal/Python session after installation

### Getting Help

1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Ensure sufficient disk space (4GB+ recommended)
4. Try running the Python interface first for debugging

## 📊 Performance Comparison

| Backend | Inference Speed | Memory Usage | Setup Complexity |
|---------|----------------|--------------|------------------|
| CoreML (M1/M2) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| CoreML (Intel) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| PyTorch MPS | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| PyTorch CPU | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🔄 Updates and Maintenance

### Updating the Model
```bash
# Remove old model
rm -rf coreml_models/

# Re-run setup
./run_granite_coreml.sh
```

### Updating Dependencies
```bash
pip3 install --upgrade torch transformers accelerate coremltools onnx
```

## 🎯 Next Steps

1. **Test the Interface**: Run the SwiftUI app and try sample conversations
2. **Customize UI**: Modify colors, layout, and features in `GraniteCoreMLApp.swift`
3. **Add Features**: Integrate additional functionality like file processing or advanced settings
4. **Optimize**: Fine-tune the CoreML model for your specific use case

## 📚 Additional Resources

- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [IBM Granite Model Card](https://huggingface.co/ibm-granite/granite-4.0-h-micro)
- [SwiftUI Documentation](https://developer.apple.com/tutorials/app-dev-training)
- [PyTorch CoreML Conversion](https://coremltools.readme.io/docs/pytorch-conversion)

---

**🎉 Enjoy your native macOS IBM Granite 4.0 experience with CoreML acceleration!**