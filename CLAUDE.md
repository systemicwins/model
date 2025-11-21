# CLAUDE.md - AI Assistant Guide for Financial Prediction Model

> **Last Updated**: 2025-11-21
> **Purpose**: Guide for AI assistants (Claude, GPT-4, etc.) working with this codebase

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Build System & Dependencies](#build-system--dependencies)
4. [Development Workflows](#development-workflows)
5. [Code Conventions & Standards](#code-conventions--standards)
6. [Architecture Deep Dive](#architecture-deep-dive)
7. [Testing Strategy](#testing-strategy)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)
10. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## Project Overview

### What This Project Does

This is a **state-of-the-art financial prediction system** combining:
- **Mamba2 State Space Models (SSM)** - Linear O(n) complexity for long sequences
- **Sparse Attention** - Content-aware attention for financial patterns
- **Matryoshka Encoding** - Multi-dimensional adaptive embeddings (64D to 1536D)
- **Adaptive Computational Time (ACT)** - Dynamic resource allocation with Q-learning
- **Context-First Architecture** - Economic indicators (FRED data) establish prediction foundation

### Key Innovation

The model uses a **context-first financial prediction approach**:
1. **Economic Context Foundation** - FRED economic indicators establish reality framework
2. **Market Interpretation Layer** - Market data interpreted within economic context
3. **Adaptive Computation** - ACT dynamically allocates resources based on sample complexity

### Technology Stack

- **Language**: C++17
- **Build System**: CMake 3.14+
- **Linear Algebra**: Eigen3
- **JSON**: nlohmann/json (FetchContent)
- **HTTP**: libcurl
- **Parallelization**: OpenMP
- **Optional GPU**: HIP/ROCm (AMD GPUs)
- **Python**: Supporting scripts for data fetching and demos

### Performance Characteristics

| Metric | Traditional Model | This Model | Improvement |
|--------|------------------|------------|-------------|
| **Time Complexity** | O(n²) | O(n) | Linear scaling |
| **Context Length** | 2K tokens | 100K+ tokens | 50x longer |
| **Memory Usage** | Full precision | 67-96% reduction | Adaptive |
| **Training Speed** | Baseline | 5-10x faster | Parallel computation |
| **Inference Speed** | Standard | 10-200x faster | ACT + Linear |
| **Computational Steps** | Fixed 16 | 2.33 avg | 71% reduction |

---

## Repository Structure

```
model/
├── .github/
│   └── workflows/
│       ├── build.yml           # General build workflow (all branches)
│       └── model-build.yml     # Model-specific build (main branch only)
│
├── include/                    # Header files
│   ├── mamba.h                # Main Mamba model interface
│   ├── transformer.h          # Transformer/SSM layer definitions
│   ├── sparse_attention.h     # SSM-guided sparse attention
│   ├── hybrid_attention.h     # Hybrid SSM + attention fusion
│   ├── act_controller.h       # Adaptive Computational Time controller
│   ├── matryoshka_encoder.h   # Multi-dimensional encoding
│   ├── context_first_architecture.h  # FRED-first context reasoning
│   ├── tokenizer.h            # Financial vocabulary tokenizer
│   ├── ticker_vocabulary.h    # Stock ticker integration
│   ├── layer_norm.h           # Layer normalization
│   ├── feedforward.h          # Feed-forward networks
│   ├── tensor_ops.h           # Tensor operations & utilities
│   ├── compact_positional_encoding.h      # Position encoding (16-bit)
│   └── compact_positional_encoding_f32.h  # Position encoding (32-bit)
│
├── src/                       # Source implementations
│   ├── main.cpp              # Entry point with CLI commands
│   ├── mamba.cpp             # Mamba model implementation
│   ├── transformer.cpp       # Transformer/SSM layers
│   ├── sparse_attention.cpp  # Sparse attention implementation
│   ├── hybrid_attention.cpp  # Hybrid attention (commented out)
│   ├── act_controller.cpp    # ACT implementation (commented out)
│   ├── attention.cpp         # Base attention mechanisms
│   ├── matryoshka_encoder.cpp     # Matryoshka encoding
│   ├── matryoshka_trainer.cpp     # Training algorithms
│   ├── matryoshka_benchmark.cpp   # Performance benchmarks
│   ├── tokenizer.cpp              # Tokenizer with financial vocab
│   ├── layer_norm.cpp             # Normalization
│   ├── feedforward.cpp            # Feed-forward networks
│   ├── tensor_ops.cpp             # Tensor operations
│   ├── financial_prediction_demo.cpp  # Demo (commented out)
│   ├── attention_benchmark.cpp        # Benchmarks (commented out)
│   ├── hybrid_attention_benchmark.cpp # Benchmarks (commented out)
│   ├── layer_comparison_benchmark.cpp # Benchmarks (commented out)
│   └── model_exporter.cpp             # Model export utilities
│
├── data/                     # Data files and scripts
│   ├── FED/                 # Federal Reserve economic data
│   ├── SEC/                 # SEC filing data
│   ├── node_modules/        # Node.js dependencies for data fetching
│   ├── trading_symbols.txt  # Stock ticker vocabulary (414+ symbols)
│   ├── sec_filing_vocab.txt # SEC-specific terms
│   └── financial_vocab.txt  # Financial domain vocabulary
│
├── db/                      # Database schemas
│   └── migrate/            # Migration scripts
│
├── sql/                     # SQL queries
│
├── scripts/                 # Utility scripts
│   ├── fetch_alpaca_symbols.py
│   └── generate_full_symbol_list.py
│
├── python/                  # Python utilities
│   └── (various demo and analysis scripts)
│
├── build/                   # Build artifacts (generated)
│
├── CMakeLists.txt          # Main build configuration
├── DESIGN.txt              # Detailed architecture documentation
├── README.md               # User-facing documentation
├── MATRYOSHKA.md           # Matryoshka encoding documentation
├── RECURSION.md            # Recursive architecture documentation
├── SPARSE.md               # Sparse attention documentation
├── SERVING.md              # Model serving documentation
├── requirements.txt        # Python dependencies
└── test_sparse_attention.cpp  # Sparse attention test executable
```

### Important File Relationships

- **Header/Source Pairs**: Each `.h` file in `include/` typically has a corresponding `.cpp` in `src/`
- **Commented Out Files**: Some files are commented out in CMakeLists.txt due to build issues:
  - `hybrid_attention.cpp` - compute_reward access issues
  - `act_controller.cpp` - Eigen compatibility issues
  - `financial_prediction_demo.cpp` - type conflicts
  - Matryoshka benchmark/trainer - dependency issues
- **Test Files**: Standalone test executables like `test_sparse_attention.cpp` include required sources directly

---

## Build System & Dependencies

### CMake Configuration

The project uses CMake 3.14+ with the following key features:

#### Required Dependencies

1. **Eigen3** (Linear algebra)
   - Version: 3.4+
   - Detection order:
     1. `EIGEN3_INCLUDE_DIR` environment variable
     2. `/tmp/eigen-3.4.0` (local development default)
     3. System paths: `/usr/include/eigen3`, `/usr/local/include/eigen3`, `/opt/homebrew/include/eigen3`
   - Install: `sudo apt-get install libeigen3-dev` (Ubuntu/Debian)

2. **libcurl** (HTTP requests for FRED data)
   - Install: `sudo apt-get install libcurl4-openssl-dev`

3. **nlohmann/json** (JSON parsing)
   - Fetched automatically via CMake FetchContent
   - Version: 3.11.3

4. **Threads** (POSIX threads)
   - Usually available by default on Unix systems

#### Optional Dependencies

1. **OpenMP** (Parallel computation)
   - Detected automatically
   - Falls back to serial execution if not available

2. **HIP/ROCm** (AMD GPU acceleration)
   - Enable with: `-DENABLE_HIP=ON`
   - Requires ROCm installation at `/opt/rocm` or `/opt/rocm-7.0.0`
   - Libraries: hipblas, hipfft, rocrand (optional)

### Build Commands

#### Standard CPU Build (Recommended)

```bash
# From project root
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

#### With Specific Eigen3 Path

```bash
export EIGEN3_INCLUDE_DIR=/path/to/eigen3
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

#### With HIP/ROCm GPU Support

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=ON
cmake --build . --parallel
```

#### Debug Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --parallel
```

### Build Targets

1. **mamba_model** - Main executable
   - Sources: Everything in CMakeLists.txt `SOURCES` variable
   - Defines: `MAX_SEQUENCE_LENGTH=100000`

2. **test_sparse_attention** - Sparse attention test
   - Sources: `test_sparse_attention.cpp` + required dependencies
   - Defines: `MAX_SEQUENCE_LENGTH=100000`

3. **matryoshka_benchmark** - Matryoshka encoding benchmark
   - Currently commented out due to build issues

### Compiler Flags

#### Release Mode
- `-O3` - Maximum optimization
- `-march=native` - CPU-specific optimizations
- `-ffast-math` - Fast floating-point math

#### Warnings (All Modes)
- `-Wall -Wextra -Wpedantic` - Enable most warnings
- `-Wno-array-bounds` - Suppress Eigen3 false positives
- `-Wno-maybe-uninitialized` - Suppress Eigen3 template warnings
- Note: `-Werror` is disabled to allow warnings without failing build

#### Standards
- C++17 required
- No compiler extensions (`CMAKE_CXX_EXTENSIONS OFF`)

---

## Development Workflows

### CI/CD Pipeline

The project uses GitHub Actions with two workflows:

#### 1. General Build Workflow (`build.yml`)
- **Triggers**: All branches, all pushes and PRs
- **Steps**:
  1. Checkout code
  2. Install dependencies (cmake, build-essential, libcurl4-openssl-dev, libeigen3-dev)
  3. Configure CMake (Release mode)
  4. Build with parallel compilation
  5. Run ctest (if available)

#### 2. Model-Specific Build (`model-build.yml`)
- **Triggers**:
  - Push to `main` branch
  - PR to `main` branch
  - Only when files in `model/` or workflow file change
- **Steps**:
  1. Checkout with recursive submodules
  2. Install dependencies
  3. Configure CMake in `model/` subdirectory
  4. Build
  5. Test executable existence and run with `--version` flag

### Git Workflow

#### Branch Naming
- Feature branches: `claude/claude-md-<session-id>`
- Main branch: `main` (no direct pushes, PR required)

#### Commit Guidelines
- Clear, descriptive commit messages
- Reference related issues if applicable
- Keep commits atomic and focused

#### Before Committing
1. **Build locally**: Ensure `cmake --build build` succeeds
2. **Check warnings**: Review compiler warnings (though not blocking)
3. **Test basic functionality**: Run `./build/mamba_model test`
4. **Review changes**: Use `git diff` to verify modifications

### Adding New Files

#### Adding a Header File
1. Create file in `include/` directory
2. Add include guards: `#ifndef FILENAME_H` / `#define FILENAME_H` / `#endif`
3. Include in relevant source files
4. Add to `HEADERS` list in CMakeLists.txt (for documentation)

#### Adding a Source File
1. Create file in `src/` directory
2. Include corresponding header from `include/`
3. Add to `SOURCES` list in CMakeLists.txt
4. If it's a standalone executable, create separate `add_executable()` target

#### Adding a Test
1. Create test file (e.g., `test_feature.cpp`)
2. Include required headers and source files
3. Add as separate executable in CMakeLists.txt:
```cmake
set(TEST_SOURCES
    test_feature.cpp
    src/required_dependency.cpp
    ...
)
add_executable(test_feature ${TEST_SOURCES})
target_compile_definitions(test_feature PRIVATE MAX_SEQUENCE_LENGTH=100000)
target_include_directories(test_feature PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(test_feature PRIVATE ${MAMBA_LINK_LIBS})
```

---

## Code Conventions & Standards

### C++ Style Guidelines

#### Header Guards
Always use traditional include guards (not `#pragma once`):
```cpp
#ifndef COMPONENT_NAME_H
#define COMPONENT_NAME_H

// ... header content ...

#endif // COMPONENT_NAME_H
```

#### Namespaces
- Primary namespace: `mamba::` for Mamba-specific code
- Additional namespaces: `transformer::`, `matryoshka::`, `act::`
- Use `using` declarations for type aliases:
```cpp
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;
using Scalar = float;
```

#### Naming Conventions
- **Classes**: PascalCase (`MambaModel`, `SparseAttention`)
- **Functions**: snake_case (`forward()`, `compute_attention()`)
- **Variables**: snake_case (`embed_dim`, `state_dim`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_SEQUENCE_LENGTH`)
- **Member variables**: snake_case with trailing underscore (`state_dim_`, `num_layers_`)
- **Namespaces**: lowercase (`mamba`, `transformer`)

#### File Organization
1. **Header file structure**:
```cpp
// Copyright/license (if applicable)

#ifndef HEADER_NAME_H
#define HEADER_NAME_H

// System includes
#include <vector>
#include <memory>

// External library includes
#include <Eigen/Dense>

// Project includes
#include "other_header.h"

namespace project_namespace {

// Forward declarations
class SomeClass;

// Type aliases
using Matrix = Eigen::MatrixXf;

// Configuration structs
struct Config {
    int param1;
    float param2;
};

// Main class declaration
class MainClass {
public:
    // Constructors
    MainClass();
    explicit MainClass(const Config& config);

    // Public methods
    void public_method();

private:
    // Private members
    int member_var_;

    // Private methods
    void private_method();
};

} // namespace project_namespace

#endif // HEADER_NAME_H
```

2. **Source file structure**:
```cpp
#include "header_name.h"

// Additional includes needed for implementation
#include <iostream>
#include "other_dependencies.h"

namespace project_namespace {

// Implementation
MainClass::MainClass() : member_var_(0) {
    // Constructor implementation
}

void MainClass::public_method() {
    // Method implementation
}

} // namespace project_namespace
```

#### Comments
- Use `//` for single-line comments
- Use `/* */` for multi-line comments
- Document complex algorithms and financial domain logic
- Example:
```cpp
// Compute attention scores using sparse pattern
// Financial data has sparse important events, so we use selective attention
Matrix scores = compute_sparse_attention(query, key);
```

#### Error Handling
- Prefer exceptions for error conditions
- Use `std::runtime_error` or custom exception classes
- Example:
```cpp
if (input.rows() == 0) {
    throw std::runtime_error("Input matrix cannot be empty");
}
```

### Eigen3 Usage Patterns

#### Matrix Types
- Primary type: `Eigen::MatrixXf` (float precision)
- Vector type: `Eigen::VectorXf`
- Use `.rows()` and `.cols()` for dimensions
- Use `.resize()` for dynamic allocation

#### Common Operations
```cpp
// Matrix creation
Matrix m = Matrix::Zero(rows, cols);
Matrix m = Matrix::Random(rows, cols);
Matrix m = Matrix::Identity(size, size);

// Element access
float val = m(row, col);  // Read
m(row, col) = val;        // Write

// Matrix operations
Matrix result = m1 * m2;           // Multiplication
Matrix result = m1 + m2;           // Addition
Matrix result = m.transpose();     // Transpose
Matrix result = m.array() * 2.0f;  // Element-wise scalar multiply

// Reductions
float sum = m.sum();
float mean = m.mean();
float max = m.maxCoeff();
Vector row_mean = m.rowwise().mean();
```

#### Performance Tips
- Use `.noalias()` for assignment without temporary: `C.noalias() = A * B;`
- Prefer in-place operations: `m.array() *= 2.0f;`
- Use block operations: `m.block(i, j, rows, cols)`
- Enable vectorization with `-march=native`

### Financial Domain Conventions

#### FRED Data Integration
- Economic indicators should be processed **before** market data
- Context-first architecture: Economic context → Market interpretation → Prediction
- Common FRED indicators:
  - `DFF` - Federal Funds Rate
  - `DGS10` - 10-Year Treasury Rate
  - `GDP` - Gross Domestic Product
  - `UNRATE` - Unemployment Rate
  - `CPIAUCSL` - Consumer Price Index

#### Financial Tokenizer
- Base vocabulary: 50,000 tokens
- Financial terms: ~500 specialized terms
- SEC filing terms: ~800 regulatory terms
- Trading symbols: 414+ stock tickers (in `data/trading_symbols.txt`)
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<cls>`, `<sep>`, `<mask>`

#### Financial Head Specializations
When using multi-head SSM or attention:
- **FINANCIAL_PRICE**: Log returns, volatility-aware, trend-following
- **FINANCIAL_VOLUME**: Sparsity-aware, burst detection
- **FINANCIAL_SENTIMENT**: Extended context, polarity-aware, drift detection
- **FINANCIAL_TECHNICAL**: Indicator normalization, crossover detection

---

## Architecture Deep Dive

### Core Components

#### 1. Mamba2 State Space Model (SSM)

**Location**: `include/mamba.h`, `src/mamba.cpp`

**Key Innovation**: Linear O(n) complexity instead of quadratic O(n²)

**State Evolution**:
```cpp
// Simplified SSM equation
h_t = A_scalar * h_{t-1} + B * x_t    // State update
y_t = C * h_t + D * x_t                // Output projection
```

**Mamba2 Improvements over Mamba1**:
- Scalar A matrix per head (instead of full matrix)
- Parallel parameter computation (5-10x faster training)
- Multi-head SSM support (2, 4, 8+ heads)
- Better hardware utilization with OpenMP

**Configuration**:
```cpp
struct MambaConfig {
    int embed_dim = 1536;           // Embedding dimension
    int state_dim = 128;            // SSM state dimension
    int num_layers = 6;             // Number of Mamba layers
    int max_seq_length = 100000;    // Max sequence length
    bool use_selective_ssm = true;  // Selective state space
};
```

#### 2. Sparse Attention

**Location**: `include/sparse_attention.h`, `src/sparse_attention.cpp`

**Purpose**: Content-aware attention for financial data with sparse important events

**Complexity**: O(n√n) for sparse patterns (vs O(n²) for dense attention)

**Attention Patterns**:
- Local attention (sliding window)
- Global attention (to key tokens)
- Random attention (for long-range connections)
- Financial-specific patterns (price movements, volume spikes)

**Configuration**:
```cpp
struct SparseAttentionConfig {
    float high_change_threshold = 0.8f;    // Threshold for dense attention
    bool is_financial_data = true;         // Enable financial specializations
    bool enable_adaptive_thresholds = true; // Dynamic threshold adjustment
};
```

#### 3. Adaptive Computational Time (ACT)

**Location**: `include/act_controller.h`, `src/act_controller.cpp` (commented out in build)

**Purpose**: Dynamic computation stopping based on confidence

**Q-Learning Components**:
- State features: Statistical properties of current state
- Q-network: Simple linear network (state_features → Q-values)
- Actions: Continue (0) or Halt (1)
- Reward: Efficiency + Confidence - Early halt penalty

**Decision Process**:
1. Extract state features (mean, std, max, min, row/col statistics)
2. Compute Q-values for continue/halt actions
3. Select action (epsilon-greedy during training, greedy during inference)
4. Compute confidence score
5. Update Q-network with temporal difference learning

**Performance**:
- 71% computational reduction
- 83% early halting rate
- Average 2.33 steps (vs fixed 16)

#### 4. Matryoshka Encoding

**Location**: `include/matryoshka_encoder.h`, `src/matryoshka_encoder.cpp`

**Purpose**: Multi-dimensional embeddings for different accuracy/speed trade-offs

**Dimensions**: 64, 128, 256, 512, 768, 1024, 1536

**Key Feature**: Single encoding, multiple dimensions available at runtime

**Use Cases**:
- 64D: Ultra-fast search, 96% memory reduction
- 512D: Balanced speed/accuracy, 67% memory reduction
- 1536D: Full fidelity, maximum accuracy

**Training**: Progressive curriculum learning, starts with 64D and gradually increases

#### 5. Context-First Architecture

**Location**: `include/context_first_architecture.h`

**Design Philosophy**: Economic context establishes foundation, market data interpreted within that context

**Processing Order**:
1. Economic Foundation (FRED data) → Economic context
2. Market Interpreter (market data + economic context) → Market interpretation
3. Cross-Domain Refinement → Final prediction

**Rationale**: Markets operate within economic reality, not in isolation

### Data Flow

```
Input Text
    ↓
[Tokenizer] - 50K vocab + financial terms + ticker symbols
    ↓
Token IDs
    ↓
[Embedding Layer] - 50000 × 1536 matrix
    ↓
Token Embeddings (seq_len × 1536)
    ↓
[Matryoshka Encoder] - Multi-scale representations with positional encoding
    ↓
Matryoshka-encoded Embeddings
    ↓
[Mamba2 Encoder] - 6 layers, SSM + convolution + gated MLP
    │   ├── 1D Convolution (local context)
    │   ├── Multi-Head SSM (temporal modeling)
    │   ├── Gated MLP (non-linear transformations)
    │   └── Layer Norm (stabilization)
    ↓
[Sparse Attention] - Content-aware sparse attention (optional)
    ↓
[ACT Controller] - Dynamic computation stopping (optional)
    ↓
[Pooling] - Mean/Max/First/Last token
    ↓
Final Embedding (1536D or reduced via Matryoshka)
    ↓
Output
```

### Memory Footprint

| Component | Size (Approximate) |
|-----------|-------------------|
| Embedding Matrix | 300 MB (50K × 1.5K × 4B) |
| Mamba2 SSM (per layer, 8 heads) | ~96 MB |
| Convolution (per layer) | ~36 MB |
| Gated MLP (per layer) | ~54 MB |
| **Total (6 layers)** | **~876 MB** |
| Matryoshka Extensions | +50 MB |
| **Final Model Size** | **~926 MB** |

---

## Testing Strategy

### Unit Tests

Currently, the project has limited formal unit tests. The main test is:

**test_sparse_attention**: Tests sparse attention implementation
```bash
./build/test_sparse_attention
```

### Manual Testing

The main executable provides test commands:

```bash
# Run basic test with sample data
./build/mamba_model test

# Encode text
./build/mamba_model encode "Your text here"

# Process text through model
./build/mamba_model process "Your text here"

# Batch processing
./build/mamba_model batch "Text 1" "Text 2" "Text 3"

# Run benchmark
./build/mamba_model benchmark
```

### Testing Guidelines for AI Assistants

When modifying code:

1. **Build Test**: Always compile after changes
   ```bash
   cmake --build build --parallel
   ```

2. **Basic Functionality Test**: Run the test command
   ```bash
   ./build/mamba_model test
   ```

3. **Component-Specific Tests**: If modifying specific components, run relevant tests
   ```bash
   ./build/test_sparse_attention  # For sparse attention changes
   ```

4. **Integration Test**: Test end-to-end with real financial text
   ```bash
   ./build/mamba_model process "AAPL reported strong Q3 earnings with EBITDA of 25.5B"
   ```

5. **Check for Warnings**: Review compiler warnings (though not blocking)

### What to Test When Making Changes

| Change Type | Tests to Run |
|------------|--------------|
| Header file only | Build test |
| Source file (implementation) | Build + functionality test |
| CMakeLists.txt | Clean build from scratch |
| New feature | Build + manual test with relevant input |
| Bug fix | Build + test case that reproduced bug |
| Performance optimization | Build + benchmark |

---

## Common Tasks

### Adding a New Component

1. **Create header file** (`include/my_component.h`):
```cpp
#ifndef MY_COMPONENT_H
#define MY_COMPONENT_H

#include <Eigen/Dense>

namespace mamba {

class MyComponent {
public:
    MyComponent();
    void process(const Matrix& input);

private:
    int internal_state_;
};

} // namespace mamba

#endif // MY_COMPONENT_H
```

2. **Create source file** (`src/my_component.cpp`):
```cpp
#include "my_component.h"

namespace mamba {

MyComponent::MyComponent() : internal_state_(0) {
    // Constructor
}

void MyComponent::process(const Matrix& input) {
    // Implementation
}

} // namespace mamba
```

3. **Add to CMakeLists.txt**:
```cmake
set(SOURCES
    # ... existing sources ...
    src/my_component.cpp
)

set(HEADERS
    # ... existing headers ...
    include/my_component.h
)
```

4. **Build and test**:
```bash
cmake --build build --parallel
./build/mamba_model test
```

### Modifying Existing Components

1. **Understand dependencies**: Check which files include the header
   ```bash
   grep -r "my_component.h" include/ src/
   ```

2. **Make changes**: Edit header and/or source file

3. **Rebuild**:
   ```bash
   cmake --build build --parallel
   ```

4. **Test**: Run relevant tests

### Adding Dependencies

1. **System library** (e.g., new apt package):
   - Add to GitHub Actions workflows (`.github/workflows/*.yml`)
   - Update CMakeLists.txt with `find_package()` or `pkg_check_modules()`
   - Document in this file

2. **Header-only library**:
   - Add to `include/` or as git submodule
   - Update `target_include_directories()` in CMakeLists.txt

3. **FetchContent library** (recommended for external deps):
```cmake
include(FetchContent)
FetchContent_Declare(
    library_name
    URL https://github.com/user/repo/archive/version.tar.gz
)
FetchContent_MakeAvailable(library_name)
target_link_libraries(mamba_model PRIVATE library_name::library_name)
```

### Debugging Build Issues

1. **Clean build**:
   ```bash
   rm -rf build
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   cmake --build . --verbose
   ```

2. **Check CMake cache**:
   ```bash
   cmake -L build  # List cache variables
   cmake -LAH build  # List with help text
   ```

3. **Eigen3 not found**:
   ```bash
   # Option 1: Set environment variable
   export EIGEN3_INCLUDE_DIR=/path/to/eigen3

   # Option 2: Install system package
   sudo apt-get install libeigen3-dev

   # Option 3: Use CMake variable
   cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen3
   ```

4. **Link errors**:
   - Check that all required sources are in CMakeLists.txt
   - Verify library link order in `target_link_libraries()`
   - For undefined references, ensure forward declarations have implementations

### Performance Profiling

1. **Build with profiling**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build . --parallel
   ```

2. **Run with profiler**:
   ```bash
   # Using perf (Linux)
   perf record ./build/mamba_model benchmark
   perf report

   # Using gprof
   gprof ./build/mamba_model gmon.out > analysis.txt
   ```

3. **Benchmark specific components**:
   ```bash
   ./build/mamba_model benchmark  # Overall benchmark
   # Or create custom benchmark in src/
   ```

---

## Troubleshooting

### Common Build Errors

#### 1. Eigen3 Not Found

**Error**: `Could not find Eigen3`

**Solutions**:
```bash
# Install system package
sudo apt-get install libeigen3-dev

# Or download manually
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzf eigen-3.4.0.tar.gz
export EIGEN3_INCLUDE_DIR=$(pwd)/eigen-3.4.0
```

#### 2. CURL Not Found

**Error**: `Could not find CURL`

**Solution**:
```bash
sudo apt-get install libcurl4-openssl-dev
```

#### 3. OpenMP Not Found

**Error**: `Could not find OpenMP`

**This is not fatal** - the project will build without OpenMP (serial execution only)

**To enable OpenMP**:
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS
brew install libomp
```

#### 4. Undefined Reference Errors

**Error**: `undefined reference to 'SomeClass::method()'`

**Likely causes**:
- Missing source file in CMakeLists.txt `SOURCES`
- Missing implementation of declared method
- Incorrect namespace or naming

**Solution**:
1. Check that source file is in CMakeLists.txt
2. Verify method is implemented in source file
3. Check namespace matches declaration

#### 5. Compilation Warnings

**Eigen-related warnings**: These are expected and suppressed
- `-Wno-array-bounds` - False positives in Eigen AVX code
- `-Wno-maybe-uninitialized` - False positives in Eigen templates

**Other warnings**: Should be investigated and fixed when possible

### Runtime Errors

#### 1. Segmentation Fault

**Common causes**:
- Invalid matrix dimensions
- Accessing out-of-bounds elements
- Null pointer dereference

**Debugging**:
```bash
# Build with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --parallel

# Run with debugger
gdb ./build/mamba_model
(gdb) run test
(gdb) backtrace  # After crash
```

#### 2. Memory Issues

**Symptoms**: Crashes with large sequences, out-of-memory errors

**Solutions**:
- Reduce `MAX_SEQUENCE_LENGTH` in CMakeLists.txt
- Use smaller batch sizes
- Enable Matryoshka encoding at lower dimensions
- Monitor memory with `top` or `htop`

#### 3. Numerical Instability

**Symptoms**: NaN or Inf values in output

**Common causes**:
- Division by zero
- Overflow in exponentials (attention softmax)
- Underflow in very small probabilities

**Debugging**:
- Add checks: `if (!std::isfinite(value)) throw std::runtime_error("NaN detected");`
- Enable assertions in debug builds
- Check input data ranges

### Performance Issues

#### 1. Slow Compilation

**Solutions**:
- Use parallel build: `cmake --build build --parallel`
- Reduce template instantiations
- Split large source files

#### 2. Slow Runtime

**Check**:
- Build type: Should be Release (`-DCMAKE_BUILD_TYPE=Release`)
- Optimization flags: `-O3 -march=native` should be enabled
- OpenMP: Should be found and used
- Sequence length: Very long sequences are slow even with O(n) complexity

**Profile** to find bottlenecks:
```bash
perf record ./build/mamba_model benchmark
perf report
```

---

## AI Assistant Guidelines

### When Working with This Codebase

#### 1. Understanding Context

Before making changes:
- Read the relevant documentation files (README.md, DESIGN.txt, MATRYOSHKA.md, etc.)
- Understand the financial domain context (this is a financial prediction model)
- Review related code sections, not just the immediate change area
- Check CMakeLists.txt to understand build dependencies

#### 2. Making Changes

**Small changes** (single file):
- Edit the file
- Rebuild
- Test basic functionality

**Medium changes** (multiple files, new feature):
- Plan the change across files
- Update headers first
- Implement in source files
- Update CMakeLists.txt if adding new files
- Build and test incrementally

**Large changes** (architecture modification):
- Discuss design first (with user or in comments)
- Break into smaller phases
- Document the change plan
- Implement phase by phase
- Test thoroughly at each phase

#### 3. Code Quality Standards

**Must have**:
- ✅ Correct C++17 syntax
- ✅ Proper include guards
- ✅ Namespace usage
- ✅ Basic error handling
- ✅ Builds without errors

**Should have**:
- ✅ Clear variable names
- ✅ Comments for complex logic
- ✅ Consistent style with existing code
- ✅ Reasonable performance
- ✅ No compiler warnings (if possible)

**Nice to have**:
- ✅ Comprehensive comments
- ✅ Unit tests
- ✅ Performance optimizations
- ✅ Example usage

#### 4. Financial Domain Awareness

This is a **financial prediction model**, so:
- Understand that FRED data represents economic indicators
- Know that market data should be interpreted within economic context
- Recognize financial terminology (EBITDA, earnings, volatility, etc.)
- Respect the context-first architecture design
- Be aware of numerical precision requirements for financial calculations

#### 5. Communication with User

**When suggesting changes**:
- Explain the purpose and impact
- Mention affected files
- Note any breaking changes
- Suggest testing approach

**When encountering issues**:
- Clearly describe the problem
- Provide relevant error messages
- Suggest multiple solutions if possible
- Be honest about limitations or uncertainty

**When explaining code**:
- Use domain-appropriate language
- Reference documentation when available
- Explain both "what" and "why"
- Provide examples when helpful

#### 6. Testing Mindset

Always consider:
- Will this build?
- Will this work with typical inputs?
- What edge cases might break this?
- How can we verify this works?

Recommend testing:
- After every significant change
- Before committing
- With realistic financial data when possible

#### 7. Performance Awareness

This model is optimized for performance:
- O(n) complexity is a key feature (don't introduce O(n²) operations)
- Memory efficiency matters (large sequences)
- Eigen3 operations should be vectorizable
- Consider using OpenMP for parallelization
- ACT and Matryoshka are about adaptive computation

#### 8. Documentation Discipline

When making changes:
- Update comments in code
- Update relevant .md files if changing architecture
- Update this CLAUDE.md if changing conventions or adding workflows
- Add TODOs for incomplete features: `// TODO: Implement feature X`

### Examples of Good Assistant Behavior

#### Example 1: Adding a New Feature

```
User: "Add support for custom attention patterns"

Good Response:
"I'll add custom attention pattern support. This will involve:

1. Updating include/sparse_attention.h to add a new AttentionPattern enum
2. Implementing the pattern in src/sparse_attention.cpp
3. Adding configuration option in SparseAttentionConfig struct
4. No changes needed to CMakeLists.txt (existing files)

The change maintains O(n√n) complexity and follows the existing
sparse attention design. Would you like me to proceed?"
```

#### Example 2: Debugging a Build Error

```
User: "Build fails with undefined reference to MatryoshkaEncoder::encode"

Good Response:
"This error suggests the method declaration exists but implementation
is missing or not linked. Let me check:

1. Is the method declared in include/matryoshka_encoder.h? [checks]
2. Is it implemented in src/matryoshka_encoder.cpp? [checks]
3. Is matryoshka_encoder.cpp in CMakeLists.txt SOURCES? [checks]

I found that matryoshka_encoder.cpp is commented out in CMakeLists.txt
(line 143) due to 'save_matrix issues'. This is causing the linker error.

Solutions:
A) Uncomment it and fix the save_matrix issue
B) Remove the declaration from the header if not needed
C) Provide a stub implementation

Which approach would you prefer?"
```

#### Example 3: Explaining Complex Code

```
User: "How does the ACT controller work?"

Good Response:
"The ACT (Adaptive Computational Time) controller dynamically decides
when to stop computation based on confidence scores. Here's the flow:

1. **State Features** (act_controller.cpp:123): Extracts statistical
   properties (mean, std, max, min) from the current state

2. **Q-Network** (act_controller.cpp:156): Simple linear network that
   predicts Q-values for two actions: Continue (0) or Halt (1)

3. **Action Selection** (act_controller.cpp:178):
   - Training: Epsilon-greedy (10% random exploration)
   - Inference: Greedy (best Q-value)

4. **Confidence Computation** (act_controller.cpp:203): Measures state
   stability and similarity to previous state

5. **Q-Learning Update** (act_controller.cpp:245): Uses temporal
   difference learning to improve the stopping policy

Result: 71% computational reduction (2.33 avg steps vs 16 fixed steps)

The code is in include/act_controller.h and src/act_controller.cpp,
though currently commented out in CMakeLists.txt due to Eigen issues."
```

### Anti-Patterns to Avoid

❌ **Don't**: Make changes without understanding the context
✅ **Do**: Read relevant documentation first

❌ **Don't**: Break existing functionality silently
✅ **Do**: Test before and after changes

❌ **Don't**: Ignore compiler warnings
✅ **Do**: Understand and fix warnings when possible

❌ **Don't**: Introduce O(n²) complexity in performance-critical paths
✅ **Do**: Maintain O(n) or O(n√n) complexity

❌ **Don't**: Add dependencies without discussion
✅ **Do**: Propose and justify new dependencies

❌ **Don't**: Use magic numbers without explanation
✅ **Do**: Use named constants with comments

❌ **Don't**: Assume user has domain knowledge
✅ **Do**: Explain financial concepts when relevant

❌ **Don't**: Make sweeping architectural changes without approval
✅ **Do**: Propose design, get feedback, then implement

---

## Quick Reference

### Essential Commands

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Test
./build/mamba_model test

# Process text
./build/mamba_model process "Your text here"

# Clean build
rm -rf build && mkdir build && cd build && cmake .. && cmake --build .
```

### Key Files for Common Tasks

| Task | Files to Modify |
|------|----------------|
| Add new Mamba layer type | `include/transformer.h`, `src/transformer.cpp` |
| Modify tokenization | `include/tokenizer.h`, `src/tokenizer.cpp` |
| Change attention mechanism | `include/sparse_attention.h`, `src/sparse_attention.cpp` |
| Adjust build config | `CMakeLists.txt` |
| Update CI/CD | `.github/workflows/build.yml` or `model-build.yml` |
| Add financial vocabulary | `data/financial_vocab.txt` or `data/ticker_vocabulary.txt` |
| Modify ACT | `include/act_controller.h`, `src/act_controller.cpp` |
| Change Matryoshka dims | `include/matryoshka_encoder.h`, `src/matryoshka_encoder.cpp` |

### Useful grep Commands

```bash
# Find where a class is used
grep -r "ClassName" include/ src/

# Find function definitions
grep -r "function_name(" src/

# Find TODOs
grep -r "TODO" include/ src/

# Find specific pattern in headers
grep -r "pattern" include/ --include="*.h"
```

### Configuration Defaults

| Parameter | Default Value | Location |
|-----------|---------------|----------|
| Embedding dimension | 1536 | `transformer.h` MambaConfig |
| State dimension | 128 | `transformer.h` MambaConfig |
| Number of layers | 6 | `transformer.h` MambaConfig |
| Max sequence length | 100,000 | CMakeLists.txt define |
| Vocabulary size | 50,000 base + 6,700 tickers | `tokenizer.h` |
| ACT max steps | 16 | `act_controller.h` |
| Matryoshka dimensions | [64, 128, 256, 512, 768, 1024, 1536] | `matryoshka_encoder.h` |

---

## Changelog

- **2025-11-21**: Initial CLAUDE.md created with comprehensive documentation
  - Repository structure documented
  - Build system and dependencies explained
  - Development workflows established
  - Code conventions defined
  - Architecture deep dive provided
  - Testing strategy outlined
  - Common tasks and troubleshooting included
  - AI assistant guidelines added

---

## Additional Resources

### Documentation Files
- **README.md** - User-facing project overview and quick start
- **DESIGN.txt** - Detailed architectural design document
- **MATRYOSHKA.md** - Matryoshka encoding implementation details
- **RECURSION.md** - Recursive hybrid architecture explanation
- **SPARSE.md** - Sparse attention mechanisms
- **SERVING.md** - Model serving and deployment

### External Resources
- [Eigen3 Documentation](https://eigen.tuxfamily.org/dox/)
- [CMake Documentation](https://cmake.org/documentation/)
- [Mamba Paper](https://arxiv.org/abs/2312.00752) - State Space Models
- [FRED Economic Data](https://fred.stlouisfed.org/) - Federal Reserve data
- [C++17 Reference](https://en.cppreference.com/w/cpp/17)

### Contact/Support
- GitHub Issues: Report bugs and feature requests
- Documentation: This file and other .md files in the repository

---

**Last Updated**: 2025-11-21
**Version**: 1.0
**Maintained by**: Project contributors and AI assistants
