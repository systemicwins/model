#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>
#include <stdexcept>

namespace transformer {

// Type aliases for clarity
using Scalar = float;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
using Tensor3D = Eigen::Tensor<Scalar, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<Scalar, 4, Eigen::RowMajor>;

// Memory pool for tensor allocation
class TensorMemoryPool {
public:
    static TensorMemoryPool& getInstance() {
        static TensorMemoryPool instance;
        return instance;
    }
    
    Matrix* allocateMatrix(int rows, int cols);
    void releaseMatrix(Matrix* mat);
    void clear();
    
private:
    TensorMemoryPool() = default;
    std::vector<std::unique_ptr<Matrix>> pool_;
    std::vector<Matrix*> available_;
};

// Efficient tensor operations
class TensorOps {
public:
    // Reshape operations with zero-copy when possible
    static Tensor3D reshape_to_3d(const Matrix& mat, int dim1, int dim2, int dim3);
    static Matrix reshape_to_2d(const Tensor3D& tensor, int rows, int cols);
    
    // Batched matrix multiplication
    static Tensor3D batched_matmul(const Tensor3D& a, const Tensor3D& b);
    
    // Attention-specific operations
    static Tensor3D scaled_dot_product_attention(
        const Tensor3D& Q,
        const Tensor3D& K, 
        const Tensor3D& V,
        const Tensor3D* mask = nullptr,
        Scalar scale = 1.0f,
        Scalar dropout_p = 0.0f,
        bool is_causal = false
    );
    
    // Split tensor for multi-head attention
    static std::vector<Tensor3D> split_heads(
        const Matrix& tensor,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim
    );
    
    // Merge heads back
    static Matrix merge_heads(
        const std::vector<Tensor3D>& heads,
        int batch_size,
        int seq_len,
        int embed_dim
    );
    
    // Optimized softmax
    static void softmax_inplace(Tensor3D& tensor, int axis = -1);
    static Matrix softmax(const Matrix& input, int axis = 1);
    
    // Layer norm with fused operations
    static Matrix layer_norm(
        const Matrix& input,
        const Vector& gamma,
        const Vector& beta,
        Scalar eps = 1e-5f
    );
    
    // GELU activation (faster approximation)
    static Matrix gelu_fast(const Matrix& input);
    static Matrix gelu_accurate(const Matrix& input);

    // Sigmoid activation
    static Matrix sigmoid(const Matrix& input);
    
    // Dropout with cached mask
    static Matrix dropout(const Matrix& input, Scalar p, bool training = true);
    
    // Attention mask creation
    static Tensor3D create_causal_mask(int seq_len, int batch_size = 1);
    static Tensor3D create_padding_mask(const Vector& lengths, int max_len);
    
    // Parallel operations using Eigen's threading
    static void parallel_apply(Matrix& mat, std::function<Scalar(Scalar)> func);
    
    // Memory-efficient operations
    static void add_inplace(Matrix& a, const Matrix& b);
    static void multiply_inplace(Matrix& a, Scalar scalar);
    
    // Numerical stability helpers
    static Scalar log_sum_exp(const Vector& vec);
    static Matrix stable_softmax(const Matrix& input);
};

// RAII wrapper for tensor operations
class TensorScope {
public:
    TensorScope() = default;
    ~TensorScope() { cleanup(); }
    
    Matrix& allocate(int rows, int cols);
    void cleanup();
    
private:
    std::vector<Matrix*> allocated_;
};

// Optimized attention mechanism
class FlashAttention {
public:
    static Matrix forward(
        const Matrix& Q,
        const Matrix& K,
        const Matrix& V,
        int num_heads,
        bool is_causal = false,
        Scalar dropout_p = 0.0f
    );
    
private:
    static constexpr int BLOCK_SIZE = 64;  // Optimal for cache
    static void compute_block(
        const Matrix& Q_block,
        const Matrix& K_block,
        const Matrix& V_block,
        Matrix& O_block,
        Matrix& l_block,
        Matrix& m_block
    );
};

// Validation utilities
class TensorValidation {
public:
    static void check_dimensions(const Matrix& a, const Matrix& b, const std::string& op);
    static void check_finite(const Matrix& mat, const std::string& name);
    static bool has_nan(const Matrix& mat);
    static bool has_inf(const Matrix& mat);
    
    template<typename T>
    static void assert_shape(const T& tensor, const std::vector<int>& expected_shape) {
        // Implementation in header for template
        bool match = true;
        if (expected_shape.size() != tensor.NumDimensions) {
            match = false;
        } else {
            for (size_t i = 0; i < expected_shape.size(); ++i) {
                if (expected_shape[i] != -1 && expected_shape[i] != tensor.dimension(i)) {
                    match = false;
                    break;
                }
            }
        }
        
        if (!match) {
            std::string msg = "Shape mismatch. Expected: [";
            for (size_t i = 0; i < expected_shape.size(); ++i) {
                msg += std::to_string(expected_shape[i]);
                if (i < expected_shape.size() - 1) msg += ", ";
            }
            msg += "], Got: [";
            for (int i = 0; i < tensor.NumDimensions; ++i) {
                msg += std::to_string(tensor.dimension(i));
                if (i < tensor.NumDimensions - 1) msg += ", ";
            }
            msg += "]";
            throw std::runtime_error(msg);
        }
    }
};

// Performance monitoring
class TensorProfiler {
public:
    static void start_timer(const std::string& name);
    static void end_timer(const std::string& name);
    static void print_summary();
    static void reset();
    
private:
    static std::unordered_map<std::string, double> timings_;
    static std::unordered_map<std::string, int> counts_;
};

} // namespace transformer

#endif // TENSOR_OPS_H