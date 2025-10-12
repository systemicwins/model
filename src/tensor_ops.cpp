#include "tensor_ops.h"
#include <Eigen/Core>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace transformer {

// Thread-local random generator for dropout
thread_local std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
thread_local std::uniform_real_distribution<Scalar> uniform_dist(0.0f, 1.0f);

// Memory pool implementation
Matrix* TensorMemoryPool::allocateMatrix(int rows, int cols) {
    // Try to reuse from pool
    for (auto it = available_.begin(); it != available_.end(); ++it) {
        if ((*it)->rows() == rows && (*it)->cols() == cols) {
            Matrix* mat = *it;
            available_.erase(it);
            return mat;
        }
    }
    
    // Allocate new matrix
    pool_.push_back(std::make_unique<Matrix>(rows, cols));
    return pool_.back().get();
}

void TensorMemoryPool::releaseMatrix(Matrix* mat) {
    if (mat) {
        available_.push_back(mat);
    }
}

void TensorMemoryPool::clear() {
    pool_.clear();
    available_.clear();
}

// TensorOps implementations
Tensor3D TensorOps::reshape_to_3d(const Matrix& mat, int dim1, int dim2, int dim3) {
    if (mat.size() != dim1 * dim2 * dim3) {
        throw std::runtime_error("Invalid reshape dimensions");
    }
    
    Tensor3D result(dim1, dim2, dim3);
    Eigen::Map<const Matrix> mat_map(mat.data(), mat.rows(), mat.cols());
    
    // Efficient memory copy
    std::memcpy(result.data(), mat.data(), mat.size() * sizeof(Scalar));
    
    return result;
}

Matrix TensorOps::reshape_to_2d(const Tensor3D& tensor, int rows, int cols) {
    if (tensor.size() != rows * cols) {
        throw std::runtime_error("Invalid reshape dimensions");
    }
    
    Matrix result(rows, cols);
    std::memcpy(result.data(), tensor.data(), tensor.size() * sizeof(Scalar));
    
    return result;
}

Tensor3D TensorOps::batched_matmul(const Tensor3D& a, const Tensor3D& b) {
    const int batch_size = a.dimension(0);
    const int m = a.dimension(1);
    const int k = a.dimension(2);
    const int n = b.dimension(2);
    
    if (b.dimension(0) != batch_size || b.dimension(1) != k) {
        throw std::runtime_error("Incompatible dimensions for batched matmul");
    }
    
    Tensor3D result(batch_size, m, n);
    
    #pragma omp parallel for
    for (int batch = 0; batch < batch_size; ++batch) {
        // Extract slices as Eigen matrices
        Eigen::Map<const Matrix> a_slice(
            a.data() + batch * m * k, m, k
        );
        Eigen::Map<const Matrix> b_slice(
            b.data() + batch * k * n, k, n
        );
        Eigen::Map<Matrix> result_slice(
            result.data() + batch * m * n, m, n
        );
        
        // Efficient matrix multiplication
        result_slice.noalias() = a_slice * b_slice;
    }
    
    return result;
}

Tensor3D TensorOps::scaled_dot_product_attention(
    const Tensor3D& Q,
    const Tensor3D& K,
    const Tensor3D& V,
    const Tensor3D* mask,
    Scalar scale,
    Scalar dropout_p,
    bool is_causal) {
    
    const int batch_size = Q.dimension(0);
    const int seq_len_q = Q.dimension(1);
    const int seq_len_k = K.dimension(1);
    const int d_k = Q.dimension(2);
    
    // Compute attention scores: Q @ K^T / sqrt(d_k)
    Tensor3D scores(batch_size, seq_len_q, seq_len_k);
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len_q; ++i) {
            for (int j = 0; j < seq_len_k; ++j) {
                Scalar score = 0.0f;
                for (int k = 0; k < d_k; ++k) {
                    score += Q(b, i, k) * K(b, j, k);
                }
                scores(b, i, j) = score * scale;
                
                // Apply causal mask
                if (is_causal && j > i) {
                    scores(b, i, j) = -1e9f;
                }
                
                // Apply provided mask
                if (mask && (*mask)(b, i, j) == 0) {
                    scores(b, i, j) = -1e9f;
                }
            }
        }
    }
    
    // Stable softmax
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len_q; ++i) {
            // Find max for numerical stability
            Scalar max_score = -1e9f;
            for (int j = 0; j < seq_len_k; ++j) {
                max_score = std::max(max_score, scores(b, i, j));
            }
            
            // Compute exp and sum
            Scalar sum = 0.0f;
            for (int j = 0; j < seq_len_k; ++j) {
                scores(b, i, j) = std::exp(scores(b, i, j) - max_score);
                sum += scores(b, i, j);
            }
            
            // Normalize
            const Scalar inv_sum = 1.0f / (sum + 1e-9f);
            for (int j = 0; j < seq_len_k; ++j) {
                scores(b, i, j) *= inv_sum;
                
                // Apply dropout
                if (dropout_p > 0 && uniform_dist(rng) < dropout_p) {
                    scores(b, i, j) = 0.0f;
                }
            }
        }
    }
    
    // Compute attention output: scores @ V
    const int d_v = V.dimension(2);
    Tensor3D output(batch_size, seq_len_q, d_v);
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len_q; ++i) {
            for (int k = 0; k < d_v; ++k) {
                Scalar sum = 0.0f;
                for (int j = 0; j < seq_len_k; ++j) {
                    sum += scores(b, i, j) * V(b, j, k);
                }
                output(b, i, k) = sum;
            }
        }
    }
    
    return output;
}

std::vector<Tensor3D> TensorOps::split_heads(
    const Matrix& tensor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim) {
    
    if (tensor.rows() != batch_size * seq_len) {
        throw std::runtime_error("Invalid tensor dimensions for split_heads");
    }
    
    std::vector<Tensor3D> heads;
    heads.reserve(num_heads);
    
    for (int h = 0; h < num_heads; ++h) {
        Tensor3D head(batch_size, seq_len, head_dim);
        
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    head(b, s, d) = tensor(b * seq_len + s, h * head_dim + d);
                }
            }
        }
        
        heads.push_back(std::move(head));
    }
    
    return heads;
}

Matrix TensorOps::merge_heads(
    const std::vector<Tensor3D>& heads,
    int batch_size,
    int seq_len,
    int embed_dim) {
    
    const int num_heads = heads.size();
    const int head_dim = embed_dim / num_heads;
    
    Matrix output(batch_size * seq_len, embed_dim);
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    output(b * seq_len + s, h * head_dim + d) = heads[h](b, s, d);
                }
            }
        }
    }
    
    return output;
}

Matrix TensorOps::softmax(const Matrix& input, int axis) {
    Matrix output = input;
    
    if (axis == 1) {  // Row-wise softmax
        #pragma omp parallel for
        for (int i = 0; i < output.rows(); ++i) {
            Scalar max_val = output.row(i).maxCoeff();
            output.row(i) = (output.row(i).array() - max_val).exp();
            output.row(i) /= output.row(i).sum();
        }
    } else if (axis == 0) {  // Column-wise softmax
        #pragma omp parallel for
        for (int j = 0; j < output.cols(); ++j) {
            Scalar max_val = output.col(j).maxCoeff();
            output.col(j) = (output.col(j).array() - max_val).exp();
            output.col(j) /= output.col(j).sum();
        }
    }
    
    return output;
}

Matrix TensorOps::layer_norm(
    const Matrix& input,
    const Vector& gamma,
    const Vector& beta,
    Scalar eps) {
    
    Matrix output(input.rows(), input.cols());
    
    #pragma omp parallel for
    for (int i = 0; i < input.rows(); ++i) {
        // Compute mean and variance
        Scalar mean = input.row(i).mean();
        Scalar var = (input.row(i).array() - mean).square().mean();
        Scalar std_inv = 1.0f / std::sqrt(var + eps);
        
        // Normalize and scale
        output.row(i) = ((input.row(i).array() - mean) * std_inv * gamma.array().transpose() + beta.array().transpose()).matrix();
    }
    
    return output;
}

Matrix TensorOps::gelu_fast(const Matrix& input) {
    // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr Scalar sqrt_2_over_pi = 0.7978845608f;
    constexpr Scalar coeff = 0.044715f;
    
    Matrix output(input.rows(), input.cols());
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            Scalar x = input(i, j);
            Scalar x3 = x * x * x;
            Scalar inner = sqrt_2_over_pi * (x + coeff * x3);
            output(i, j) = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }
    
    return output;
}

Matrix TensorOps::gelu_accurate(const Matrix& input) {
    // Accurate GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    Matrix output(input.rows(), input.cols());
    constexpr Scalar inv_sqrt_2 = 0.7071067812f;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            Scalar x = input(i, j);
            output(i, j) = 0.5f * x * (1.0f + std::erf(x * inv_sqrt_2));
        }
    }

    return output;
}

Matrix TensorOps::sigmoid(const Matrix& input) {
    // Sigmoid: 1 / (1 + exp(-x))
    Matrix output(input.rows(), input.cols());

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            Scalar x = input(i, j);
            output(i, j) = 1.0f / (1.0f + std::exp(-x));
        }
    }

    return output;
}

Matrix TensorOps::dropout(const Matrix& input, Scalar p, bool training) {
    if (!training || p <= 0.0f) {
        return input;
    }
    
    Matrix output = input;
    const Scalar scale = 1.0f / (1.0f - p);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            if (uniform_dist(rng) < p) {
                output(i, j) = 0.0f;
            } else {
                output(i, j) *= scale;
            }
        }
    }
    
    return output;
}

Tensor3D TensorOps::create_causal_mask(int seq_len, int batch_size) {
    Tensor3D mask(batch_size, seq_len, seq_len);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                mask(b, i, j) = (j <= i) ? 1.0f : 0.0f;
            }
        }
    }
    
    return mask;
}

void TensorOps::add_inplace(Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::runtime_error("Dimension mismatch in add_inplace");
    }
    
    #pragma omp parallel for
    for (int i = 0; i < a.size(); ++i) {
        a.data()[i] += b.data()[i];
    }
}

void TensorOps::multiply_inplace(Matrix& a, Scalar scalar) {
    #pragma omp parallel for
    for (int i = 0; i < a.size(); ++i) {
        a.data()[i] *= scalar;
    }
}

// TensorScope implementation
Matrix& TensorScope::allocate(int rows, int cols) {
    Matrix* mat = TensorMemoryPool::getInstance().allocateMatrix(rows, cols);
    allocated_.push_back(mat);
    return *mat;
}

void TensorScope::cleanup() {
    for (Matrix* mat : allocated_) {
        TensorMemoryPool::getInstance().releaseMatrix(mat);
    }
    allocated_.clear();
}

// TensorValidation implementation
void TensorValidation::check_dimensions(const Matrix& a, const Matrix& b, const std::string& op) {
    if (op == "matmul") {
        if (a.cols() != b.rows()) {
            throw std::runtime_error("Matrix dimensions incompatible for multiplication: [" +
                std::to_string(a.rows()) + "x" + std::to_string(a.cols()) + "] @ [" +
                std::to_string(b.rows()) + "x" + std::to_string(b.cols()) + "]");
        }
    } else if (op == "add" || op == "subtract") {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw std::runtime_error("Matrix dimensions must match for " + op);
        }
    }
}

void TensorValidation::check_finite(const Matrix& mat, const std::string& name) {
    if (has_nan(mat)) {
        throw std::runtime_error("NaN detected in " + name);
    }
    if (has_inf(mat)) {
        throw std::runtime_error("Inf detected in " + name);
    }
}

bool TensorValidation::has_nan(const Matrix& mat) {
    return mat.hasNaN();
}

bool TensorValidation::has_inf(const Matrix& mat) {
    return !mat.allFinite();
}

// TensorProfiler implementation
std::unordered_map<std::string, double> TensorProfiler::timings_;
std::unordered_map<std::string, int> TensorProfiler::counts_;

void TensorProfiler::start_timer(const std::string& /*name*/) {
    // Implementation would use high_resolution_clock
}

void TensorProfiler::end_timer(const std::string& /*name*/) {
    // Implementation would calculate elapsed time
}

void TensorProfiler::print_summary() {
    for (const auto& [name, time] : timings_) {
        std::cout << name << ": " << time << "ms (" << counts_[name] << " calls)\n";
    }
}

void TensorProfiler::reset() {
    timings_.clear();
    counts_.clear();
}

// FlashAttention implementation (simplified version)
Matrix FlashAttention::forward(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V,
    int num_heads,
    bool is_causal,
    Scalar /*dropout_p*/) {
    
    const int seq_len = Q.rows();
    const int embed_dim = Q.cols();
    const int head_dim = embed_dim / num_heads;
    
    Matrix output = Matrix::Zero(seq_len, embed_dim);
    
    // Process in blocks for better cache utilization
    for (int h = 0; h < num_heads; ++h) {
        for (int block_i = 0; block_i < seq_len; block_i += BLOCK_SIZE) {
            int block_size_i = std::min(BLOCK_SIZE, seq_len - block_i);
            
            for (int block_j = 0; block_j < seq_len; block_j += BLOCK_SIZE) {
                int block_size_j = std::min(BLOCK_SIZE, seq_len - block_j);
                
                // Skip blocks that would be masked in causal attention
                if (is_causal && block_j > block_i + block_size_i) {
                    continue;
                }
                
                // Extract blocks
                Matrix Q_block = Q.block(block_i, h * head_dim, block_size_i, head_dim);
                Matrix K_block = K.block(block_j, h * head_dim, block_size_j, head_dim);
                Matrix V_block = V.block(block_j, h * head_dim, block_size_j, head_dim);
                
                // Compute attention for this block
                Matrix scores = Q_block * K_block.transpose() / std::sqrt(head_dim);
                
                // Apply causal mask if needed
                if (is_causal) {
                    for (int i = 0; i < block_size_i; ++i) {
                        for (int j = 0; j < block_size_j; ++j) {
                            if (block_i + i < block_j + j) {
                                scores(i, j) = -1e9f;
                            }
                        }
                    }
                }
                
                // Softmax
                scores = TensorOps::softmax(scores, 1);
                
                // Apply to values
                Matrix block_output = scores * V_block;
                
                // Accumulate output
                output.block(block_i, h * head_dim, block_size_i, head_dim) += block_output;
            }
        }
    }
    
    return output;
}

} // namespace transformer