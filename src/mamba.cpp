#include "../include/transformer.h"
#include "../include/sparse_attention.h"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace mamba {

// MambaBlock implementation with Mamba2 scalar A innovation
MambaBlock::MambaBlock(const MambaConfig& config)
    : config_(config) {
    
    // Seed random number generator for consistent initialization
    std::srand(42);
    
    // Mamba2 Innovation: Initialize scalar A values (one per attention head)
    if (config_.use_scalar_a) {
        scalar_a_values_.resize(config_.num_heads);
        for (int i = 0; i < config_.num_heads; ++i) {
            // Initialize with small random values for stable training
            scalar_a_values_[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        }
    }
    
    // Initialize remaining SSM parameters for scalar A per head
    B_ = transformer::Matrix::Random(config_.state_dim, config_.embed_dim) * 0.1f;
    C_ = transformer::Matrix::Random(config_.embed_dim, config_.state_dim) * 0.1f;
    D_ = transformer::Matrix::Random(config_.embed_dim, config_.embed_dim) * 0.1f;
    conv_weight_ = transformer::Matrix::Random(config_.embed_dim, config_.conv_kernel_size) * 0.1f;
    gate_proj_ = transformer::Matrix::Random(config_.embed_dim, config_.embed_dim) * 0.1f;
}

MambaBlock::~MambaBlock() = default;

transformer::Matrix MambaBlock::forward(const transformer::Matrix& input,
                                         const transformer::Matrix* /*mask*/) {
    
    // Apply scalar A SSM if enabled (Mamba2 innovation)
    if (config_.use_scalar_a) {
        return apply_scalar_ssm(input);
    }
    
    // Fallback to traditional matrix A for comparison
    return apply_scalar_ssm(input); // For now, use scalar implementation
}

// Mamba2 scalar A implementation
transformer::Matrix MambaBlock::apply_scalar_ssm(const transformer::Matrix& input) {
    // Validate input dimensions match expected configuration
    const int seq_len = input.rows();
    const int embed_dim = input.cols();
    
    if (seq_len <= 0 || embed_dim <= 0) {
        throw std::invalid_argument("Invalid input dimensions for SSM");
    }

    // Step 1: For now, just return the input to test basic functionality
    // This ensures we have working dimensions before adding complexity
    transformer::Matrix output = input;
    
    // Step 2: Apply gating (simplified)
    transformer::Matrix gate = input * gate_proj_;
    gate = gate.unaryExpr([](float x) {
        return x * (1.0f / (1.0f + std::exp(-x)));
    });
    
    // Step 3: Element-wise multiplication
    transformer::Matrix gated = output.array() * gate.array();

    return gated;
}

// Parallel state evolution with scalar A values
transformer::Matrix MambaBlock::parallel_state_evolution(const transformer::Matrix& conv_out) {
    const int seq_len = conv_out.rows();
    const int embed_dim = conv_out.cols();
    
    // Initialize output matrix
    transformer::Matrix output = transformer::Matrix::Zero(seq_len, embed_dim);
    
    // Process each timestep
    for (int t = 0; t < seq_len; ++t) {
        // Get current input
        transformer::Vector input_row = conv_out.row(t).transpose();
        
        // Apply scalar A SSM - simplified approach
        // For now, just return the input as output to test basic functionality
        // This ensures the dimensions are correct
        output.row(t) = input_row.transpose();
    }
    
    return output;
}

void MambaBlock::set_training(bool /*training*/) {}

MambaModel::MambaModel(const MambaConfig& config)
    : config_(config) {
}

MambaModel::~MambaModel() = default;

transformer::Matrix MambaModel::forward(const transformer::Matrix& embeddings,
                                        const transformer::Matrix* /*mask*/) {
    // Minimal placeholder: pass-through
    return embeddings;
}

std::vector<float> MambaModel::encode(const std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) {
        return {};
    }

    const int seq_len = static_cast<int>(embeddings.size());
    const int embed_dim = static_cast<int>(embeddings.front().size());

    transformer::Matrix input(seq_len, embed_dim);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            input(i, j) = embeddings[i][j];
        }
    }

    transformer::Matrix output = forward(input);

    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(output.rows()) * static_cast<size_t>(output.cols()));
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            flat.push_back(output(i, j));
        }
    }
    return flat;
}

transformer::Vector MambaModel::get_pooled_output(const transformer::Matrix& encoded,
                                                  const std::string& pooling_method) {
    if (encoded.rows() == 0) {
        return transformer::Vector();
    }

    if (pooling_method == "mean") {
        return encoded.colwise().mean();
    }
    if (pooling_method == "max") {
        return encoded.colwise().maxCoeff();
    }
    if (pooling_method == "first") {
        return encoded.row(0);
    }
    if (pooling_method == "last") {
        return encoded.row(encoded.rows() - 1);
    }
    // Default to mean pooling
    return encoded.colwise().mean();
}

transformer::Matrix MambaModel::get_embeddings_at_dimension(const transformer::Matrix& input,
                                                            int target_dim) {
    const int in_dim = input.cols();
    const int use_dim = std::min(in_dim, target_dim);

    transformer::Matrix out(input.rows(), use_dim);
    out = input.leftCols(use_dim);
    return out;
}

void MambaModel::save_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    // Serialize minimal config
    file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
    file.close();
}

void MambaModel::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }
    // Read minimal config
    MambaConfig loaded{};
    file.read(reinterpret_cast<char*>(&loaded), sizeof(loaded));
    file.close();
    // Keep current config_, but ensure dimensions align if needed
}

// NEW: Integration method with sparse attention using scalar A values
transformer::Matrix MambaBlock::forward_with_sparse_attention(
    const transformer::Matrix& input,
    class SSMSparseAttention& sparse_attention) {
    
    const int seq_len = input.rows();
    const int embed_dim = input.cols();
    
    // Step 1: Apply convolution for local mixing
    transformer::Matrix conv_out = transformer::Matrix::Zero(seq_len, embed_dim);
    for (int t = 0; t < seq_len; ++t) {
        for (int k = 0; k < config_.conv_kernel_size; ++k) {
            int idx = t - k;
            if (idx >= 0) {
                conv_out.row(t) += input.row(idx) * conv_weight_.col(k);
            }
        }
    }
    
    // Step 2: Use scalar A values to inform sparse attention patterns
    std::vector<transformer::Matrix> ssm_states;
    ssm_states.reserve(seq_len);
    
    // Collect hidden states at each timestep for state change analysis
    std::vector<transformer::Vector> head_hidden_states(config_.num_heads);
    for (int h = 0; h < config_.num_heads; ++h) {
        head_hidden_states[h] = transformer::Vector::Zero(config_.state_dim);
    }
    
    for (int t = 0; t < seq_len; ++t) {
        // Parallel processing across all heads with scalar A
        #pragma omp parallel for
        for (int h = 0; h < config_.num_heads; ++h) {
            if (t == 0) {
                head_hidden_states[h] = B_.row(h).transpose() * conv_out.row(t);
            } else {
                head_hidden_states[h] = scalar_a_values_[h] * head_hidden_states[h] +
                                       B_.row(h).transpose() * conv_out.row(t);
            }
        }
        
        // Combine all heads for state storage
        transformer::Vector combined_state = transformer::Vector::Zero(config_.state_dim);
        for (int h = 0; h < config_.num_heads; ++h) {
            combined_state += head_hidden_states[h];
        }
        ssm_states.push_back(combined_state);
    }
    
    // Step 3: Analyze state changes using scalar A values for better attention patterns
    auto state_metrics = sparse_attention.analyze_state_change_with_scalar_a(
        ssm_states, scalar_a_values_, seq_len - 1
    );
    
    // Step 4: Select optimal attention pattern based on scalar A dynamics
    auto attention_pattern = sparse_attention.select_attention_pattern(state_metrics);
    
    // Step 5: Apply the selected attention pattern
    transformer::Matrix attention_output;
    switch (attention_pattern) {
        case mamba::AttentionPattern::DENSE:
            // Full attention computation for high-change regions
            attention_output = compute_dense_attention_from_states(ssm_states, conv_out);
            break;
        case mamba::AttentionPattern::LOCAL_WINDOW:
            // Local attention for moderate changes
            attention_output = compute_local_attention_from_states(ssm_states, conv_out);
            break;
        case mamba::AttentionPattern::STRIDED:
            // Strided attention for low changes
            attention_output = compute_strided_attention_from_states(ssm_states, conv_out);
            break;
        case mamba::AttentionPattern::GLOBAL_SPARSE:
        default:
            // Global sparse attention for very low changes
            attention_output = compute_sparse_attention_from_states(ssm_states, conv_out);
            break;
    }
    
    // Step 6: Apply gating and return
    transformer::Matrix gate = conv_out * gate_proj_;
    transformer::Matrix gated = attention_output.array() * gate.unaryExpr([](float x) {
        return x * (1.0f / (1.0f + std::exp(-x)));
    }).array();
    
    return gated;
}

// Helper method: Compute dense attention from SSM states
transformer::Matrix MambaBlock::compute_dense_attention_from_states(
    const std::vector<transformer::Matrix>& states,
    const transformer::Matrix& conv_out) {
    
    const int seq_len = states.size();
    const int embed_dim = conv_out.cols();
    
    transformer::Matrix output = transformer::Matrix::Zero(seq_len, embed_dim);
    
    // Apply full attention computation using state information
    for (int t = 0; t < seq_len; ++t) {
        // Use state information to inform attention
        transformer::Vector state_influence = C_ * states[t].col(0);  // Take first column as vector
        output.row(t) = state_influence.transpose() + conv_out.row(t) * D_;
    }
    
    return output;
}

// Helper method: Compute local attention from SSM states
transformer::Matrix MambaBlock::compute_local_attention_from_states(
    const std::vector<transformer::Matrix>& states,
    const transformer::Matrix& conv_out) {
    
    const int seq_len = states.size();
    const int embed_dim = conv_out.cols();
    const int window_size = 4;  // Local window size
    
    transformer::Matrix output = transformer::Matrix::Zero(seq_len, embed_dim);
    
    for (int t = 0; t < seq_len; ++t) {
        // Compute local attention within window
        int start = std::max(0, t - window_size);
        int end = std::min(seq_len, t + window_size + 1);
        
        transformer::Vector local_attention = transformer::Vector::Zero(embed_dim);
        for (int i = start; i < end; ++i) {
            float weight = 1.0f / (1.0f + std::abs(i - t));  // Distance-based weighting
            local_attention += weight * (C_ * states[i].col(0));
        }
        
        output.row(t) = local_attention.transpose() + conv_out.row(t) * D_;
    }
    
    return output;
}

// Helper method: Compute strided attention from SSM states
transformer::Matrix MambaBlock::compute_strided_attention_from_states(
    const std::vector<transformer::Matrix>& states,
    const transformer::Matrix& conv_out) {
    
    const int seq_len = states.size();
    const int embed_dim = conv_out.cols();
    const int stride = 2;  // Stride size
    
    transformer::Matrix output = transformer::Matrix::Zero(seq_len, embed_dim);
    
    for (int t = 0; t < seq_len; ++t) {
        // Strided attention: attend to positions t, t+stride, t+2*stride, etc.
        transformer::Vector strided_attention = transformer::Vector::Zero(embed_dim);
        for (int i = t; i < seq_len; i += stride) {
            float weight = std::exp(-0.1f * (i - t));  // Decay with distance
            strided_attention += weight * (C_ * states[i].col(0));
        }
        
        output.row(t) = strided_attention.transpose() + conv_out.row(t) * D_;
    }
    
    return output;
}

// Helper method: Compute sparse attention from SSM states
transformer::Matrix MambaBlock::compute_sparse_attention_from_states(
    const std::vector<transformer::Matrix>& states,
    const transformer::Matrix& conv_out) {
    
    const int seq_len = states.size();
    const int embed_dim = conv_out.cols();
    const int sparse_factor = 8;  // Only attend to 1/sparse_factor positions
    
    transformer::Matrix output = transformer::Matrix::Zero(seq_len, embed_dim);
    
    for (int t = 0; t < seq_len; ++t) {
        // Sparse attention: only attend to key positions
        transformer::Vector sparse_attention = transformer::Vector::Zero(embed_dim);
        for (int i = 0; i < seq_len; i += sparse_factor) {
            sparse_attention += C_ * states[i].col(0);
        }
        
        // Normalize by number of attended positions
        sparse_attention /= (seq_len / sparse_factor);
        output.row(t) = sparse_attention.transpose() + conv_out.row(t) * D_;
    }
    
    return output;
}

} // namespace mamba