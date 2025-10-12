#include "../include/transformer.h"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace mamba {

// MambaBlock implementation with continuous SSM view
MambaBlock::MambaBlock(const MambaConfig& config)
    : config_(config) {
    // Initialize SSM parameters for continuous-time view
    A_ = transformer::Matrix::Random(config_.state_dim, config_.state_dim) * 0.1f;
    B_ = transformer::Matrix::Random(config_.embed_dim, config_.state_dim) * 0.1f;
    C_ = transformer::Matrix::Random(config_.state_dim, config_.embed_dim) * 0.1f;
    D_ = transformer::Matrix::Random(config_.embed_dim, config_.embed_dim) * 0.1f;
    conv_weight_ = transformer::Matrix::Random(config_.embed_dim, config_.conv_kernel_size) * 0.1f;
    gate_proj_ = transformer::Matrix::Random(config_.embed_dim, config_.embed_dim) * 0.1f;
}

MambaBlock::~MambaBlock() = default;

transformer::Matrix MambaBlock::forward(const transformer::Matrix& input,
                                         const transformer::Matrix* /*mask*/) {
    const int seq_len = input.rows();
    const int embed_dim = input.cols();

    // Apply convolution for local mixing
    transformer::Matrix conv_out = transformer::Matrix::Zero(seq_len, embed_dim);
    for (int t = 0; t < seq_len; ++t) {
        for (int k = 0; k < config_.conv_kernel_size; ++k) {
            int idx = t - k;
            if (idx >= 0) {
                conv_out.row(t) += input.row(idx) * conv_weight_.col(k);
            }
        }
    }

    // Initialize hidden state
    transformer::Matrix hidden_state = transformer::Matrix::Zero(seq_len, config_.state_dim);

    // Continuous-time SSM forward pass
    for (int t = 0; t < seq_len; ++t) {
        // Compute delta_t for this timestep (learnable parameter)
        float delta_t = config_.ssm_dt_init * config_.ssm_dt_scale;

        // Discretize continuous-time SSM
        transformer::Matrix A_discrete = transformer::Matrix::Identity(config_.state_dim, config_.state_dim) + A_ * delta_t;

        // SSM step: h_t = A_discrete * h_{t-1} + B * x_t
        if (t == 0) {
            hidden_state.row(t) = (B_.transpose() * conv_out.row(t).transpose()).transpose();
        } else {
            hidden_state.row(t) = (A_discrete * hidden_state.row(t-1).transpose()).transpose() +
                                  (B_.transpose() * conv_out.row(t).transpose()).transpose();
        }
    }

    // Output projection: y_t = C * h_t + D * x_t
    transformer::Matrix output = transformer::Matrix::Zero(seq_len, embed_dim);
    for (int t = 0; t < seq_len; ++t) {
        output.row(t) = (C_ * hidden_state.row(t).transpose()).transpose() + (D_ * conv_out.row(t));
    }

    // Apply gating (Swish activation)
    transformer::Matrix gate = conv_out * gate_proj_;
    transformer::Matrix gated = output.array() * gate.unaryExpr([](float x) { return x * (1.0f / (1.0f + std::exp(-x))); }).array();

    return gated;
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

} // namespace mamba