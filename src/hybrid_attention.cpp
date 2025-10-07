#include "../include/hybrid_attention.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace transformer {

// ==========================================
// Mamba2SSM Implementation
// ==========================================

Mamba2SSM::Mamba2SSM(int input_dim, int state_dim, int conv_kernel_size)
    : input_dim_(input_dim), state_dim_(state_dim), conv_kernel_size_(conv_kernel_size) {
    initialize_parameters();
}

SSMState Mamba2SSM::forward(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf conv_out = apply_convolution(input);
    Eigen::MatrixXf delta = compute_delta(conv_out);
    Eigen::MatrixXf A_discrete = discretize_state_matrix(delta);

    Eigen::MatrixXf B_proj = conv_out * B_proj_;

    SSMState state;
    state.hidden_state = Eigen::MatrixXf::Zero(input.rows(), state_dim_);
    state.output_gate = Eigen::MatrixXf::Zero(input.rows(), input_dim_);

    for (int t = 0; t < input.rows(); ++t) {
        state = step_ssm(B_proj.row(t), state);
        state.output_gate.row(t) = state.hidden_state.row(t) * C_proj_;
    }

    return state;
}

Eigen::MatrixXf Mamba2SSM::parallel_scan(const Eigen::MatrixXf& inputs) {
    std::vector<SSMState> states;
    states.reserve(inputs.rows());

    SSMState initial_state;
    initial_state.hidden_state = Eigen::VectorXf::Zero(state_dim_);
    initial_state.output_gate = Eigen::VectorXf::Zero(input_dim_);

    for (int t = 0; t < inputs.rows(); ++t) {
        initial_state = step_ssm(inputs.row(t), initial_state);
        states.push_back(initial_state);
    }

    Eigen::MatrixXf output(inputs.rows(), input_dim_);
    for (int t = 0; t < inputs.rows(); ++t) {
        output.row(t) = states[t].output_gate;
    }

    return output;
}

void Mamba2SSM::initialize_parameters() {
    A_log_ = Eigen::MatrixXf::Random(state_dim_, state_dim_) * 0.1f;
    B_proj_ = Eigen::MatrixXf::Random(input_dim_, state_dim_) * std::sqrt(2.0f / input_dim_);
    C_proj_ = Eigen::MatrixXf::Random(state_dim_, input_dim_) * std::sqrt(2.0f / state_dim_);
    D_ = Eigen::MatrixXf::Random(input_dim_, input_dim_) * 0.1f;
    conv_weight_ = Eigen::MatrixXf::Random(conv_kernel_size_, input_dim_) * std::sqrt(2.0f / conv_kernel_size_);
    conv_bias_ = Eigen::VectorXf::Zero(input_dim_);
}

Eigen::MatrixXf Mamba2SSM::compute_delta(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf delta = (input.array() * input.array()).rowwise().mean();
    delta = delta.unaryExpr([](float x) { return std::max(0.001f, std::min(0.1f, x)); });
    return delta;
}

Eigen::MatrixXf Mamba2SSM::discretize_state_matrix(const Eigen::MatrixXf& delta) {
    Eigen::MatrixXf A_discrete = A_log_;
    for (int i = 0; i < state_dim_; ++i) {
        for (int j = 0; j < state_dim_; ++j) {
            A_discrete(i, j) = std::exp(delta(0) * A_log_(i, j));
        }
    }
    return A_discrete;
}

SSMState Mamba2SSM::step_ssm(const Eigen::MatrixXf& input, const SSMState& prev_state) {
    SSMState new_state = prev_state;
    new_state.hidden_state = prev_state.hidden_state * A_log_.transpose() + input * B_proj_.transpose();
    return new_state;
}

Eigen::MatrixXf Mamba2SSM::apply_convolution(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf conv_out = Eigen::MatrixXf::Zero(input.rows(), input_dim_);

    for (int t = 0; t < input.rows(); ++t) {
        for (int k = 0; k < conv_kernel_size_; ++k) {
            int src_t = t - k;
            if (src_t >= 0) {
                conv_out.row(t) += input.row(src_t) * conv_weight_.row(k);
            }
        }
        conv_out.row(t) += conv_bias_;
    }

    return conv_out;
}

// ==========================================
// SparsityGenerator Implementation
// ==========================================

SparsityGenerator::SparsityGenerator(int seq_len, int embed_dim, float base_sparsity)
    : seq_len_(seq_len), embed_dim_(embed_dim), base_sparsity_(base_sparsity) {
    adaptation_rate_ = 0.01f;
}

SparsityPattern SparsityGenerator::generate_pattern(const std::vector<SSMState>& scan_states) {
    Eigen::MatrixXf importance_scores = extract_importance_scores(scan_states);
    return create_sparsity_pattern(importance_scores);
}

float SparsityGenerator::compute_adaptive_sparsity_ratio(const std::vector<SSMState>& scan_states) {
    if (scan_states.empty()) return base_sparsity_;

    float total_variation = 0.0f;
    for (size_t t = 1; t < scan_states.size(); ++t) {
        total_variation += (scan_states[t].hidden_state - scan_states[t-1].hidden_state).norm();
    }

    float avg_variation = total_variation / scan_states.size();
    float complexity_factor = std::tanh(avg_variation * 10.0f);
    float adaptive_ratio = base_sparsity_ + adaptation_rate_ * complexity_factor;

    return std::max(0.05f, std::min(0.5f, adaptive_ratio));
}

Eigen::MatrixXf SparsityGenerator::extract_importance_scores(const std::vector<SSMState>& scan_states) {
    Eigen::MatrixXf scores(seq_len_, 1);

    for (int t = 0; t < seq_len_; ++t) {
        scores(t, 0) = (t < static_cast<int>(scan_states.size())) ? scan_states[t].hidden_state.norm() : 0.0f;
    }

    return scores;
}

SparsityPattern SparsityGenerator::create_sparsity_pattern(const Eigen::MatrixXf& importance_scores) {
    SparsityPattern pattern;

    std::vector<std::pair<float, int>> importance_list;
    for (int i = 0; i < importance_scores.rows(); ++i) {
        importance_list.emplace_back(importance_scores(i, 0), i);
    }

    std::sort(importance_list.rbegin(), importance_list.rend());

    int num_sparse = std::max(1, static_cast<int>(seq_len_ * base_sparsity_));
    pattern.sparse_indices.reserve(num_sparse);

    for (int i = 0; i < num_sparse && i < static_cast<int>(importance_list.size()); ++i) {
        pattern.sparse_indices.push_back(importance_list[i].second);
    }

    pattern.attention_mask = Eigen::MatrixXf::Zero(seq_len_, seq_len_);
    for (int query_pos : pattern.sparse_indices) {
        for (int key_pos : pattern.sparse_indices) {
            pattern.attention_mask(query_pos, key_pos) = 1.0f;
        }
    }

    pattern.sparsity_ratio = static_cast<float>(pattern.sparse_indices.size()) / seq_len_;
    pattern.budget_used = pattern.sparse_indices.size();

    return pattern;
}

// ==========================================
// SparseAttention Implementation
// ==========================================

SparseAttention::SparseAttention(int embed_dim, int num_heads, float dropout_rate)
    : embed_dim_(embed_dim), num_heads_(num_heads), dropout_rate_(dropout_rate) {

    head_dim_ = embed_dim_ / num_heads_;
    scale_ = 1.0f / std::sqrt(head_dim_);
    initialize_parameters();
}

Eigen::MatrixXf SparseAttention::forward(const Eigen::MatrixXf& input,
                                        const SparsityPattern& pattern,
                                        const Eigen::MatrixXf* mask) {
    Eigen::MatrixXf Q = input * W_q_;
    Eigen::MatrixXf K = input * W_k_;
    Eigen::MatrixXf V = input * W_v_;

    return compute_sparse_attention(Q, K, V, pattern);
}

void SparseAttention::initialize_parameters() {
    W_q_ = Eigen::MatrixXf::Random(embed_dim_, embed_dim_) * std::sqrt(2.0f / embed_dim_);
    W_k_ = Eigen::MatrixXf::Random(embed_dim_, embed_dim_) * std::sqrt(2.0f / embed_dim_);
    W_v_ = Eigen::MatrixXf::Random(embed_dim_, embed_dim_) * std::sqrt(2.0f / embed_dim_);
    W_o_ = Eigen::MatrixXf::Random(embed_dim_, embed_dim_) * std::sqrt(2.0f / embed_dim_);

    b_q_ = Eigen::VectorXf::Zero(embed_dim_);
    b_k_ = Eigen::VectorXf::Zero(embed_dim_);
    b_v_ = Eigen::VectorXf::Zero(embed_dim_);
    b_o_ = Eigen::VectorXf::Zero(embed_dim_);
}

Eigen::MatrixXf SparseAttention::compute_sparse_attention(const Eigen::MatrixXf& Q,
                                                        const Eigen::MatrixXf& K,
                                                        const Eigen::MatrixXf& V,
                                                        const SparsityPattern& pattern) {
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(Q.rows(), embed_dim_);

    for (int query_idx : pattern.sparse_indices) {
        Eigen::VectorXf attention_weights = Eigen::VectorXf::Zero(K.rows());

        float sum_weights = 0.0f;
        for (int key_idx : pattern.sparse_indices) {
            float score = (Q.row(query_idx) * K.row(key_idx).transpose())(0, 0) * scale_;
            attention_weights(key_idx) = std::exp(score);
            sum_weights += attention_weights(key_idx);
        }

        if (sum_weights > 0) {
            attention_weights /= sum_weights;
        }

        Eigen::VectorXf weighted_sum = Eigen::VectorXf::Zero(embed_dim_);
        for (int key_idx : pattern.sparse_indices) {
            weighted_sum += attention_weights(key_idx) * V.row(key_idx);
        }

        output.row(query_idx) = weighted_sum;
    }

    return output * W_o_;
}

// ==========================================
// FusionGate Implementation
// ==========================================

FusionGate::FusionGate(int embed_dim) : embed_dim_(embed_dim) {
    fusion_proj_ = Eigen::MatrixXf::Random(embed_dim_, embed_dim_) * std::sqrt(2.0f / embed_dim_);
    fusion_bias_ = Eigen::VectorXf::Zero(embed_dim_);
}

Eigen::MatrixXf FusionGate::fuse(const SSMState& ssm_state,
                                const Eigen::MatrixXf& attention_output,
                                const Eigen::MatrixXf& original_input) {
    float fusion_ratio = compute_fusion_ratio(original_input);
    Eigen::MatrixXf combined = fusion_ratio * ssm_state.output_gate + (1.0f - fusion_ratio) * attention_output;
    return combined * fusion_proj_;
}

float FusionGate::compute_fusion_ratio(const Eigen::MatrixXf& input) {
    Eigen::VectorXf row_means = input.rowwise().mean();
    Eigen::MatrixXf centered = input.rowwise() - row_means * Eigen::MatrixXf::Ones(1, input.cols());
    Eigen::VectorXf row_vars = centered.rowwise().squaredNorm() / input.cols();

    float avg_var = row_vars.mean();
    return 0.3f + 0.4f * std::tanh(avg_var * 10.0f);
}

// ==========================================
// HybridAttention Implementation
// ==========================================

HybridAttention::HybridAttention(const HybridConfig& config, int embed_dim)
    : config_(config), embed_dim_(embed_dim), current_ssm_balance_(config.ssm_attention_balance) {

    ssm_ = std::make_unique<Mamba2SSM>(embed_dim, config.state_dim, config.conv_kernel_size);
    sparsity_gen_ = std::make_unique<SparsityGenerator>(0, embed_dim, config.base_sparsity_ratio);
    sparse_attn_ = std::make_unique<SparseAttention>(embed_dim, 8, 0.1f);
    fusion_gate_ = std::make_unique<FusionGate>(embed_dim);

    input_proj_ = Eigen::MatrixXf::Random(embed_dim_, embed_dim_) * std::sqrt(2.0f / embed_dim_);
    input_bias_ = Eigen::VectorXf::Zero(embed_dim_);
    output_proj_ = Eigen::MatrixXf::Random(embed_dim_, embed_dim_) * std::sqrt(2.0f / embed_dim_);
    output_bias_ = Eigen::VectorXf::Zero(embed_dim_);
}

Eigen::MatrixXf HybridAttention::forward(const Eigen::MatrixXf& input, const Eigen::MatrixXf* mask) {
    Eigen::MatrixXf x = input * input_proj_;
    x = x + input_bias_.transpose().replicate(x.rows(), 1);

    // Phase 1: Mamba2 SSM scan
    std::vector<SSMState> scan_states;
    for (int t = 0; t < x.rows(); ++t) {
        SSMState state = ssm_->forward(x.row(t));
        scan_states.push_back(state);
    }

    // Phase 2: Generate sparsity pattern
    SparsityPattern pattern = sparsity_gen_->generate_pattern(scan_states);

    // Phase 3: Apply sparse attention
    Eigen::MatrixXf sparse_output = sparse_attn_->forward(x, pattern, mask);

    // Phase 4: Adaptive fusion
    SSMState combined_state = scan_states.back();
    Eigen::MatrixXf fused_output = fusion_gate_->fuse(combined_state, sparse_output, input);

    // Apply output projection
    Eigen::MatrixXf output = fused_output * output_proj_;
    output = output + output_bias_.transpose().replicate(output.rows(), 1);

    return output;
}

void HybridAttention::set_ssm_attention_balance(float balance) {
    current_ssm_balance_ = std::max(0.0f, std::min(1.0f, balance));
}

// ==========================================
// MultiScaleHybridAttention Implementation
// ==========================================

MultiScaleHybridAttention::MultiScaleHybridAttention(const HybridConfig& config,
                                                   const std::vector<int>& scale_dims)
    : config_(config), scale_dims_(scale_dims) {

    scale_layers_.reserve(scale_dims.size());
    for (int dim : scale_dims) {
        scale_layers_.push_back(std::make_unique<HybridAttention>(config, dim));
    }
}

Eigen::MatrixXf MultiScaleHybridAttention::forward(const Eigen::MatrixXf& input) {
    std::vector<Eigen::MatrixXf> scale_outputs;
    scale_outputs.reserve(scale_layers_.size());

    for (size_t i = 0; i < scale_layers_.size(); ++i) {
        int target_dim = scale_dims_[i];
        Eigen::MatrixXf scale_input = input.leftCols(std::min(target_dim, static_cast<int>(input.cols())));
        scale_outputs.push_back(scale_layers_[i]->forward(scale_input));
    }

    return fuse_multiscale_outputs(scale_outputs);
}

Eigen::MatrixXf MultiScaleHybridAttention::fuse_multiscale_outputs(const std::vector<Eigen::MatrixXf>& scale_outputs) {
    if (scale_outputs.empty()) {
        return Eigen::MatrixXf::Zero(1, scale_dims_[0]);
    }

    int total_dim = 0;
    for (const auto& output : scale_outputs) {
        total_dim += output.cols();
    }

    Eigen::MatrixXf concatenated(scale_outputs[0].rows(), total_dim);
    int col_offset = 0;

    for (const auto& output : scale_outputs) {
        concatenated.middleCols(col_offset, output.cols()) = output;
        col_offset += output.cols();
    }

    return concatenated.leftCols(scale_outputs[0].cols());
}

} // namespace transformer