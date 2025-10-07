#ifndef HYBRID_ATTENTION_H
#define HYBRID_ATTENTION_H

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace transformer {

// Clean configuration structure
struct HybridConfig {
    int state_dim = 128;
    int conv_kernel_size = 4;
    float ssm_dt_init = 0.001f;
    bool use_selective_ssm = true;
    float base_sparsity_ratio = 0.1f;
    float sparsity_adaptation_rate = 0.01f;
    int max_sparsity_budget = 256;
    bool use_adaptive_sparsity = true;
    float ssm_attention_balance = 0.5f;
    bool use_content_aware_fusion = true;
    float fusion_temperature = 1.0f;
    bool use_fused_kernels = true;
    int compute_capability = 80;
};

// SSM state representation
struct SSMState {
    Eigen::MatrixXf hidden_state;
    Eigen::MatrixXf output_gate;
    float delta_t;
};

// Sparsity pattern for attention
struct SparsityPattern {
    std::vector<int> sparse_indices;
    Eigen::MatrixXf attention_mask;
    float sparsity_ratio;
    int budget_used;
};

// Core Mamba2 SSM implementation
class Mamba2SSM {
public:
    Mamba2SSM(int input_dim, int state_dim, int conv_kernel_size = 4);
    ~Mamba2SSM() = default;

    SSMState forward(const Eigen::MatrixXf& input);
    Eigen::MatrixXf parallel_scan(const Eigen::MatrixXf& inputs);
    void initialize_parameters();

private:
    Eigen::MatrixXf compute_delta(const Eigen::MatrixXf& input);
    Eigen::MatrixXf discretize_state_matrix(const Eigen::MatrixXf& delta);
    SSMState step_ssm(const Eigen::MatrixXf& input, const SSMState& prev_state);
    Eigen::MatrixXf apply_convolution(const Eigen::MatrixXf& input);

    int input_dim_, state_dim_, conv_kernel_size_;
    Eigen::MatrixXf A_log_, B_proj_, C_proj_, D_;
    Eigen::MatrixXf conv_weight_;
    Eigen::VectorXf conv_bias_;
};

// Scan-informed sparsity pattern generator
class SparsityGenerator {
public:
    SparsityGenerator(int seq_len, int embed_dim, float base_sparsity = 0.1f);
    ~SparsityGenerator() = default;

    SparsityPattern generate_pattern(const std::vector<SSMState>& scan_states);
    float compute_adaptive_sparsity_ratio(const std::vector<SSMState>& scan_states);

private:
    Eigen::MatrixXf extract_importance_scores(const std::vector<SSMState>& scan_states);
    SparsityPattern create_sparsity_pattern(const Eigen::MatrixXf& importance_scores);

    int seq_len_, embed_dim_;
    float base_sparsity_, adaptation_rate_;
};

// Sparse attention mechanism
class SparseAttention {
public:
    SparseAttention(int embed_dim, int num_heads, float dropout_rate = 0.1f);
    ~SparseAttention() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input,
                           const SparsityPattern& pattern,
                           const Eigen::MatrixXf* /*mask*/ = nullptr);
    void initialize_parameters();

private:
    Eigen::MatrixXf compute_sparse_attention(const Eigen::MatrixXf& Q,
                                           const Eigen::MatrixXf& K,
                                           const Eigen::MatrixXf& V,
                                           const SparsityPattern& pattern);

    int embed_dim_, num_heads_, head_dim_;
    float dropout_rate_, scale_;
    Eigen::MatrixXf W_q_, W_k_, W_v_, W_o_;
    Eigen::VectorXf b_q_, b_k_, b_v_, b_o_;
};

// Adaptive fusion gate
class FusionGate {
public:
    FusionGate(int embed_dim);
    ~FusionGate() = default;

    Eigen::MatrixXf fuse(const SSMState& ssm_state,
                        const Eigen::MatrixXf& attention_output,
                        const Eigen::MatrixXf& original_input);
    float compute_fusion_ratio(const Eigen::MatrixXf& input);

private:
    Eigen::MatrixXf fusion_proj_;
    Eigen::VectorXf fusion_bias_;
    int embed_dim_;
};

// Main hybrid attention layer
class HybridAttention {
public:
    HybridAttention(const HybridConfig& config, int embed_dim);
    ~HybridAttention() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input, const Eigen::MatrixXf* mask = nullptr);
    void set_ssm_attention_balance(float balance);

private:
    HybridConfig config_;
    int embed_dim_;
    float current_ssm_balance_;

    std::unique_ptr<Mamba2SSM> ssm_;
    std::unique_ptr<SparsityGenerator> sparsity_gen_;
    std::unique_ptr<SparseAttention> sparse_attn_;
    std::unique_ptr<FusionGate> fusion_gate_;

    Eigen::MatrixXf input_proj_, output_proj_;
    Eigen::VectorXf input_bias_, output_bias_;
};

// Multi-scale hybrid processing
class MultiScaleHybridAttention {
public:
    MultiScaleHybridAttention(const HybridConfig& config,
                             const std::vector<int>& scale_dims);
    ~MultiScaleHybridAttention() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);

private:
    HybridConfig config_;
    std::vector<int> scale_dims_;
    std::vector<std::unique_ptr<HybridAttention>> scale_layers_;

    Eigen::MatrixXf fuse_multiscale_outputs(const std::vector<Eigen::MatrixXf>& scale_outputs);
};

} // namespace transformer

#endif // HYBRID_ATTENTION_H