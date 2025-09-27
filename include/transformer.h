#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "tensor_ops.h"

namespace mamba {

// Import tensor types from transformer namespace for convenience
using transformer::Scalar;
using transformer::Matrix;
using transformer::Vector;

class LayerNorm;

struct MambaConfig {
    int embed_dim = 1536;
    int state_dim = 128;           // SSM internal state dimension
    int num_layers = 6;
    int conv_kernel_size = 4;      // Convolution kernel size
    float dropout_rate = 0.1f;
    int max_seq_length = 100000;   // Increased for long context
    float layer_norm_eps = 1e-6f;
    bool use_selective_ssm = true;  // Use selective state space modeling
    int num_threads = 0;           // 0 = auto-detect
    float ssm_dt_init = 0.001f;    // SSM time step initialization
    float ssm_dt_scale = 1.0f;     // SSM time step scaling
    float ssm_dt_min = 0.001f;     // SSM time step minimum
    float ssm_dt_max = 0.1f;       // SSM time step maximum
    bool use_conv = true;          // Use 1D convolution
    int expand_factor = 2;         // MLP expansion factor
    std::string activation = "swish"; // Activation function
};

class MambaBlock {
public:
    MambaBlock(const MambaConfig& config);
    ~MambaBlock();

    Matrix forward(const Matrix& input, const Matrix* mask = nullptr);
    void set_training(bool training);

private:
    MambaConfig config_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;

    // SSM parameters
    Matrix A_;  // State transition matrix (state_dim x state_dim)
    Matrix B_;  // Input projection matrix (state_dim x embed_dim)
    Matrix C_;  // Output projection matrix (embed_dim x state_dim)
    Matrix D_;  // Direct feedthrough matrix (embed_dim x embed_dim)

    // Convolution parameters
    Matrix conv_weight_;  // Convolution kernel (embed_dim x embed_dim x kernel_size)

    // Gating parameters
    Matrix gate_proj_;  // Gate projection (embed_dim x (expand_factor * embed_dim))

    // Production implementation
    class MambaBlockImpl;
    std::unique_ptr<MambaBlockImpl> pImpl;
};

class MambaModel {
public:
    explicit MambaModel(const MambaConfig& config = MambaConfig());
    ~MambaModel();

    // Process embeddings from OpenAI text-embedding-3-small
    Matrix forward(const Matrix& embeddings, const Matrix* mask = nullptr);

    // Process sequence of embeddings
    std::vector<float> encode(const std::vector<std::vector<float>>& embeddings);

    // Get output for classification/regression tasks
    Vector get_pooled_output(const Matrix& encoded, const std::string& pooling_method = "mean");

    // Get embeddings at specific Matryoshka dimension for adaptive computation
    Matrix get_embeddings_at_dimension(const Matrix& input, int target_dim);

    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

private:
    MambaConfig config_;
    std::vector<std::unique_ptr<MambaBlock>> layers_;
    std::unique_ptr<LayerNorm> final_norm_;

    // Learnable parameters
    Matrix positional_encoding_;
    Matrix output_projection_;

    // Production implementation
    class MambaModelImpl;
    std::unique_ptr<MambaModelImpl> pImpl;
};

} // namespace mamba

#endif // TRANSFORMER_H