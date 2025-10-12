#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <memory>
#include <string>
#include "layer_norm.h"
#include <Eigen/Dense>
#include "tensor_ops.h"

namespace transformer {

// Forward declarations and type definitions
class TransformerBlock;
class TransformerModel;

// Transformer configuration
struct TransformerConfig {
    int embed_dim = 768;
    int num_heads = 12;
    int num_layers = 12;
    int ff_dim = 3072;
    float dropout_rate = 0.1f;
    int max_seq_length = 512;
    float layer_norm_eps = 1e-5f;
    bool gradient_checkpointing = false;
};

// Main transformer classes
class TransformerBlock {
public:
    TransformerBlock(const TransformerConfig& config);
    ~TransformerBlock();

    Matrix forward(const Matrix& input, const Matrix* mask = nullptr);

private:
    class TransformerBlockImpl;
    std::unique_ptr<TransformerBlockImpl> pImpl;
    TransformerConfig config_;
};

class TransformerModel {
public:
    explicit TransformerModel(const TransformerConfig& config = TransformerConfig());
    ~TransformerModel();

    Matrix forward(const Matrix& embeddings, const Matrix* mask = nullptr);
    Matrix get_embeddings_at_dimension(const Matrix& input, int target_dim);
    std::vector<float> encode(const std::vector<std::vector<float>>& embeddings);
    Vector get_pooled_output(const Matrix& encoded, const std::string& pooling_method = "mean");
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

private:
    class TransformerModelImpl;
    std::unique_ptr<TransformerModelImpl> pImpl;
    TransformerConfig config_;
};

} // namespace transformer

namespace mamba {

// Import tensor types from transformer namespace for convenience
using transformer::Scalar;
using transformer::Matrix;
using transformer::Vector;
using transformer::Tensor3D;
using transformer::Tensor4D;

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

    transformer::Matrix forward(const transformer::Matrix& input, const transformer::Matrix* mask = nullptr);
    void set_training(bool training);

private:
    MambaConfig config_;

    // SSM parameters (placeholders)
    transformer::Matrix A_;
    transformer::Matrix B_;
    transformer::Matrix C_;
    transformer::Matrix D_;

    // Convolution and gating (placeholders)
    transformer::Matrix conv_weight_;
    transformer::Matrix gate_proj_;
};

class MambaModel {
public:
    explicit MambaModel(const MambaConfig& config = MambaConfig());
    ~MambaModel();

    // Process embeddings from OpenAI text-embedding-3-small
    transformer::Matrix forward(const transformer::Matrix& embeddings, const transformer::Matrix* mask = nullptr);

    // Process sequence of embeddings
    std::vector<float> encode(const std::vector<std::vector<float>>& embeddings);

    // Get output for classification/regression tasks
    transformer::Vector get_pooled_output(const transformer::Matrix& encoded, const std::string& pooling_method = "mean");

    // Get embeddings at specific Matryoshka dimension for adaptive computation
    transformer::Matrix get_embeddings_at_dimension(const transformer::Matrix& input, int target_dim);

    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

private:
    MambaConfig config_;
    std::vector<std::unique_ptr<MambaBlock>> layers_;

    // Learnable parameters
    transformer::Matrix positional_encoding_;
    transformer::Matrix output_projection_;
};

} // namespace mamba

#endif // TRANSFORMER_H