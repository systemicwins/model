#include "../include/transformer.h"
#include "../include/attention.h"
#include "../include/feedforward.h"
#include "../include/layer_norm.h"
#include "../include/tensor_ops.h"
#include "../include/compact_positional_encoding_f32.h"

// Include GAM header (already included in attention.h)
#include <Eigen/Dense>
// #include "matryoshka_encoder.h"  // TODO: Enable when implemented
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>

namespace transformer {

// Production-ready TransformerBlock implementation
class TransformerBlock::TransformerBlockImpl {
public:
    TransformerBlockImpl(const TransformerConfig& config)
        : config_(config),
          dropout_rate_(config.dropout_rate),
          is_training_(true) {
        
        // Pre-allocate weight matrices for efficiency
        qkv_weight_ = Matrix::Random(config.embed_dim, 3 * config.embed_dim) * 
                      std::sqrt(2.0f / config.embed_dim);
        qkv_bias_ = Vector::Zero(3 * config.embed_dim);
        
        out_proj_weight_ = Matrix::Random(config.embed_dim, config.embed_dim) * 
                          std::sqrt(2.0f / config.embed_dim);
        out_proj_bias_ = Vector::Zero(config.embed_dim);
        
        // Layer norm parameters
        ln1_gamma_ = Vector::Ones(config.embed_dim);
        ln1_beta_ = Vector::Zero(config.embed_dim);
        ln2_gamma_ = Vector::Ones(config.embed_dim);
        ln2_beta_ = Vector::Zero(config.embed_dim);
        
        // Feed-forward weights
        std::cout << "DEBUG: Initializing FF weights - embed_dim=" << config.embed_dim << ", ff_dim=" << config.ff_dim << std::endl;
        ff1_weight_ = Matrix::Random(config.embed_dim, config.ff_dim) *
                     std::sqrt(2.0f / config.embed_dim);
        std::cout << "DEBUG: FF weights initialized - ff1_weight_ shape: " << ff1_weight_.rows() << " x " << ff1_weight_.cols() << std::endl;
        ff1_bias_ = Vector::Zero(config.ff_dim);
        ff2_weight_ = Matrix::Random(config.ff_dim, config.embed_dim) *
                     std::sqrt(2.0f / config.ff_dim);
        ff2_bias_ = Vector::Zero(config.embed_dim);
    }
    
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr) {
        std::cout << "DEBUG: TransformerBlock forward starting..." << std::endl;
        TensorScope scope;  // RAII for memory management

        // Validate input
        TensorValidation::check_finite(input, "transformer_input");
        if (input.cols() != config_.embed_dim) {
            throw std::runtime_error("Input dimension mismatch");
        }

        const int batch_size = 1;  // Can be extended for batch processing
        const int seq_len = input.rows();
        std::cout << "DEBUG: TransformerBlock - seq_len=" << seq_len << ", embed_dim=" << config_.embed_dim << std::endl;
        
        // Multi-head self-attention with fused QKV projection
        std::cout << "DEBUG: Allocating QKV matrix..." << std::endl;
        Matrix qkv = scope.allocate(seq_len, 3 * config_.embed_dim);
        std::cout << "DEBUG: Computing QKV projection..." << std::endl;
        qkv.noalias() = input * qkv_weight_;
        qkv.rowwise() += RowVector(qkv_bias_);
        std::cout << "DEBUG: QKV projection completed" << std::endl;

        // Split into Q, K, V
        std::cout << "DEBUG: Splitting Q, K, V..." << std::endl;
        Matrix Q = qkv.leftCols(config_.embed_dim);
        Matrix K = qkv.middleCols(config_.embed_dim, config_.embed_dim);
        Matrix V = qkv.rightCols(config_.embed_dim);
        std::cout << "DEBUG: Q, K, V split completed" << std::endl;

        // Efficient multi-head attention
        std::cout << "DEBUG: Running multi-head attention..." << std::endl;
        Matrix attn_output = multihead_attention_optimized(Q, K, V, mask);
        std::cout << "DEBUG: Multi-head attention completed" << std::endl;
        
        // Residual connection + layer norm
        std::cout << "DEBUG: Applying residual connection and layer norm..." << std::endl;
        Matrix x1 = scope.allocate(seq_len, config_.embed_dim);
        TensorOps::add_inplace(attn_output, input);
        x1 = TensorOps::layer_norm(attn_output, ln1_gamma_, ln1_beta_, config_.layer_norm_eps);
        std::cout << "DEBUG: Residual connection and layer norm completed" << std::endl;

        // Feed-forward network with GELU activation
        std::cout << "DEBUG: Running feed-forward network..." << std::endl;
        Matrix ff_output = feed_forward_optimized(x1);
        std::cout << "DEBUG: Feed-forward network completed" << std::endl;
        
        // Second residual connection + layer norm
        std::cout << "DEBUG: Applying second residual connection and layer norm..." << std::endl;
        TensorOps::add_inplace(ff_output, x1);
        Matrix output = TensorOps::layer_norm(ff_output, ln2_gamma_, ln2_beta_, config_.layer_norm_eps);
        std::cout << "DEBUG: Second residual connection and layer norm completed" << std::endl;

        // Validate output
        TensorValidation::check_finite(output, "transformer_output");

        std::cout << "DEBUG: TransformerBlock forward completed successfully" << std::endl;
        return output;
    }
    
    void set_training(bool training) { is_training_ = training; }
    
private:
    Matrix multihead_attention_optimized(const Matrix& Q, const Matrix& K, const Matrix& V,
                                        const Matrix* mask) {
        // Use Gated Attention Mechanism (GAM) for financial data processing
        TensorScope scope;

        // Initialize GAM for this layer
        GatedMultiHeadAttention gam(config_.embed_dim, config_.num_heads, dropout_rate_);

        // Apply GAM to Q, K, V
        Matrix output = gam.forward(Q, K, V, mask);

        return output;
    }
    
    Matrix feed_forward_optimized(const Matrix& input) {
        const int seq_len = input.rows();
        std::cout << "DEBUG: Feed-forward - allocating hidden matrix: " << seq_len << " x " << config_.ff_dim << std::endl;
        TensorScope scope;

        // First linear layer with GELU
        Matrix hidden = scope.allocate(seq_len, config_.ff_dim);
        std::cout << "DEBUG: Feed-forward - input shape: " << input.rows() << " x " << input.cols() << std::endl;
        std::cout << "DEBUG: Feed-forward - ff1_weight_ shape: " << ff1_weight_.rows() << " x " << ff1_weight_.cols() << std::endl;
        std::cout << "DEBUG: Feed-forward - hidden shape: " << hidden.rows() << " x " << hidden.cols() << std::endl;
        std::cout << "DEBUG: Feed-forward - computing first linear layer..." << std::endl;
        hidden.noalias() = input * ff1_weight_;
        std::cout << "DEBUG: Feed-forward - matrix multiplication completed" << std::endl;
        std::cout << "DEBUG: Feed-forward - ff1_bias_ size: " << ff1_bias_.size() << std::endl;

        // Try different approach for bias addition
        std::cout << "DEBUG: Feed-forward - trying alternative bias addition..." << std::endl;
        for (int i = 0; i < seq_len; ++i) {
            hidden.row(i) += ff1_bias_.transpose();
        }
        std::cout << "DEBUG: Feed-forward - alternative bias addition completed" << std::endl;
        std::cout << "DEBUG: Feed-forward - first linear layer completed" << std::endl;

        // Use fast GELU approximation for speed
        std::cout << "DEBUG: Feed-forward - applying GELU..." << std::endl;
        hidden = TensorOps::gelu_fast(hidden);
        std::cout << "DEBUG: Feed-forward - GELU completed" << std::endl;
        
        // Dropout
        if (is_training_ && dropout_rate_ > 0) {
            std::cout << "DEBUG: Feed-forward - applying dropout..." << std::endl;
            hidden = TensorOps::dropout(hidden, dropout_rate_, true);
            std::cout << "DEBUG: Feed-forward - dropout completed" << std::endl;
        }

        // Second linear layer
        std::cout << "DEBUG: Feed-forward - creating output matrix: " << seq_len << " x " << config_.embed_dim << std::endl;
        Matrix output(seq_len, config_.embed_dim);
        std::cout << "DEBUG: Feed-forward - computing second linear layer..." << std::endl;
        output.noalias() = hidden * ff2_weight_;
        std::cout << "DEBUG: Feed-forward - second matrix multiplication completed" << std::endl;

        // Alternative bias addition
        std::cout << "DEBUG: Feed-forward - adding bias to output..." << std::endl;
        for (int i = 0; i < seq_len; ++i) {
            output.row(i) += ff2_bias_.transpose();
        }
        std::cout << "DEBUG: Feed-forward - output bias addition completed" << std::endl;
        std::cout << "DEBUG: Feed-forward - second linear layer completed" << std::endl;

        // Output dropout
        if (is_training_ && dropout_rate_ > 0) {
            std::cout << "DEBUG: Feed-forward - applying output dropout..." << std::endl;
            output = TensorOps::dropout(output, dropout_rate_, true);
            std::cout << "DEBUG: Feed-forward - output dropout completed" << std::endl;
        }
        
        return output;
    }
    
    TransformerConfig config_;
    float dropout_rate_;
    bool is_training_;
    
    // Attention weights
    Matrix qkv_weight_;
    Vector qkv_bias_;
    Matrix out_proj_weight_;
    Vector out_proj_bias_;
    
    // Layer norm parameters
    Vector ln1_gamma_, ln1_beta_;
    Vector ln2_gamma_, ln2_beta_;
    
    // Feed-forward weights
    Matrix ff1_weight_, ff1_bias_;
    Matrix ff2_weight_, ff2_bias_;
};

// Production TransformerBlock wrapper
TransformerBlock::TransformerBlock(const TransformerConfig& config)
    : config_(config) {
    pImpl = std::make_unique<TransformerBlockImpl>(config);
}

TransformerBlock::~TransformerBlock() = default;

Matrix TransformerBlock::forward(const Matrix& input, const Matrix* mask) {
    return pImpl->forward(input, mask);
}

// Production TransformerModel implementation  
class TransformerModel::TransformerModelImpl {
public:
    TransformerModelImpl(const TransformerConfig& config)
        : config_(config), is_training_(true) {
        
        // Initialize Matryoshka encoder as required component
        // TODO: Enable when matryoshka_encoder.cpp is implemented
        // matryoshka::MatryoshkaConfig matryoshka_config;
        // matryoshka_config.transformer_config = config;
        // matryoshka_config.use_learned_pooling = true;
        // matryoshka_config.use_dimension_specific_heads = true;
        // matryoshka_encoder_ = std::make_unique<matryoshka::MatryoshkaEncoder>(matryoshka_config);
        
        // Initialize transformer blocks
        layers_.reserve(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            layers_.push_back(std::make_unique<TransformerBlock>(config));
        }
        
        // Initialize embeddings and positional encoding
        initialize_positional_encoding();
        
        // Initialize compact positional encoder
        CompactPositionalEncodingF32::Config pos_config;
        pos_config.max_seq_length = config.max_seq_length;
        pos_config.max_embed_dim = config.embed_dim;
        pos_config.type = CompactPositionalEncodingF32::ROTARY;
        compact_pos_encoder_ = std::make_unique<CompactPositionalEncodingF32>(pos_config);
        
        // Output projection with proper initialization
        output_projection_ = Matrix::Random(config.embed_dim, config.embed_dim) * 
                            std::sqrt(2.0f / (config.num_layers * config.embed_dim));
        
        // Final layer norm
        final_ln_gamma_ = Vector::Ones(config.embed_dim);
        final_ln_beta_ = Vector::Zero(config.embed_dim);
        
        // Pre-compute attention mask for efficiency
        if (config.max_seq_length > 0) {
            causal_mask_ = create_causal_mask(config.max_seq_length);
        }
    }
    
    Matrix forward(const Matrix& embeddings, const Matrix* mask = nullptr) {
        TensorProfiler::start_timer("transformer_forward");

        // Input validation
        const int seq_len = embeddings.rows();
        std::cout << "DEBUG: Forward pass starting with seq_len=" << seq_len << ", embed_dim=" << config_.embed_dim << std::endl;
        if (seq_len > config_.max_seq_length) {
            throw std::runtime_error("Sequence length exceeds maximum");
        }
        TensorValidation::check_finite(embeddings, "input_embeddings");
        
        // Apply Matryoshka encoding (includes positional encoding internally)
        // TODO: Enable when matryoshka_encoder is implemented
        // Matrix x = apply_matryoshka_encoding(embeddings, config_.embed_dim);
        
        // For now, use traditional positional encoding
        std::cout << "DEBUG: Adding positional encoding..." << std::endl;
        Matrix x = embeddings + positional_encoding_.topRows(seq_len);
        std::cout << "DEBUG: Positional encoding added successfully" << std::endl;

        // Apply dropout to embeddings
        if (is_training_ && config_.dropout_rate > 0) {
            std::cout << "DEBUG: Applying dropout..." << std::endl;
            x = TensorOps::dropout(x, config_.dropout_rate, true);
            std::cout << "DEBUG: Dropout applied successfully" << std::endl;
        }
        
        // Determine mask to use
        const Matrix* effective_mask = mask;
        Matrix truncated_mask;
        if (!mask && seq_len <= causal_mask_.rows()) {
            truncated_mask = causal_mask_.topLeftCorner(seq_len, seq_len);
            effective_mask = &truncated_mask;
        }
        
        // Pass through transformer blocks with gradient checkpointing
        std::cout << "DEBUG: Starting transformer layer processing..." << std::endl;
        for (size_t i = 0; i < layers_.size(); ++i) {
            std::cout << "DEBUG: Processing layer " << i << "..." << std::endl;
            // Optional: Add gradient checkpointing for memory efficiency
            if (config_.gradient_checkpointing && i % 2 == 0) {
                // Store intermediate for backward pass
                checkpoint_states_.push_back(x);
            }

            x = layers_[i]->forward(x, effective_mask);

            // Check for numerical issues
            if (i % 4 == 0) {  // Check every 4 layers
                TensorValidation::check_finite(x, "layer_" + std::to_string(i));
            }
            std::cout << "DEBUG: Layer " << i << " completed" << std::endl;
        }
        std::cout << "DEBUG: All transformer layers completed" << std::endl;
        
        // Final layer normalization
        std::cout << "DEBUG: Applying final layer normalization..." << std::endl;
        x = TensorOps::layer_norm(x, final_ln_gamma_, final_ln_beta_, config_.layer_norm_eps);
        std::cout << "DEBUG: Final layer normalization completed" << std::endl;

        // Output projection
    std::cout << "DEBUG: Applying output projection (in->out)..." << std::endl;
    Matrix output = x * output_projection_;
        std::cout << "DEBUG: Output projection completed" << std::endl;

        TensorProfiler::end_timer("transformer_forward");

        return output;
    }
    
    void set_training(bool training) {
        is_training_ = training;
        for ([[maybe_unused]] auto& layer : layers_) {
            // Propagate training mode to all layers
            // layer->set_training(training);
        }
    }
    
    void save_weights(const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }
        
        // Save config
        file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        
        // Save weights with versioning
        const int version = 2;  // Model version for compatibility
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Save output projection
        save_matrix(file, output_projection_);
        save_vector(file, final_ln_gamma_);
        save_vector(file, final_ln_beta_);
        
        // TODO: Save individual layer weights
        
        file.close();
    }
    
    void load_weights(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filepath);
        }
        
        // Load config
        TransformerConfig loaded_config;
        file.read(reinterpret_cast<char*>(&loaded_config), sizeof(loaded_config));
        
        // Verify compatibility
        if (loaded_config.embed_dim != config_.embed_dim ||
            loaded_config.num_layers != config_.num_layers) {
            throw std::runtime_error("Model configuration mismatch");
        }
        
        // Load version
        int version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        
        if (version != 2) {
            throw std::runtime_error("Unsupported model version");
        }
        
        // Load weights
        load_matrix(file, output_projection_);
        load_vector(file, final_ln_gamma_);
        load_vector(file, final_ln_beta_);
        
        file.close();
    }
    
    // Apply Matryoshka encoding with specified target dimension
    Matrix apply_matryoshka_encoding(const Matrix& embeddings, int target_dim) {
        // Convert Matrix to vector format for Matryoshka encoder
        std::vector<std::vector<float>> input_sequence;
        input_sequence.reserve(embeddings.rows());
        
        for (int i = 0; i < embeddings.rows(); ++i) {
            std::vector<float> row_vec(embeddings.cols());
            for (int j = 0; j < embeddings.cols(); ++j) {
                row_vec[j] = embeddings(i, j);
            }
            input_sequence.push_back(row_vec);
        }
        
        // Encode with Matryoshka at target dimension (includes positional encoding)
        // TODO: Enable when matryoshka_encoder.cpp is implemented
        // auto encoded = matryoshka_encoder_->encode_at_dimension(input_sequence, target_dim);
        
        // For now, use compact positional encoding directly
        Matrix result = embeddings.leftCols(std::min(target_dim, (int)embeddings.cols()));
        
        // Apply compact positional encoding
        if (compact_pos_encoder_) {
            compact_pos_encoder_->apply_to_sequence(result, target_dim);
        }
        
        return result;
    }
    
    // Memory-efficient multi-scale embeddings with positional encoding
    std::vector<Matrix> get_multiscale_embeddings(const Matrix& embeddings) {
        std::vector<Matrix> scales;
        for (int dim : {64, 128, 256, 512, 768, 1024, 1536}) {
            if (dim <= config_.embed_dim) {
                scales.push_back(apply_matryoshka_encoding(embeddings, dim));
            }
        }
        return scales;
    }
    
private:
    void initialize_positional_encoding() {
        positional_encoding_ = Matrix::Zero(config_.max_seq_length, config_.embed_dim);
        
        // Sinusoidal positional encoding
        for (int pos = 0; pos < config_.max_seq_length; ++pos) {
            for (int i = 0; i < config_.embed_dim; i += 2) {
                const float freq = 1.0f / std::pow(10000.0f, float(i) / config_.embed_dim);
                positional_encoding_(pos, i) = std::sin(pos * freq);
                if (i + 1 < config_.embed_dim) {
                    positional_encoding_(pos, i + 1) = std::cos(pos * freq);
                }
            }
        }
    }
    
    Matrix create_causal_mask(int seq_len) {
        Matrix mask(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                mask(i, j) = (j <= i) ? 1.0f : 0.0f;
            }
        }
        return mask;
    }
    
    // Matrix file format: [uint32 magic 'MWGT'][uint8 version][uint32 rows][uint32 cols][float data...]
    void save_matrix(std::ofstream& file, const Matrix& mat) {
        const uint32_t magic = 0x4D574754; // 'MWGT'
        const uint8_t version = 1;
        uint32_t rows = static_cast<uint32_t>(mat.rows());
        uint32_t cols = static_cast<uint32_t>(mat.cols());
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        file.write(reinterpret_cast<const char*>(mat.data()), sizeof(float) * rows * cols);
    }

    // load_matrix supports the new versioned format as well as a legacy raw format.
    // If a legacy (unversioned) matrix is detected, and transpose_legacy is true,
    // the loaded matrix will be transposed to match the in->out convention.
    void load_matrix(std::ifstream& file, Matrix& mat, bool transpose_legacy = true) {
        // Peek magic
        uint32_t magic = 0;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (!file) throw std::runtime_error("Failed reading matrix magic");
        if (magic == 0x4D574754) {
            // New format
            uint8_t version = 0;
            file.read(reinterpret_cast<char*>(&version), sizeof(version));
            if (version != 1) {
                throw std::runtime_error("Unsupported matrix file version");
            }
            uint32_t rows = 0, cols = 0;
            file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            mat.resize(static_cast<int>(rows), static_cast<int>(cols));
            file.read(reinterpret_cast<char*>(mat.data()), sizeof(float) * rows * cols);
        } else {
            // Legacy format: first 4 bytes are actually rows (uint32)
            uint32_t rows = magic;
            uint32_t cols = 0;
            file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            Matrix tmp(static_cast<int>(rows), static_cast<int>(cols));
            file.read(reinterpret_cast<char*>(tmp.data()), sizeof(float) * rows * cols);
            if (transpose_legacy) {
                mat = tmp.transpose();
            } else {
                mat = tmp;
            }
        }
    }
    
    void save_vector(std::ofstream& file, const Vector& vec) {
        int size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), sizeof(float) * size);
    }
    
    void load_vector(std::ifstream& file, Vector& vec) {
        int size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * size);
    }
    
    TransformerConfig config_;
    bool is_training_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    Matrix positional_encoding_;
    Matrix output_projection_;
    Matrix causal_mask_;
    Vector final_ln_gamma_, final_ln_beta_;
    
    // For gradient checkpointing
    std::vector<Matrix> checkpoint_states_;
    
    // Compact positional encoding for memory efficiency
    std::unique_ptr<CompactPositionalEncodingF32> compact_pos_encoder_;
    
    // Matryoshka encoder for multi-scale representations
    // TODO: Enable when matryoshka_encoder.cpp is implemented
    // std::unique_ptr<matryoshka::MatryoshkaEncoder> matryoshka_encoder_;
};

// Production TransformerModel wrapper
TransformerModel::TransformerModel(const TransformerConfig& config)
    : config_(config) {
    pImpl = std::make_unique<TransformerModelImpl>(config);
}

TransformerModel::~TransformerModel() = default;

Matrix TransformerModel::forward(const Matrix& embeddings, const Matrix* mask) {
    return pImpl->forward(embeddings, mask);
}

Matrix TransformerModel::get_embeddings_at_dimension(const Matrix& input, int target_dim) {
    return pImpl->apply_matryoshka_encoding(input, target_dim);
}

std::vector<float> TransformerModel::encode(const std::vector<std::vector<float>>& embeddings) {
    // Convert to Eigen matrix
    const int seq_len = embeddings.size();
    const int embed_dim = embeddings[0].size();
    
    Matrix input(seq_len, embed_dim);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            input(i, j) = embeddings[i][j];
        }
    }
    
    // Forward pass
    Matrix output = forward(input);
    
    // Convert back to vector
    std::vector<float> result;
    result.reserve(output.size());
    
    for (int i = 0; i < output.size(); ++i) {
        result.push_back(output.data()[i]);
    }
    
    return result;
}

Vector TransformerModel::get_pooled_output(const Matrix& encoded, const std::string& pooling_method) {
    if (pooling_method == "mean") {
        return encoded.colwise().mean().transpose();
    } else if (pooling_method == "max") {
        return encoded.colwise().maxCoeff().transpose();
    } else if (pooling_method == "first") {
        return encoded.row(0).transpose();
    } else if (pooling_method == "last") {
        return encoded.row(encoded.rows() - 1).transpose();
    } else if (pooling_method == "cls") {
        // Assume first token is CLS token
        return encoded.row(0).transpose();
    } else {
        // Default to mean pooling
        return encoded.colwise().mean().transpose();
    }
}

void TransformerModel::save_weights(const std::string& filepath) {
    pImpl->save_weights(filepath);
}

void TransformerModel::load_weights(const std::string& filepath) {
    pImpl->load_weights(filepath);
}

} // namespace transformer