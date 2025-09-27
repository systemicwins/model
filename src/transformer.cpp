#include "transformer.h"
#include "attention.h"
#include "feedforward.h"
#include "layer_norm.h"
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>

namespace transformer {

TransformerBlock::TransformerBlock(const TransformerConfig& config)
    : config_(config) {
    attention_ = std::make_unique<MultiHeadAttention>(config.embed_dim, config.num_heads, config.dropout_rate);
    norm1_ = std::make_unique<LayerNorm>(config.embed_dim, config.layer_norm_eps);
    norm2_ = std::make_unique<LayerNorm>(config.embed_dim, config.layer_norm_eps);
    ffn_ = std::make_unique<FeedForward>(config.embed_dim, config.ff_dim, config.dropout_rate);
}

TransformerBlock::~TransformerBlock() = default;

Matrix TransformerBlock::forward(const Matrix& input, const Matrix* mask) {
    // Self-attention with residual connection
    Matrix attn_output = attention_->forward(input, input, input, mask);
    Matrix x = norm1_->forward(input + attn_output);
    
    // Feed-forward with residual connection
    Matrix ff_output = ffn_->forward(x);
    Matrix output = norm2_->forward(x + ff_output);
    
    return output;
}

TransformerModel::TransformerModel(const TransformerConfig& config)
    : config_(config) {
    
    // Initialize transformer blocks
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.push_back(std::make_unique<TransformerBlock>(config));
    }
    
    final_norm_ = std::make_unique<LayerNorm>(config.embed_dim, config.layer_norm_eps);
    
    // Initialize positional encoding
    initialize_positional_encoding();
    
    // Initialize output projection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / config.embed_dim));
    
    output_projection_ = Matrix::Zero(config.embed_dim, config.embed_dim);
    for (int i = 0; i < config.embed_dim; ++i) {
        for (int j = 0; j < config.embed_dim; ++j) {
            output_projection_(i, j) = dist(gen);
        }
    }
}

TransformerModel::~TransformerModel() = default;

void TransformerModel::initialize_positional_encoding() {
    positional_encoding_ = Matrix::Zero(config_.max_seq_length, config_.embed_dim);
    
    for (int pos = 0; pos < config_.max_seq_length; ++pos) {
        for (int i = 0; i < config_.embed_dim; ++i) {
            if (i % 2 == 0) {
                positional_encoding_(pos, i) = 
                    std::sin(pos / std::pow(10000.0f, float(i) / config_.embed_dim));
            } else {
                positional_encoding_(pos, i) = 
                    std::cos(pos / std::pow(10000.0f, float(i - 1) / config_.embed_dim));
            }
        }
    }
}

Matrix TransformerModel::forward(const Matrix& embeddings, const Matrix* mask) {
    int seq_len = embeddings.rows();
    
    // Add positional encoding
    Matrix x = embeddings;
    if (seq_len <= config_.max_seq_length) {
        x += positional_encoding_.topRows(seq_len);
    }
    
    // Pass through transformer blocks
    for (auto& layer : layers_) {
        x = layer->forward(x, mask);
    }
    
    // Final normalization
    x = final_norm_->forward(x);
    
    // Output projection
    Matrix output = x * output_projection_.transpose();
    
    return output;
}

std::vector<float> TransformerModel::encode(const std::vector<std::vector<float>>& embeddings) {
    // Convert input to Eigen matrix
    int seq_len = embeddings.size();
    int embed_dim = embeddings[0].size();
    
    Matrix input(seq_len, embed_dim);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            input(i, j) = embeddings[i][j];
        }
    }
    
    // Forward pass
    Matrix output = forward(input);
    
    // Convert output to vector
    std::vector<float> result;
    result.reserve(output.rows() * output.cols());
    
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            result.push_back(output(i, j));
        }
    }
    
    return result;
}

Vector TransformerModel::get_pooled_output(const Matrix& encoded, const std::string& pooling_method) {
    if (pooling_method == "mean") {
        return encoded.colwise().mean();
    } else if (pooling_method == "max") {
        return encoded.colwise().maxCoeff();
    } else if (pooling_method == "first") {
        return encoded.row(0);
    } else if (pooling_method == "last") {
        return encoded.row(encoded.rows() - 1);
    } else {
        // Default to mean pooling
        return encoded.colwise().mean();
    }
}

void TransformerModel::save_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    // Save configuration
    file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
    
    // Save output projection matrix
    int rows = output_projection_.rows();
    int cols = output_projection_.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(output_projection_.data()), 
               sizeof(float) * rows * cols);
    
    file.close();
}

void TransformerModel::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }
    
    // Load configuration
    file.read(reinterpret_cast<char*>(&config_), sizeof(config_));
    
    // Load output projection matrix
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    output_projection_.resize(rows, cols);
    file.read(reinterpret_cast<char*>(output_projection_.data()), 
              sizeof(float) * rows * cols);
    
    file.close();
}

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, float dropout_rate)
    : embed_dim_(embed_dim), num_heads_(num_heads), dropout_rate_(dropout_rate) {
    head_dim_ = embed_dim / num_heads;
    scale_ = 1.0f / std::sqrt(float(head_dim_));
    initialize_weights();
}

MultiHeadAttention::~MultiHeadAttention() = default;

void MultiHeadAttention::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim_));
    
    W_q_ = Matrix::Zero(embed_dim_, embed_dim_);
    W_k_ = Matrix::Zero(embed_dim_, embed_dim_);
    W_v_ = Matrix::Zero(embed_dim_, embed_dim_);
    W_o_ = Matrix::Zero(embed_dim_, embed_dim_);
    
    b_q_ = Vector::Zero(embed_dim_);
    b_k_ = Vector::Zero(embed_dim_);
    b_v_ = Vector::Zero(embed_dim_);
    b_o_ = Vector::Zero(embed_dim_);
    
    for (int i = 0; i < embed_dim_; ++i) {
        for (int j = 0; j < embed_dim_; ++j) {
            W_q_(i, j) = dist(gen);
            W_k_(i, j) = dist(gen);
            W_v_(i, j) = dist(gen);
            W_o_(i, j) = dist(gen);
        }
    }
}

Matrix MultiHeadAttention::forward(const Matrix& query, 
                                  const Matrix& key, 
                                  const Matrix& value,
                                  const Matrix* mask) {
    int batch_size = query.rows();
    
    // Linear projections
    Matrix Q = (query * W_q_.transpose()).rowwise() + b_q_.transpose();
    Matrix K = (key * W_k_.transpose()).rowwise() + b_k_.transpose();
    Matrix V = (value * W_v_.transpose()).rowwise() + b_v_.transpose();
    
    // Reshape for multi-head attention
    // Note: Simplified implementation - in production, would use proper tensor operations
    Matrix attention_output = scaled_dot_product_attention(Q, K, V, mask);
    
    // Output projection
    Matrix output = (attention_output * W_o_.transpose()).rowwise() + b_o_.transpose();
    
    return apply_dropout(output);
}

Matrix MultiHeadAttention::scaled_dot_product_attention(const Matrix& Q, 
                                                       const Matrix& K, 
                                                       const Matrix& V,
                                                       const Matrix* mask) {
    // Compute attention scores
    Matrix scores = (Q * K.transpose()) * scale_;
    
    // Apply mask if provided
    if (mask != nullptr) {
        scores = scores.array() + mask->array();
    }
    
    // Softmax
    Matrix attention_weights = scores.array().exp();
    Vector row_sums = attention_weights.rowwise().sum();
    for (int i = 0; i < attention_weights.rows(); ++i) {
        attention_weights.row(i) /= row_sums(i);
    }
    
    // Apply attention to values
    return attention_weights * V;
}

Matrix MultiHeadAttention::apply_dropout(const Matrix& input) {
    if (dropout_rate_ <= 0.0f) {
        return input;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    Matrix mask = Matrix::Ones(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            if (dist(gen) < dropout_rate_) {
                mask(i, j) = 0.0f;
            }
        }
    }
    
    return input.array() * mask.array() / (1.0f - dropout_rate_);
}

// FeedForward implementation
FeedForward::FeedForward(int embed_dim, int ff_dim, float dropout_rate)
    : embed_dim_(embed_dim), ff_dim_(ff_dim), dropout_rate_(dropout_rate) {
    initialize_weights();
}

FeedForward::~FeedForward() = default;

void FeedForward::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / embed_dim_));
    
    W1_ = Matrix::Zero(embed_dim_, ff_dim_);
    W2_ = Matrix::Zero(ff_dim_, embed_dim_);
    b1_ = Vector::Zero(ff_dim_);
    b2_ = Vector::Zero(embed_dim_);
    
    for (int i = 0; i < embed_dim_; ++i) {
        for (int j = 0; j < ff_dim_; ++j) {
            W1_(i, j) = dist(gen);
        }
    }
    
    for (int i = 0; i < ff_dim_; ++i) {
        for (int j = 0; j < embed_dim_; ++j) {
            W2_(i, j) = dist(gen);
        }
    }
}

Matrix FeedForward::forward(const Matrix& input) {
    // First linear layer + GELU
    Matrix hidden = (input * W1_.transpose()).rowwise() + b1_.transpose();
    hidden = gelu_activation(hidden);
    hidden = apply_dropout(hidden);
    
    // Second linear layer
    Matrix output = (hidden * W2_.transpose()).rowwise() + b2_.transpose();
    
    return output;
}

Matrix FeedForward::gelu_activation(const Matrix& input) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    Matrix x3 = input.array().cube();
    Matrix inner = sqrt_2_over_pi * (input.array() + 0.044715f * x3.array());
    Matrix tanh_inner = inner.array().tanh();
    return 0.5f * input.array() * (1.0f + tanh_inner.array());
}

Matrix FeedForward::apply_dropout(const Matrix& input) {
    if (dropout_rate_ <= 0.0f) {
        return input;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    Matrix mask = Matrix::Ones(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            if (dist(gen) < dropout_rate_) {
                mask(i, j) = 0.0f;
            }
        }
    }
    
    return input.array() * mask.array() / (1.0f - dropout_rate_);
}

// LayerNorm implementation
LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape_(normalized_shape), eps_(eps) {
    initialize_parameters();
}

LayerNorm::~LayerNorm() = default;

void LayerNorm::initialize_parameters() {
    gamma_ = Vector::Ones(normalized_shape_);
    beta_ = Vector::Zero(normalized_shape_);
}

Matrix LayerNorm::forward(const Matrix& input) {
    // Compute mean and variance along the last dimension
    Vector mean = input.rowwise().mean();
    Matrix centered = input;
    for (int i = 0; i < input.rows(); ++i) {
        centered.row(i) = input.row(i).array() - mean(i);
    }
    
    Vector variance = centered.array().square().rowwise().mean();
    
    // Normalize
    Matrix normalized = Matrix::Zero(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); ++i) {
        float std_dev = std::sqrt(variance(i) + eps_);
        normalized.row(i) = centered.row(i) / std_dev;
    }
    
    // Scale and shift
    Matrix output = Matrix::Zero(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); ++i) {
        output.row(i) = (normalized.row(i).array() * gamma_.array() + beta_.array()).matrix();
    }
    
    return output;
}

} // namespace transformer