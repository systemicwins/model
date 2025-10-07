#include "attention.h"
#include <cmath>

namespace transformer {

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, float dropout_rate)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      dropout_rate_(dropout_rate),
      scale_(1.0f / std::sqrt(static_cast<float>(embed_dim / num_heads))) {
    if (embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }

    // Initialize parameter matrices and biases
    W_q_.resize(embed_dim_, embed_dim_);
    W_k_.resize(embed_dim_, embed_dim_);
    W_v_.resize(embed_dim_, embed_dim_);
    W_o_.resize(embed_dim_, embed_dim_);

    b_q_.resize(embed_dim_);
    b_k_.resize(embed_dim_);
    b_v_.resize(embed_dim_);
    b_o_.resize(embed_dim_);

    initialize_weights();
}

MultiHeadAttention::~MultiHeadAttention() = default;

void MultiHeadAttention::initialize_weights() {
    // Kaiming-like scaling for stability
    const float scale = std::sqrt(2.0f / static_cast<float>(embed_dim_));
    W_q_ = Matrix::Random(embed_dim_, embed_dim_) * scale;
    W_k_ = Matrix::Random(embed_dim_, embed_dim_) * scale;
    W_v_ = Matrix::Random(embed_dim_, embed_dim_) * scale;
    W_o_ = Matrix::Random(embed_dim_, embed_dim_) * scale;

    b_q_ = Vector::Zero(embed_dim_);
    b_k_ = Vector::Zero(embed_dim_);
    b_v_ = Vector::Zero(embed_dim_);
    b_o_ = Vector::Zero(embed_dim_);
}

Matrix MultiHeadAttention::apply_dropout(const Matrix& input) {
    if (dropout_rate_ <= 0.0f) {
        return input;
    }
    return TensorOps::dropout(input, dropout_rate_, true);
}

Matrix MultiHeadAttention::scaled_dot_product_attention(const Matrix& Q,
                                                        const Matrix& K,
                                                        const Matrix& V,
                                                        const Matrix* mask) {
    // Q, K, V shapes: [seq_len, head_dim]
    Matrix scores = (Q * K.transpose()) * scale_;

    // Apply mask if provided (mask expected shape: [seq_len, seq_len], 1 keep, 0 mask)
    // Temporarily disabled to debug dimension issues
    // if (mask) {
    //     // Ensure mask dimensions match scores dimensions
    //     if (mask->rows() == scores.rows() && mask->cols() == scores.cols()) {
    //         // scores = scores * mask + (1 - mask) * (-1e9)
    //         Matrix ones_mask(mask->rows(), mask->cols());
    //         ones_mask.setOnes();
    //         Matrix neg_mask = ones_mask - (*mask);
    //         scores = scores.cwiseProduct(*mask) + neg_mask * (-1e9f);
    //     } else {
    //         // Create appropriate mask for this head if dimensions don't match
    //         Matrix head_mask = Matrix::Ones(scores.rows(), scores.cols());
    //         if (mask->rows() >= scores.rows() && mask->cols() >= scores.cols()) {
    //             head_mask = mask->topLeftCorner(scores.rows(), scores.cols());
    //         }
    //         Matrix ones_head = Matrix::Ones(head_mask.rows(), head_mask.cols());
    //         Matrix neg_head = ones_head - head_mask;
    //         scores = scores.cwiseProduct(head_mask) + neg_head * (-1e9f);
    //     }
    // }

    // Softmax over keys dimension
    Matrix attn_weights = TensorOps::softmax(scores, 1);

    // Apply dropout to attention weights
    attn_weights = apply_dropout(attn_weights);

    // Weighted sum of values
    Matrix output = attn_weights * V;
    return output;
}

Matrix MultiHeadAttention::forward(const Matrix& query,
                                   const Matrix& key,
                                   const Matrix& value,
                                   const Matrix* mask) {
    // Validate dimensions
    if (query.cols() != embed_dim_ || key.cols() != embed_dim_ || value.cols() != embed_dim_) {
        throw std::runtime_error("Input embedding dimension mismatch in MultiHeadAttention::forward");
    }
    if (key.rows() != value.rows()) {
        throw std::runtime_error("Key and Value sequence lengths must match");
    }

    const int seq_len_q = query.rows();
    const int seq_len_kv = key.rows();

    // Linear projections
    Matrix Q = query * W_q_;
    Q.rowwise() += RowVector(b_q_);
    Matrix K = key * W_k_;
    K.rowwise() += RowVector(b_k_);
    Matrix V = value * W_v_;
    V.rowwise() += RowVector(b_v_);

    // Output placeholder (initialize to zero to make parallel writes safe)
    Matrix concatenated_heads(seq_len_q, embed_dim_);
    concatenated_heads.setZero();

    // Process each head independently
    #pragma omp parallel for
    for (int h = 0; h < num_heads_; ++h) {
        const int col_start = h * head_dim_;
        // Slice head-specific projections
        Matrix Q_h = Q.middleCols(col_start, head_dim_);
        Matrix K_h = K.middleCols(col_start, head_dim_);
        Matrix V_h = V.middleCols(col_start, head_dim_);

        // Compute attention for this head
        Matrix head_out = scaled_dot_product_attention(Q_h, K_h, V_h, mask);

        // Write head output back
        concatenated_heads.middleCols(col_start, head_dim_) = head_out;
    }

    // Final output projection
    Matrix output = concatenated_heads * W_o_;
    output.rowwise() += RowVector(b_o_);

    // Optional dropout on output
    output = apply_dropout(output);

    return output;
}

// ============================================================================
// Gated Multi-Head Attention Implementation (GAM)
// ============================================================================

GatedMultiHeadAttention::GatedMultiHeadAttention(int embed_dim, int num_heads, float dropout_rate)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      dropout_rate_(dropout_rate),
      scale_(1.0f / std::sqrt(static_cast<float>(embed_dim / num_heads))) {

    if (embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("embed_dim must be divisible by num_heads");
    }

    // Initialize parameter matrices for standard attention
    W_q_.resize(embed_dim_, embed_dim_);
    W_k_.resize(embed_dim_, embed_dim_);
    W_v_.resize(embed_dim_, embed_dim_);
    W_o_.resize(embed_dim_, embed_dim_);

    b_q_.resize(embed_dim_);
    b_k_.resize(embed_dim_);
    b_v_.resize(embed_dim_);
    b_o_.resize(embed_dim_);

    // Initialize parameter matrices for gating mechanism
    W_gate_q_.resize(embed_dim_, embed_dim_);
    W_gate_k_.resize(embed_dim_, embed_dim_);
    W_gate_v_.resize(embed_dim_, embed_dim_);
    W_gate_out_.resize(embed_dim_, embed_dim_);

    b_gate_q_.resize(embed_dim_);
    b_gate_k_.resize(embed_dim_);
    b_gate_v_.resize(embed_dim_);
    b_gate_out_.resize(embed_dim_);

    initialize_weights();
}

GatedMultiHeadAttention::~GatedMultiHeadAttention() = default;

void GatedMultiHeadAttention::initialize_weights() {
    // Standard attention weights - Glorot uniform initialization
    W_q_.setRandom();
    W_q_ = W_q_ * std::sqrt(2.0f / (embed_dim_ + embed_dim_));

    W_k_.setRandom();
    W_k_ = W_k_ * std::sqrt(2.0f / (embed_dim_ + embed_dim_));

    W_v_.setRandom();
    W_v_ = W_v_ * std::sqrt(2.0f / (embed_dim_ + embed_dim_));

    W_o_.setRandom();
    W_o_ = W_o_ * std::sqrt(2.0f / (embed_dim_ + embed_dim_));

    // Gate weights - smaller initialization for better gating behavior
    W_gate_q_.setRandom();
    W_gate_q_ = W_gate_q_ * std::sqrt(1.0f / (embed_dim_ + embed_dim_));

    W_gate_k_.setRandom();
    W_gate_k_ = W_gate_k_ * std::sqrt(1.0f / (embed_dim_ + embed_dim_));

    W_gate_v_.setRandom();
    W_gate_v_ = W_gate_v_ * std::sqrt(1.0f / (embed_dim_ + embed_dim_));

    W_gate_out_.setRandom();
    W_gate_out_ = W_gate_out_ * std::sqrt(1.0f / (embed_dim_ + embed_dim_));

    // Biases to zero
    b_q_.setZero();
    b_k_.setZero();
    b_v_.setZero();
    b_o_.setZero();

    b_gate_q_.setZero();
    b_gate_k_.setZero();
    b_gate_v_.setZero();
    b_gate_out_.setZero();
}

Matrix GatedMultiHeadAttention::compute_gate(const Matrix& Q, const Matrix& K, const Matrix& V) {
    // Compute gate projections using full matrices
    // Q, K, V are full embedding dimension matrices (not head-split)
    const int seq_len = Q.rows();

    // Compute gate projections directly on full matrices
    // Using weight layout: W is in->out (input * W)
    Matrix Q_gate = Q * W_gate_q_;
    Q_gate.rowwise() += RowVector(b_gate_q_);
    Matrix K_gate = K * W_gate_k_;
    K_gate.rowwise() += RowVector(b_gate_k_);

    // Gate computation - sigmoid of linear combination
    // This controls which information flows through the attention
    Matrix gate_scores = Q_gate * K_gate.transpose();

    // Apply sigmoid to get gate values between 0 and 1
    Matrix gate = TensorOps::sigmoid(gate_scores);

    return gate;
}

Matrix GatedMultiHeadAttention::apply_gate(const Matrix& attention_output, const Matrix& gate) {
    // Apply gate to control information flow
    Matrix gated_output = attention_output.cwiseProduct(gate);

    // Final gate output projection
    // Project final gated output using W_gate_out_ with in->out layout
    Matrix final_output = gated_output * W_gate_out_;
    final_output.rowwise() += RowVector(b_gate_out_);

    return final_output;
}

Matrix GatedMultiHeadAttention::compute_gated_attention(const Matrix& Q,
                                                       const Matrix& K,
                                                       const Matrix& V,
                                                       const Matrix* mask) {
    // Compute gate first using full matrices
    Matrix gate = compute_gate(Q, K, V);

    // Standard attention computation
    Matrix scores = (Q * K.transpose()) * scale_;

    // Apply mask if provided (causal mask for autoregressive)
    if (mask) {
        // Ensure mask dimensions match scores dimensions
        if (mask->rows() == scores.rows() && mask->cols() == scores.cols()) {
            // scores = scores * mask + (1 - mask) * (-1e9)
            Matrix ones_mask(mask->rows(), mask->cols());
            ones_mask.setOnes();
            Matrix neg_mask = ones_mask - (*mask);
            scores = scores.cwiseProduct(*mask) + neg_mask * (-1e9f);
        } else {
            // Create appropriate mask for this head if dimensions don't match
            Matrix head_mask = Matrix::Ones(scores.rows(), scores.cols());
            if (mask->rows() >= scores.rows() && mask->cols() >= scores.cols()) {
                head_mask = mask->topLeftCorner(scores.rows(), scores.cols());
            }
            Matrix ones_head = Matrix::Ones(head_mask.rows(), head_mask.cols());
            Matrix neg_head = ones_head - head_mask;
            scores = scores.cwiseProduct(head_mask) + neg_head * (-1e9f);
        }
    }

    // Apply gate to attention weights before computing output
    Matrix attn_weights = TensorOps::softmax(scores, 1);

    // Apply gate to attention weights
    Matrix gated_attn_weights = attn_weights.cwiseProduct(gate);

    // Apply dropout to gated attention weights
    gated_attn_weights = apply_dropout(gated_attn_weights);

    // Weighted sum of values with gated attention
    Matrix attention_output = gated_attn_weights * V;

    return attention_output;
}

Matrix GatedMultiHeadAttention::apply_dropout(const Matrix& input) {
    if (dropout_rate_ <= 0.0f) {
        return input;
    }
    return TensorOps::dropout(input, dropout_rate_, true);
}

Matrix GatedMultiHeadAttention::forward(const Matrix& query,
                                        const Matrix& key,
                                        const Matrix& value,
                                        const Matrix* mask) {
    // Validate dimensions
    if (query.cols() != embed_dim_ || key.cols() != embed_dim_ || value.cols() != embed_dim_) {
        throw std::runtime_error("Input embedding dimension mismatch in GatedMultiHeadAttention::forward");
    }

    // Project to Q, K, V using W as in->out
    Matrix Q = query * W_q_;
    Q.rowwise() += RowVector(b_q_);
    Matrix K = key * W_k_;
    K.rowwise() += RowVector(b_k_);
    Matrix V = value * W_v_;
    V.rowwise() += RowVector(b_v_);

    // Apply gated attention and final projection (W_o_ uses in->out layout)
    Matrix output = compute_gated_attention(Q, K, V, mask);
    output = output * W_o_;
    output.rowwise() += RowVector(b_o_);

    return output;
}

} // namespace transformer