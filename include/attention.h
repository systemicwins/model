#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>
#include "tensor_ops.h"
#include <memory>

namespace transformer {


class MultiHeadAttention {
public:
    MultiHeadAttention(int embed_dim, int num_heads, float dropout_rate = 0.1f);
    ~MultiHeadAttention();

    Matrix forward(const Matrix& query,
                   const Matrix& key,
                   const Matrix& value,
                   const Matrix* mask = nullptr);

    void initialize_weights();

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    float dropout_rate_;
    float scale_;

    // Weight matrices
    Matrix W_q_, W_k_, W_v_, W_o_;
    Vector b_q_, b_k_, b_v_, b_o_;

    Matrix scaled_dot_product_attention(const Matrix& Q,
                                       const Matrix& K,
                                       const Matrix& V,
                                       const Matrix* mask = nullptr);

    Matrix apply_dropout(const Matrix& input);
};

// Gated Attention Mechanism (GAM) for financial data
class GatedMultiHeadAttention {
public:
    GatedMultiHeadAttention(int embed_dim, int num_heads, float dropout_rate = 0.1f);
    ~GatedMultiHeadAttention();

    Matrix forward(const Matrix& query,
                   const Matrix& key,
                   const Matrix& value,
                   const Matrix* mask = nullptr);

    void initialize_weights();

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    float dropout_rate_;
    float scale_;

    // Standard attention weights
    Matrix W_q_, W_k_, W_v_, W_o_;
    Vector b_q_, b_k_, b_v_, b_o_;

    // Gate-specific weights for controlling information flow
    Matrix W_gate_q_, W_gate_k_, W_gate_v_;
    Vector b_gate_q_, b_gate_k_, b_gate_v_;
    Matrix W_gate_out_;
    Vector b_gate_out_;

    // Gated attention computation
    Matrix compute_gated_attention(const Matrix& Q,
                                  const Matrix& K,
                                  const Matrix& V,
                                  const Matrix* mask = nullptr);

    Matrix compute_gate(const Matrix& Q, const Matrix& K, const Matrix& V);
    Matrix apply_gate(const Matrix& attention_output, const Matrix& gate);

    Matrix apply_dropout(const Matrix& input);
};

} // namespace transformer

#endif // ATTENTION_H