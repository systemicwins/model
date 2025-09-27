#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <Eigen/Dense>
#include "tensor_ops.h"
namespace transformer {


class FeedForward {
public:
    FeedForward(int embed_dim, int ff_dim, float dropout_rate = 0.1f);
    ~FeedForward();

    Matrix forward(const Matrix& input);
    void initialize_weights();

private:
    int embed_dim_;
    int ff_dim_;
    float dropout_rate_;

    // Weight matrices for two linear layers
    Matrix W1_, W2_;
    Vector b1_, b2_;

    Matrix gelu_activation(const Matrix& input);
    Matrix apply_dropout(const Matrix& input);
};

} // namespace transformer

#endif // FEEDFORWARD_H