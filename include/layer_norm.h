#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <Eigen/Dense>
#include "tensor_ops.h"

namespace transformer {

class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-6f);
    ~LayerNorm();

    Matrix forward(const Matrix& input);
    void initialize_parameters();

private:
    int normalized_shape_;
    float eps_;
    
    // Learnable parameters
    Vector gamma_;  // scale
    Vector beta_;   // shift
};

} // namespace transformer

#endif // LAYER_NORM_H