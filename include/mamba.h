#ifndef MAMBA_H
#define MAMBA_H

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "tensor_ops.h"
#include "transformer.h"

// HIP/ROCm includes for GPU support
#ifdef HIP_ENABLED
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hipfft.h>
#include <rocrand/rocrand.h>
#endif

namespace mamba {

// Import tensor types from transformer namespace
using Matrix = transformer::Matrix;
using Vector = transformer::Vector;
using Scalar = transformer::Scalar;

// Multi-GPU support
#ifdef HIP_ENABLED
struct GPUDevice {
    int device_id;
    hipDeviceProp_t properties;
    size_t memory_available;
    bool is_primary;
};

class MultiGPUManager {
private:
    std::vector<GPUDevice> devices;
    int primary_device;
    int secondary_device;

public:
    MultiGPUManager();
    ~MultiGPUManager();

    bool initialize();
    bool set_device(int device_id);
    bool split_workload(const Matrix& input, Matrix& output_primary, Matrix& output_secondary);
    size_t get_total_memory();
    size_t get_device_memory(int device_id);
    int get_device_count() { return devices.size(); }
    int get_primary_device() { return primary_device; }
    int get_secondary_device() { return secondary_device; }

private:
    bool discover_devices();
    bool allocate_device_memory();
};
#endif

// MambaConfig struct definition
struct MambaConfig {
    int num_layers = 4;
    int state_dim = 64;
    int embed_dim = 1536;
    int vocab_size = 50000;
    int max_seq_length = 100000;
    float dropout_rate = 0.1f;
};

// MambaBlock class definition
class MambaBlock {
public:
    explicit MambaBlock(const MambaConfig& config);
    ~MambaBlock();

    transformer::Matrix forward(const transformer::Matrix& input,
                               const transformer::Matrix* mask = nullptr);
    void set_training(bool training);

private:
    MambaConfig config_;
};

// Forward declarations
struct MambaConfig;
class MambaBlock;
class MambaModel;

// MambaBlock class definition
class MambaBlock {
public:
    explicit MambaBlock(const MambaConfig& config);
    ~MambaBlock();

    transformer::Matrix forward(const transformer::Matrix& input,
                               const transformer::Matrix* mask = nullptr);
    void set_training(bool training);

private:
    MambaConfig config_;
};

// MambaModel class definition
class MambaModel {
public:
    explicit MambaModel(const MambaConfig& config);
    ~MambaModel();

    transformer::Matrix forward(const transformer::Matrix& embeddings,
                               const transformer::Matrix* mask = nullptr);
    std::vector<float> encode(const std::vector<std::vector<float>>& embeddings);
    transformer::Vector get_pooled_output(const transformer::Matrix& encoded,
                                         const std::string& pooling_method = "mean");
    transformer::Matrix get_embeddings_at_dimension(const transformer::Matrix& input,
                                                    int target_dim);
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);

private:
    MambaConfig config_;
};

} // namespace mamba

#endif // MAMBA_H