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

// Forward declarations (definitions are in transformer.h)
struct MambaConfig;
class MambaBlock;
class MambaModel;

} // namespace mamba

#endif // MAMBA_H