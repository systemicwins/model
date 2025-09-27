#ifndef COMPACT_POSITIONAL_ENCODING_F32_H
#define COMPACT_POSITIONAL_ENCODING_F32_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <memory>
#include <iostream>
#ifdef __x86_64__
#include <immintrin.h>  // For SIMD with float32 (x86 only)
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>  // For SIMD on ARM
#endif

namespace transformer {

using Scalar = float;  // Maintain float32 precision
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/**
 * Memory-efficient positional encoding with full float32 precision
 * Optimized for financial data where precision is critical
 */
class CompactPositionalEncodingF32 {
public:
    enum EncodingType {
        SINUSOIDAL,           // Standard transformer
        ROTARY,               // RoPE - better for sequential financial data
        LEARNED_SHARED,       // Shared learned base + dimension projections
        HIERARCHICAL          // Hierarchical encoding for multi-scale patterns
    };
    
    struct Config {
        int max_seq_length;
        int max_embed_dim;
        EncodingType type;
        bool cache_computed;
        int cache_size;
        bool use_shared_base;
        
        Config() : max_seq_length(8191), max_embed_dim(1536), type(ROTARY),
                   cache_computed(true), cache_size(512), use_shared_base(true) {}
    };
    
    explicit CompactPositionalEncodingF32(const Config& config)
        : config_(config) {
        
        initialize_frequency_bases();
        
        if (config.cache_computed) {
            initialize_cache();
        }
        
        if (config.use_shared_base) {
            initialize_shared_base();
        }
    }
    
    /**
     * Get positional encoding with full float32 precision
     * Uses caching for frequently accessed positions
     */
    Vector get_position_encoding(int position, int embed_dim) const {
        // Check cache first
        if (position < cached_positions_.rows() && embed_dim <= config_.max_embed_dim) {
            return cached_positions_.row(position).head(embed_dim).transpose();
        }
        
        // Compute on-the-fly for uncached positions
        Vector encoding(embed_dim);
        compute_encoding(position, embed_dim, encoding.data());
        return encoding;
    }
    
    /**
     * Apply positional encoding to sequence (maintains float32 precision)
     */
    void apply_to_sequence(Matrix& sequence, int target_dim = -1) {
        const int seq_len = sequence.rows();
        const int embed_dim = (target_dim > 0) ? 
                              std::min(target_dim, static_cast<int>(sequence.cols())) : 
                              sequence.cols();
        
        // Use cached positions when possible
        if (seq_len <= cached_positions_.rows()) {
            sequence.leftCols(embed_dim) += cached_positions_.topRows(seq_len).leftCols(embed_dim);
        } else {
            // Compute for positions beyond cache
            #pragma omp parallel for
            for (int pos = 0; pos < seq_len; ++pos) {
                if (pos < cached_positions_.rows()) {
                    sequence.row(pos).head(embed_dim) += cached_positions_.row(pos).head(embed_dim);
                } else {
                    Vector encoding(embed_dim);
                    compute_encoding(pos, embed_dim, encoding.data());
                    sequence.row(pos).head(embed_dim) += encoding.transpose();
                }
            }
        }
    }
    
    /**
     * Shared base encoding with dimension-specific projections
     * Reduces memory while maintaining precision
     */
    class SharedBaseEncoding {
    public:
        SharedBaseEncoding(int max_seq, int base_dim, const std::vector<int>& target_dims) 
            : base_dim_(base_dim) {
            
            // Store one base encoding at maximum granularity
            base_encoding_ = compute_base_sinusoidal(max_seq, base_dim);
            
            // Store small projection matrices for each target dimension
            for (int dim : target_dims) {
                if (dim < base_dim) {
                    // Simple truncation for smaller dimensions
                    projection_matrices_[dim] = Matrix::Identity(dim, base_dim).leftCols(dim);
                } else if (dim > base_dim) {
                    // Learned projection for larger dimensions (shouldn't happen typically)
                    projection_matrices_[dim] = Matrix::Random(dim, base_dim) * 
                                               std::sqrt(2.0f / base_dim);
                }
            }
        }
        
        Vector get_position(int pos, int target_dim) const {
            if (pos >= base_encoding_.rows()) {
                // Compute on-the-fly for positions beyond pre-computed range
                return compute_position_sinusoidal(pos, target_dim);
            }
            
            if (target_dim <= base_dim_) {
                // Simple truncation
                return base_encoding_.row(pos).head(target_dim).transpose();
            } else if (projection_matrices_.count(target_dim)) {
                // Apply projection
                return projection_matrices_.at(target_dim) * 
                       base_encoding_.row(pos).transpose();
            } else {
                // Fallback to computation
                return compute_position_sinusoidal(pos, target_dim);
            }
        }
        
        size_t memory_bytes() const {
            size_t total = base_encoding_.size() * sizeof(Scalar);
            for (const auto& [dim, mat] : projection_matrices_) {
                total += mat.size() * sizeof(Scalar);
            }
            return total;
        }
        
    private:
        int base_dim_;
        Matrix base_encoding_;  // [max_seq, base_dim] - shared across all dimensions
        std::unordered_map<int, Matrix> projection_matrices_;  // Small projections
        
        static Matrix compute_base_sinusoidal(int max_seq, int embed_dim) {
            Matrix encoding(max_seq, embed_dim);
            
            for (int pos = 0; pos < max_seq; ++pos) {
                for (int i = 0; i < embed_dim; i += 2) {
                    Scalar angle = pos / std::pow(10000.0f, Scalar(i) / embed_dim);
                    encoding(pos, i) = std::sin(angle);
                    if (i + 1 < embed_dim) {
                        encoding(pos, i + 1) = std::cos(angle);
                    }
                }
            }
            
            return encoding;
        }
        
        static Vector compute_position_sinusoidal(int pos, int embed_dim) {
            Vector encoding(embed_dim);
            
            for (int i = 0; i < embed_dim; i += 2) {
                Scalar angle = pos / std::pow(10000.0f, Scalar(i) / embed_dim);
                encoding(i) = std::sin(angle);
                if (i + 1 < embed_dim) {
                    encoding(i + 1) = std::cos(angle);
                }
            }
            
            return encoding;
        }
    };
    
    /**
     * RoPE with float32 precision - critical for maintaining numerical stability
     */
    void apply_rope_f32(Matrix& sequence, int target_dim = -1) {
        const int seq_len = sequence.rows();
        const int embed_dim = (target_dim > 0) ? 
                              std::min(target_dim, static_cast<int>(sequence.cols())) : 
                              sequence.cols();
        
        #pragma omp parallel for
        for (int pos = 0; pos < seq_len; ++pos) {
            Scalar* row_data = sequence.row(pos).data();
            
            // Process pairs with full float32 precision
            for (int i = 0; i < embed_dim - 1; i += 2) {
                // High precision angle computation
                const Scalar theta = compute_rope_angle_precise(pos, i, embed_dim);
                const Scalar cos_theta = std::cos(theta);
                const Scalar sin_theta = std::sin(theta);
                
                const Scalar x0 = row_data[i];
                const Scalar x1 = row_data[i + 1];
                
                // Maintain precision in rotation
                row_data[i]     = x0 * cos_theta - x1 * sin_theta;
                row_data[i + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
    
    /**
     * Hierarchical encoding for multi-scale financial patterns
     * (daily, weekly, monthly, quarterly patterns)
     */
    class HierarchicalEncoding {
    public:
        HierarchicalEncoding(int max_seq, int embed_dim) {
            // Different scales for financial time series
            scales_ = {1, 5, 21, 63, 252};  // day, week, month, quarter, year (trading days)
            
            for (int scale : scales_) {
                scale_encodings_[scale] = compute_scale_encoding(max_seq, embed_dim, scale);
            }
        }
        
        Vector get_position(int pos, int dim) const {
            Vector combined = Vector::Zero(dim);
            
            for (int scale : scales_) {
                if (scale_encodings_.count(scale)) {
                    int scaled_pos = pos / scale;
                    if (scaled_pos < scale_encodings_.at(scale).rows()) {
                        combined += scale_encodings_.at(scale).row(scaled_pos).head(dim).transpose() / 
                                   std::sqrt(static_cast<Scalar>(scales_.size()));
                    }
                }
            }
            
            return combined;
        }
        
    private:
        std::vector<int> scales_;
        std::unordered_map<int, Matrix> scale_encodings_;
        
        static Matrix compute_scale_encoding(int max_seq, int embed_dim, int scale) {
            int scaled_seq = (max_seq + scale - 1) / scale;
            Matrix encoding(scaled_seq, embed_dim);
            
            for (int pos = 0; pos < scaled_seq; ++pos) {
                for (int i = 0; i < embed_dim; i += 2) {
                    Scalar angle = (pos * scale) / std::pow(10000.0f, Scalar(i) / embed_dim);
                    encoding(pos, i) = std::sin(angle);
                    if (i + 1 < embed_dim) {
                        encoding(pos, i + 1) = std::cos(angle);
                    }
                }
            }
            
            return encoding;
        }
    };
    
    /**
     * Memory usage report
     */
    struct MemoryUsage {
        size_t cached_positions;
        size_t frequency_bases;
        size_t shared_base;
        size_t total;
        
        void print() const {
            std::cout << "Positional Encoding Memory Usage (float32):\n";
            std::cout << "  Cached positions: " << cached_positions / (1024.0f * 1024.0f) << " MB\n";
            std::cout << "  Frequency bases: " << frequency_bases / 1024.0f << " KB\n";
            std::cout << "  Shared base: " << shared_base / (1024.0f * 1024.0f) << " MB\n";
            std::cout << "  Total: " << total / (1024.0f * 1024.0f) << " MB\n";
        }
    };
    
    MemoryUsage get_memory_usage() const {
        MemoryUsage usage;
        usage.cached_positions = cached_positions_.size() * sizeof(Scalar);
        usage.frequency_bases = freq_bases_.size() * sizeof(Scalar);
        usage.shared_base = shared_base_ ? shared_base_->memory_bytes() : 0;
        usage.total = usage.cached_positions + usage.frequency_bases + usage.shared_base;
        return usage;
    }
    
private:
    Config config_;
    
    // Cached frequently used positions (first N positions)
    Matrix cached_positions_;
    
    // Pre-computed frequency bases for fast computation
    std::vector<Scalar> freq_bases_;
    
    // Shared base encoding
    std::unique_ptr<SharedBaseEncoding> shared_base_;
    
    // Optional hierarchical encoding for financial patterns
    std::unique_ptr<HierarchicalEncoding> hierarchical_;
    
    void initialize_frequency_bases() {
        freq_bases_.reserve(config_.max_embed_dim / 2);
        for (int i = 0; i < config_.max_embed_dim; i += 2) {
            freq_bases_.push_back(1.0f / std::pow(10000.0f, Scalar(i) / config_.max_embed_dim));
        }
    }
    
    void initialize_cache() {
        const int cache_size = std::min(config_.cache_size, config_.max_seq_length);
        cached_positions_ = Matrix(cache_size, config_.max_embed_dim);
        
        for (int pos = 0; pos < cache_size; ++pos) {
            compute_encoding(pos, config_.max_embed_dim, cached_positions_.row(pos).data());
        }
    }
    
    void initialize_shared_base() {
        std::vector<int> target_dims = {64, 128, 256, 512, 768, 1024, 1536};
        shared_base_ = std::make_unique<SharedBaseEncoding>(
            config_.max_seq_length, 
            config_.max_embed_dim,
            target_dims
        );
    }
    
    inline void compute_encoding(int pos, int dim, Scalar* output) const {
        switch (config_.type) {
            case SINUSOIDAL:
                compute_sinusoidal_f32(pos, dim, output);
                break;
            case ROTARY:
                // RoPE is applied as rotation, not additive
                std::fill(output, output + dim, 0.0f);
                break;
            case LEARNED_SHARED:
                if (shared_base_) {
                    Vector enc = shared_base_->get_position(pos, dim);
                    std::memcpy(output, enc.data(), dim * sizeof(Scalar));
                }
                break;
            case HIERARCHICAL:
                if (hierarchical_) {
                    Vector enc = hierarchical_->get_position(pos, dim);
                    std::memcpy(output, enc.data(), dim * sizeof(Scalar));
                }
                break;
        }
    }
    
    inline void compute_sinusoidal_f32(int pos, int dim, Scalar* output) const {
        // Maintain full float32 precision in computation
        for (int i = 0; i < dim; i += 2) {
            const Scalar angle = pos * freq_bases_[i / 2];
            output[i] = std::sin(angle);
            if (i + 1 < dim) {
                output[i + 1] = std::cos(angle);
            }
        }
    }
    
    inline Scalar compute_rope_angle_precise(int pos, int dim_idx, int embed_dim) const {
        // High precision computation for financial data
        return static_cast<Scalar>(pos) / std::pow(10000.0f, 2.0f * dim_idx / embed_dim);
    }
};

} // namespace transformer

#endif // COMPACT_POSITIONAL_ENCODING_F32_H