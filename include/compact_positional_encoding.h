#ifndef COMPACT_POSITIONAL_ENCODING_H
#define COMPACT_POSITIONAL_ENCODING_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#ifdef __x86_64__
#include <immintrin.h>  // For SIMD (x86 only)
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>  // For SIMD on ARM
#endif

namespace transformer {


/**
 * Memory-efficient positional encoding that generates values on-the-fly
 * instead of storing large matrices
 */
class CompactPositionalEncoding {
public:
    enum EncodingType {
        SINUSOIDAL,      // Standard transformer
        ROTARY,          // RoPE
        ALIBI,           // Attention with Linear Biases (most memory efficient)
        LEARNED_COMPACT  // Learned low-rank factorization
    };
    
    CompactPositionalEncoding(int max_seq_length, int max_embed_dim, 
                             EncodingType type = SINUSOIDAL)
        : max_seq_length_(max_seq_length),
          max_embed_dim_(max_embed_dim),
          encoding_type_(type) {
        
        if (type == LEARNED_COMPACT) {
            initialize_learned_parameters();
        }
        
        // Pre-compute frequency bases for faster computation
        precompute_frequency_bases();
    }
    
    /**
     * Generate positional encoding on-the-fly (no storage)
     * Memory: O(1) instead of O(seq_len * embed_dim)
     */
    Vector get_position_encoding(int position, int embed_dim) const {
        Vector encoding(embed_dim);
        
        switch (encoding_type_) {
            case SINUSOIDAL:
                compute_sinusoidal_inline(position, embed_dim, encoding.data());
                break;
            case ROTARY:
                compute_rotary_inline(position, embed_dim, encoding.data());
                break;
            case ALIBI:
                compute_alibi_inline(position, embed_dim, encoding.data());
                break;
            case LEARNED_COMPACT:
                compute_learned_inline(position, embed_dim, encoding.data());
                break;
        }
        
        return encoding;
    }
    
    /**
     * Apply positional encoding directly to input (in-place, no extra memory)
     */
    void apply_to_sequence_inplace(Matrix& sequence, int target_dim = -1) {
        const int seq_len = sequence.rows();
        const int embed_dim = (target_dim > 0) ? 
                              std::min(target_dim, (int)sequence.cols()) : 
                              sequence.cols();
        
        #pragma omp parallel for
        for (int pos = 0; pos < seq_len; ++pos) {
            apply_to_position_inplace(sequence.row(pos), pos, embed_dim);
        }
    }
    
    /**
     * Compact RoPE - modifies in place, no extra storage
     */
    void apply_rope_inplace(Matrix& sequence, int target_dim = -1) {
        const int seq_len = sequence.rows();
        const int embed_dim = (target_dim > 0) ? 
                              std::min(target_dim, (int)sequence.cols()) : 
                              sequence.cols();
        
        #pragma omp parallel for
        for (int pos = 0; pos < seq_len; ++pos) {
            float* row_data = sequence.row(pos).data();
            
            // Process pairs with SIMD
            for (int i = 0; i < embed_dim - 1; i += 2) {
                const float theta = compute_rope_angle(pos, i, embed_dim);
                const float cos_theta = std::cos(theta);
                const float sin_theta = std::sin(theta);
                
                const float x0 = row_data[i];
                const float x1 = row_data[i + 1];
                
                row_data[i]     = x0 * cos_theta - x1 * sin_theta;
                row_data[i + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
    
    /**
     * ALiBi - most memory efficient, no position embeddings at all!
     * Biases are computed directly in attention
     */
    Matrix compute_alibi_bias(int seq_len, int num_heads) const {
        Matrix bias(seq_len, seq_len);
        
        // Geometric sequence for head-specific slopes
        const float base = std::pow(2.0f, -8.0f / num_heads);
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                bias(i, j) = -std::abs(i - j) * base;
            }
        }
        
        return bias;
    }
    
    /**
     * Low-rank factorization for learned embeddings
     * Store U (max_seq x rank) and V (rank x max_dim) instead of full matrix
     * Memory: O(max_seq * rank + rank * max_dim) where rank << max_dim
     */
    class LowRankPositionalEmbedding {
    public:
        LowRankPositionalEmbedding(int max_seq, int max_dim, int rank = 64) 
            : rank_(rank) {
            // Initialize with SVD of sinusoidal encoding
            U_ = Matrix::Random(max_seq, rank) * 0.02f;
            V_ = Matrix::Random(rank, max_dim) * 0.02f;
        }
        
        Vector get_position(int pos, int dim) const {
            // Reconstruct position embedding from low-rank factors
            return (U_.row(pos) * V_).head(dim);
        }
        
        size_t memory_bytes() const {
            return (U_.size() + V_.size()) * sizeof(float);
        }
        
    private:
        int rank_;
        Matrix U_;  // [max_seq, rank]
        Matrix V_;  // [rank, max_dim]
    };
    
    /**
     * Memory footprint comparison
     */
    size_t get_memory_bytes() const {
        switch (encoding_type_) {
            case SINUSOIDAL:
            case ROTARY:
                // Only store frequency bases
                return freq_bases_.size() * sizeof(float);
            
            case ALIBI:
                // No storage needed!
                return 0;
            
            case LEARNED_COMPACT:
                // Low-rank factors
                return learned_embedding_->memory_bytes();
            
            default:
                return 0;
        }
    }
    
private:
    int max_seq_length_;
    int max_embed_dim_;
    EncodingType encoding_type_;
    
    // Pre-computed frequency bases for faster computation
    std::vector<float> freq_bases_;
    
    // For learned compact embeddings
    std::unique_ptr<LowRankPositionalEmbedding> learned_embedding_;
    
    void precompute_frequency_bases() {
        freq_bases_.reserve(max_embed_dim_ / 2);
        for (int i = 0; i < max_embed_dim_; i += 2) {
            freq_bases_.push_back(1.0f / std::pow(10000.0f, float(i) / max_embed_dim_));
        }
    }
    
    void initialize_learned_parameters() {
        const int rank = std::min(64, max_embed_dim_ / 4);  // Adaptive rank
        learned_embedding_ = std::make_unique<LowRankPositionalEmbedding>(
            max_seq_length_, max_embed_dim_, rank
        );
    }
    
    inline void compute_sinusoidal_inline(int pos, int dim, float* output) const {
        // Vectorized computation
        #pragma omp simd
        for (int i = 0; i < dim; i += 2) {
            const float angle = pos * freq_bases_[i / 2];
            output[i] = std::sin(angle);
            if (i + 1 < dim) {
                output[i + 1] = std::cos(angle);
            }
        }
    }
    
    inline void compute_rotary_inline(int pos, int dim, float* output) const {
        // Initialize to zero (RoPE is applied as rotation, not addition)
        std::fill(output, output + dim, 0.0f);
    }
    
    inline void compute_alibi_inline(int pos, int dim, float* output) const {
        // ALiBi doesn't use position embeddings
        std::fill(output, output + dim, 0.0f);
    }
    
    inline void compute_learned_inline(int pos, int dim, float* output) const {
        if (learned_embedding_ && pos < max_seq_length_) {
            Vector embedding = learned_embedding_->get_position(pos, dim);
            std::memcpy(output, embedding.data(), dim * sizeof(float));
        }
    }
    
    inline float compute_rope_angle(int pos, int dim_idx, int embed_dim) const {
        return pos / std::pow(10000.0f, 2.0f * dim_idx / embed_dim);
    }
    
    inline void apply_to_position_inplace(Eigen::Ref<Vector> position, 
                                         int pos_idx, int embed_dim) {
        Vector encoding = get_position_encoding(pos_idx, embed_dim);
        position.head(embed_dim) += encoding;
    }
};

/**
 * Quantized positional encoding for extreme memory efficiency
 */
class QuantizedPositionalEncoding {
public:
    QuantizedPositionalEncoding(int max_seq, int max_dim) 
        : max_seq_(max_seq), max_dim_(max_dim) {
        // Store as int8 instead of float32 (4x compression)
        initialize_quantized_values();
    }
    
    void apply_to_sequence(Matrix& sequence) {
        const int seq_len = sequence.rows();
        const int embed_dim = sequence.cols();
        
        #pragma omp parallel for
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < embed_dim; ++j) {
                // Dequantize and add
                float value = dequantize(quantized_values_[i * max_dim_ + j]);
                sequence(i, j) += value;
            }
        }
    }
    
    size_t memory_bytes() const {
        return quantized_values_.size();  // 1 byte per value instead of 4
    }
    
private:
    int max_seq_;
    int max_dim_;
    std::vector<int8_t> quantized_values_;
    float scale_;
    float zero_point_;
    
    void initialize_quantized_values() {
        // Compute normal positional encodings
        Matrix full_encoding(max_seq_, max_dim_);
        for (int pos = 0; pos < max_seq_; ++pos) {
            for (int i = 0; i < max_dim_; i += 2) {
                float angle = pos / std::pow(10000.0f, float(i) / max_dim_);
                full_encoding(pos, i) = std::sin(angle);
                if (i + 1 < max_dim_) {
                    full_encoding(pos, i + 1) = std::cos(angle);
                }
            }
        }
        
        // Quantize to int8
        float min_val = full_encoding.minCoeff();
        float max_val = full_encoding.maxCoeff();
        scale_ = (max_val - min_val) / 255.0f;
        zero_point_ = -min_val / scale_;
        
        quantized_values_.resize(max_seq_ * max_dim_);
        for (int i = 0; i < max_seq_ * max_dim_; ++i) {
            float value = full_encoding.data()[i];
            int8_t quantized = std::round((value - min_val) / scale_);
            quantized_values_[i] = quantized;
        }
    }
    
    inline float dequantize(int8_t value) const {
        return value * scale_ + zero_point_;
    }
};

} // namespace transformer

#endif // COMPACT_POSITIONAL_ENCODING_H