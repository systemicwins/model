#ifndef MATRYOSHKA_ENCODER_H
#define MATRYOSHKA_ENCODER_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Dense>
#include "mamba.h"
#include "compact_positional_encoding_f32.h"

namespace matryoshka {

// Bring core tensor types from mamba namespace into scope
using mamba::Matrix;
using mamba::Vector;
using transformer::CompactPositionalEncodingF32;


// Supported embedding dimensions (powers of 2 for efficiency)
const std::vector<int> MATRYOSHKA_DIMS = {64, 128, 256, 512, 768, 1024, 1536};

struct MatryoshkaConfig {
    // Base mamba config
    mamba::MambaConfig mamba_config;

    // Matryoshka-specific settings
    std::vector<int> embedding_dims = MATRYOSHKA_DIMS;
    bool use_learned_pooling = true;
    bool use_dimension_specific_heads = true;
    float temperature = 0.07f;  // For contrastive loss
    float lambda_reg = 0.01f;   // Regularization for dimension consistency

    // Training strategy
    bool progressive_training = true;
    int warmup_steps_per_dim = 1000;
    float base_learning_rate = 5e-5f;
    float dim_specific_lr_scale = 0.8f;  // Scale LR for larger dims
};

class DimensionSpecificHead {
public:
    DimensionSpecificHead(int input_dim, int output_dim);
    Vector forward(const Vector& input);
    Matrix forward_batch(const Matrix& input);
    
private:
    Matrix W_;
    Vector b_;
    Matrix layer_norm_gamma_;
    Matrix layer_norm_beta_;
    float eps_ = 1e-6f;
};

class MatryoshkaLoss {
public:
    MatryoshkaLoss(const MatryoshkaConfig& config);
    
    // Compute multi-scale contrastive loss
    float compute_loss(const std::vector<Matrix>& embeddings_at_dims,
                      const Matrix& positive_pairs,
                      const Matrix& negative_pairs);
    
    // Dimension consistency loss (ensures larger dims contain smaller dim info)
    float dimension_consistency_loss(const std::vector<Matrix>& embeddings_at_dims);
    
    // Information preservation loss
    float information_preservation_loss(const Matrix& full_embedding,
                                      const std::vector<Matrix>& truncated_embeddings);

private:
    MatryoshkaConfig config_;
    float cosine_similarity(const Vector& a, const Vector& b);
    Matrix compute_similarity_matrix(const Matrix& embeddings);
};

class MatryoshkaEncoder {
public:
    explicit MatryoshkaEncoder(const MatryoshkaConfig& config = MatryoshkaConfig());
    ~MatryoshkaEncoder();
    
    // Get embedding at specific dimension for single embedding
    Vector encode(const std::vector<float>& input_embedding, int target_dim);
    
    // Encode sequence with positional encoding
    Matrix encode_sequence(const std::vector<std::vector<float>>& sequence, int target_dim);
    
    // Get embeddings at all supported dimensions
    std::unordered_map<int, Vector> encode_all_dims(const std::vector<float>& input_embedding);
    
    // Batch encoding
    Matrix encode_batch(const std::vector<std::vector<float>>& embeddings, int target_dim);
    
    // Training methods
    void train_step(const std::vector<std::vector<float>>& anchor_embeddings,
                   const std::vector<std::vector<float>>& positive_embeddings,
                   const std::vector<std::vector<float>>& negative_embeddings,
                   float learning_rate);
    
    // Fine-tune on specific dimension
    void fine_tune_dimension(int target_dim,
                            const std::vector<std::vector<float>>& embeddings,
                            const std::vector<int>& labels,
                            int epochs = 10);
    
    // Compression ratio analysis
    float get_compression_ratio(int target_dim) const;
    
    // Performance metrics
    struct PerformanceMetrics {
        float cosine_similarity;
        float euclidean_distance;
        float information_retention;  // Percentage of information retained
        float inference_time_ms;
        float memory_usage_mb;
    };
    
    PerformanceMetrics evaluate_dimension(int target_dim,
                                         const std::vector<std::vector<float>>& test_embeddings,
                                         const std::vector<std::vector<float>>& reference_embeddings);
    
    // Save/Load model
    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath);
    
    // Adaptive dimension selection based on task requirements
    int select_optimal_dimension(float target_accuracy, 
                                float max_latency_ms,
                                float max_memory_mb);

private:
    // Allow trainer to access internal training state
    friend class MatryoshkaTrainer;
    MatryoshkaConfig config_;
    std::unique_ptr<mamba::MambaModel> base_model_;
    std::unordered_map<int, std::unique_ptr<DimensionSpecificHead>> dimension_heads_;
    std::unique_ptr<MatryoshkaLoss> loss_function_;
    
    // Learned pooling weights for each dimension
    std::unordered_map<int, Matrix> pooling_weights_;
    
    // Positional encoding caches
    Matrix base_positional_encoding_;
    std::unordered_map<int, Matrix> dimension_positional_encodings_;
    
    // Dimension-specific normalization parameters
    std::unordered_map<int, std::pair<Vector, Vector>> norm_params_;  // mean, std
    
    // Compact positional encoding with full float32 precision
    std::unique_ptr<CompactPositionalEncodingF32> positional_encoder_;
    bool use_rotary_embeddings_ = false;  // Option for RoPE
    
    // Training state
    int current_training_step_;
    int current_dimension_focus_;
    std::vector<float> dimension_loss_history_;
    
    // Initialize dimension-specific components
    void initialize_dimension_heads();
    void initialize_pooling_weights();
    void initialize_positional_encodings();
    
    // Core encoding logic
    Vector encode_at_dimension(const Matrix& full_embedding, int target_dim);
    Matrix encode_sequence_at_dimension(const Matrix& sequence_embedding, int target_dim);
    
    // Dimension-aware pooling
    Vector apply_learned_pooling(const Matrix& features, int target_dim);
    
    // Progressive dimension reduction
    Matrix progressive_reduction(const Matrix& input, int from_dim, int to_dim);
    
    // Positional encoding methods
    Matrix apply_positional_encoding(const Matrix& embeddings, int target_dim);
    Matrix apply_rotary_positional_encoding(const Matrix& embeddings, int target_dim);
    Matrix compute_sinusoidal_encoding(int seq_len, int embed_dim);
    
    // Optimize for specific hardware
    void optimize_for_hardware(const std::string& hardware_profile);
};

// Utility class for model evaluation and benchmarking
class MatryoshkaBenchmark {
public:
    struct BenchmarkResult {
        int dimension;
        float quality_score;
        float baseline_score;
        float relative_performance;
        float speed_improvement;
        float memory_savings;
        std::string recommendation;
    };
    
    static BenchmarkResult evaluate(MatryoshkaEncoder& encoder,
                                   const std::vector<std::vector<float>>& test_embeddings,
                                   const std::string& task_type = "similarity");
    
    static void generate_report(const std::vector<BenchmarkResult>& results,
                               const std::string& output_file);
    
    // Benchmark all dimensions
    static std::vector<BenchmarkResult> benchmark_all_dimensions(
        MatryoshkaEncoder& encoder,
        const std::vector<std::vector<float>>& test_embeddings);
};

// Training utilities
class MatryoshkaTrainer {
public:
    MatryoshkaTrainer(MatryoshkaEncoder& encoder, const MatryoshkaConfig& config);
    
    void train(const std::vector<std::vector<float>>& training_embeddings,
              const std::vector<int>& labels,
              int epochs = 100);
    
    // Curriculum learning: start with smaller dims, gradually increase
    void curriculum_train(const std::vector<std::vector<float>>& embeddings,
                         const std::vector<int>& labels);
    
    // Distillation from transformer embeddings
    void distill_from_mamba(const std::vector<std::vector<float>>& mamba_embeddings,
                                 const std::vector<std::string>& texts);

private:
    MatryoshkaEncoder& encoder_;
    MatryoshkaConfig config_;
    
    void update_learning_rate(int epoch, int current_dim);
    std::vector<std::pair<int, int>> generate_positive_pairs(const std::vector<int>& labels);
    std::vector<std::pair<int, int>> generate_negative_pairs(const std::vector<int>& labels);
};

} // namespace matryoshka

#endif // MATRYOSHKA_ENCODER_H
