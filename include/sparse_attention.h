#ifndef SPARSE_ATTENTION_H
#define SPARSE_ATTENTION_H

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <chrono>
#include "tensor_ops.h"
#include "mamba.h"

namespace mamba {

// State Space Model (SSM) head types for specialization
enum class SSMHeadType {
    GENERAL,          // General purpose processing
    FINANCIAL_PRICE,  // Financial price volatility
    FINANCIAL_VOLUME, // Volume change patterns
    FINANCIAL_SENTIMENT, // Market sentiment shifts
    MEDICAL_SIGNAL,   // Medical monitoring data
    MEDICAL_ANOMALY,  // Anomaly detection
    IOT_SENSOR,       // IoT sensor data
    IOT_EVENT         // Event-driven sensor data
};

// State change metrics
struct StateChangeMetrics {
    float change_rate;           // Rate of state change (L2 norm of delta)
    float change_magnitude;      // Magnitude of state change
    float volatility;           // Volatility of state changes over time
    bool is_transition_point;   // Whether this is a significant transition
    float attention_weight;     // Computed attention weight (0.0 to 1.0)
    std::vector<float> head_change_rates; // Per-head change rates
};

// Sparse attention patterns
enum class AttentionPattern {
    DENSE,           // Full attention computation
    LOCAL_WINDOW,    // Local window attention
    STRIDED,         // Strided attention
    GLOBAL_SPARSE    // Global sparse attention
};

// Configuration for sparse attention
struct SparseAttentionConfig {
    float high_change_threshold = 0.8f;    // Dense attention threshold
    float moderate_change_threshold = 0.5f; // Local window threshold  
    float low_change_threshold = 0.2f;     // Strided attention threshold
    float very_low_threshold = 0.05f;      // Global sparse threshold
    
    int local_window_size = 8;    // Local window size
    int stride_size = 4;          // Stride size for strided attention
    int global_sparse_factor = 8; // 1/N positions for global sparse
    
    bool track_state_changes = true;
    bool enable_sparse_attention = true;
    
    std::vector<SSMHeadType> head_types;
    
    // Domain-specific configurations
    bool is_financial_data = false;
    bool is_medical_data = false;
    bool is_iot_data = false;
    
    // Adaptation settings
    bool enable_adaptive_thresholds = true;
    float adaptation_rate = 0.1f; // How quickly thresholds adapt
    int adaptation_window = 50;   // Window size for threshold adaptation
};

// SSM-guided sparse attention class
class SSMSparseAttention {
private:
    SparseAttentionConfig config;
    std::vector<StateChangeMetrics> state_history;
    std::vector<Matrix> threshold_history;
    int sequence_position = 0;
    
    // Domain-specific parameters
    std::vector<float> financial_volatility_thresholds;
    std::vector<float> medical_anomaly_thresholds;
    std::vector<float> iot_event_thresholds;
    
public:
    SSMSparseAttention(const SparseAttentionConfig& config = SparseAttentionConfig());
    ~SSMSparseAttention() = default;
    
    // Main sparse attention computation
    Matrix compute_sparse_attention(
        const Matrix& query, 
        const Matrix& key, 
        const Matrix& value,
        const std::vector<SSMHeadType>& head_types = {}
    );
    
    // State change detection and analysis
    StateChangeMetrics analyze_state_change(
        const std::vector<Matrix>& ssm_states,
        int current_position
    );
    
    // Dynamic pattern selection based on state change
    AttentionPattern select_attention_pattern(const StateChangeMetrics& metrics);
    
    // Multi-head specialized processing
    std::vector<Matrix> forward_multihead(
        const std::vector<Matrix>& inputs,
        const std::vector<SSMHeadType>& head_types
    );
    
    // Threshold adaptation based on recent state changes
    void adapt_thresholds(const StateChangeMetrics& current_metrics);
    
    // Get current state of the sparse attention system
    SparseAttentionConfig get_config() const { return config; }
    void set_config(const SparseAttentionConfig& new_config) { 
        config = new_config; 
        if (config.is_financial_data) initialize_financial_parameters();
        if (config.is_medical_data) initialize_medical_parameters();
        if (config.is_iot_data) initialize_iot_parameters();
    }
    
    // Reset internal state (for new sequences)
    void reset();
    
    // Get performance statistics
    struct PerformanceStats {
        float avg_attention_sparsity;
        float dense_attention_percentage;
        float computation_reduction;
        int total_computations;
        int sparse_computations;
    };
    
    PerformanceStats get_performance_stats() const;
    
private:
    // Pattern-specific attention computations
    Matrix compute_dense_attention(const Matrix& query, const Matrix& key, const Matrix& value);
    Matrix compute_local_window_attention(const Matrix& query, const Matrix& key, const Matrix& value);
    Matrix compute_strided_attention(const Matrix& query, const Matrix& key, const Matrix& value);
    Matrix compute_global_sparse_attention(const Matrix& query, const Matrix& key, const Matrix& value);
    
    // Domain-specific initialization
    void initialize_financial_parameters();
    void initialize_medical_parameters();
    void initialize_iot_parameters();
    
    // State history management
    void update_state_history(const StateChangeMetrics& metrics);
    float compute_volatility(const std::vector<float>& recent_changes);
    
    // Utility functions
    float compute_l2_norm(const Matrix& matrix);
    std::vector<float> compute_per_head_change_rates(const std::vector<Matrix>& states);
    
    // Performance tracking
    mutable PerformanceStats perf_stats;
    void update_performance_stats(AttentionPattern pattern);
};

} // namespace mamba

#endif // SPARSE_ATTENTION_H