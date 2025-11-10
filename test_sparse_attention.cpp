#include "sparse_attention.h"
#include <iostream>
#include <chrono>
#include <random>

using namespace mamba;
using namespace std;

// Utility function to generate random matrix
Matrix generate_random_matrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    Matrix matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = dis(gen);
        }
    }
    return matrix;
}

// Test basic sparse attention functionality
void test_basic_sparse_attention() {
    cout << "=== Testing Basic Sparse Attention ===" << endl;
    
    // Create configuration
    SparseAttentionConfig config;
    config.enable_sparse_attention = true;
    config.track_state_changes = true;
    config.high_change_threshold = 0.8f;
    config.moderate_change_threshold = 0.5f;
    config.low_change_threshold = 0.2f;
    
    // Initialize sparse attention
    SSMSparseAttention sparse_attention(config);
    
    // Generate test data
    Matrix query = generate_random_matrix(4, 64);
    Matrix key = generate_random_matrix(4, 64);
    Matrix value = generate_random_matrix(4, 64);
    
    cout << "Query shape: " << query.rows() << " x " << query.cols() << endl;
    cout << "Key shape: " << key.rows() << " x " << key.cols() << endl;
    cout << "Value shape: " << value.rows() << " x " << value.cols() << endl;
    
    // Test sparse attention computation
    auto start = chrono::high_resolution_clock::now();
    Matrix result = sparse_attention.compute_sparse_attention(query, key, value);
    auto end = chrono::high_resolution_clock::now();
    
    cout << "Result shape: " << result.rows() << " x " << result.cols() << endl;
    cout << "Computation time: " 
         << chrono::duration_cast<chrono::microseconds>(end - start).count() 
         << " microseconds" << endl;
    
    // Get performance statistics
    auto perf_stats = sparse_attention.get_performance_stats();
    cout << "Performance Statistics:" << endl;
    cout << "  Average attention sparsity: " << perf_stats.avg_attention_sparsity << endl;
    cout << "  Dense attention percentage: " << perf_stats.dense_attention_percentage << endl;
    cout << "  Computation reduction: " << perf_stats.computation_reduction << "%" << endl;
    cout << "  Total computations: " << perf_stats.total_computations << endl;
    cout << "  Sparse computations: " << perf_stats.sparse_computations << endl;
    
    cout << "Basic sparse attention test completed successfully!" << endl;
    cout << endl;
}

// Test state change analysis
void test_state_change_analysis() {
    cout << "=== Testing State Change Analysis ===" << endl;
    
    SparseAttentionConfig config;
    config.track_state_changes = true;
    SSMSparseAttention sparse_attention(config);
    
    // Generate state sequence with varying change rates
    vector<Matrix> ssm_states;
    int sequence_length = 10;
    int state_dim = 32;
    
    // Create states with different change patterns
    for (int t = 0; t < sequence_length; ++t) {
        Matrix state(state_dim, 1);
        
        if (t == 0) {
            // Initial state
            for (int i = 0; i < state_dim; ++i) {
                state(i, 0) = 0.1f;
            }
        } else {
            // Subsequent states with varying change rates
            float change_magnitude = (t % 3 == 0) ? 0.9f : (t % 3 == 1) ? 0.3f : 0.05f;
            
            for (int i = 0; i < state_dim; ++i) {
                state(i, 0) = ssm_states[t-1](i, 0) + 
                             change_magnitude * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
            }
        }
        
        ssm_states.push_back(state);
    }
    
    // Analyze state changes
    for (int t = 1; t < sequence_length; ++t) {
        StateChangeMetrics metrics = sparse_attention.analyze_state_change(ssm_states, t);
        
        cout << "Time " << t << ":" << endl;
        cout << "  Change rate: " << metrics.change_rate << endl;
        cout << "  Change magnitude: " << metrics.change_magnitude << endl;
        cout << "  Volatility: " << metrics.volatility << endl;
        cout << "  Is transition point: " << (metrics.is_transition_point ? "Yes" : "No") << endl;
        cout << "  Attention weight: " << metrics.attention_weight << endl;
        
        AttentionPattern pattern = sparse_attention.select_attention_pattern(metrics);
        cout << "  Selected pattern: ";
        switch (pattern) {
            case AttentionPattern::DENSE: cout << "DENSE"; break;
            case AttentionPattern::LOCAL_WINDOW: cout << "LOCAL_WINDOW"; break;
            case AttentionPattern::STRIDED: cout << "STRIDED"; break;
            case AttentionPattern::GLOBAL_SPARSE: cout << "GLOBAL_SPARSE"; break;
        }
        cout << endl << endl;
    }
    
    cout << "State change analysis test completed!" << endl;
    cout << endl;
}

// Test multi-head processing
void test_multihead_processing() {
    cout << "=== Testing Multi-Head Processing ===" << endl;
    
    SparseAttentionConfig config;
    SSMSparseAttention sparse_attention(config);
    
    // Create input for multiple heads
    vector<Matrix> inputs = {
        generate_random_matrix(4, 64),
        generate_random_matrix(4, 64),
        generate_random_matrix(4, 64),
        generate_random_matrix(4, 64)
    };
    
    // Define head types
    vector<SSMHeadType> head_types = {
        SSMHeadType::FINANCIAL_PRICE,
        SSMHeadType::FINANCIAL_VOLUME,
        SSMHeadType::FINANCIAL_SENTIMENT,
        SSMHeadType::GENERAL
    };
    
    cout << "Processing " << inputs.size() << " heads with different types:" << endl;
    for (size_t i = 0; i < head_types.size(); ++i) {
        cout << "  Head " << i << ": ";
        switch (head_types[i]) {
            case SSMHeadType::FINANCIAL_PRICE: cout << "FINANCIAL_PRICE"; break;
            case SSMHeadType::FINANCIAL_VOLUME: cout << "FINANCIAL_VOLUME"; break;
            case SSMHeadType::FINANCIAL_SENTIMENT: cout << "FINANCIAL_SENTIMENT"; break;
            default: cout << "GENERAL"; break;
        }
        cout << endl;
    }
    
    // Process multi-head
    auto outputs = sparse_attention.forward_multihead(inputs, head_types);
    
    cout << "Multi-head processing completed successfully!" << endl;
    cout << "Output shapes: ";
    for (const auto& output : outputs) {
        cout << output.rows() << "x" << output.cols() << " ";
    }
    cout << endl;
    cout << endl;
}

// Test domain-specific configurations
void test_domain_specific_configurations() {
    cout << "=== Testing Domain-Specific Configurations ===" << endl;
    
    // Test financial configuration
    SparseAttentionConfig financial_config;
    financial_config.is_financial_data = true;
    SSMSparseAttention financial_attention(financial_config);
    cout << "Financial attention initialized" << endl;
    
    cout << "Domain-specific configuration test completed!" << endl;
    cout << endl;
}

// Test threshold adaptation
void test_threshold_adaptation() {
    cout << "=== Testing Threshold Adaptation ===" << endl;
    
    SparseAttentionConfig config;
    config.enable_adaptive_thresholds = true;
    config.adaptation_rate = 0.1f;
    config.adaptation_window = 3;
    
    SSMSparseAttention adaptive_attention(config);
    
    cout << "Initial thresholds:" << endl;
    cout << "  High: " << config.high_change_threshold << endl;
    cout << "  Moderate: " << config.moderate_change_threshold << endl;
    cout << "  Low: " << config.low_change_threshold << endl;
    
    // Simulate state changes to trigger adaptation
    vector<Matrix> states;
    for (int i = 0; i < 5; ++i) {
        states.push_back(generate_random_matrix(32, 1));
        if (i > 0) {
            StateChangeMetrics metrics = adaptive_attention.analyze_state_change(states, i);
            adaptive_attention.adapt_thresholds(metrics);
        }
    }
    
    cout << "After adaptation (thresholds may have changed):" << endl;
    cout << "  High: " << adaptive_attention.get_config().high_change_threshold << endl;
    cout << "  Moderate: " << adaptive_attention.get_config().moderate_change_threshold << endl;
    cout << "  Low: " << adaptive_attention.get_config().low_change_threshold << endl;
    
    cout << "Threshold adaptation test completed!" << endl;
    cout << endl;
}

// Main test function
int main() {
    cout << "SSM-Guided Sparse Attention Test Suite" << endl;
    cout << "=======================================" << endl;
    cout << endl;
    
    try {
        test_basic_sparse_attention();
        test_state_change_analysis();
        test_multihead_processing();
        test_domain_specific_configurations();
        test_threshold_adaptation();
        
        cout << "All tests completed successfully!" << endl;
        
    } catch (const exception& e) {
        cerr << "Test failed with exception: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}