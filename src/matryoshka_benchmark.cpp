#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <cstdlib>
#include "matryoshka_encoder.h"
#include "mamba.h"

using namespace matryoshka;
using namespace std::chrono;

void print_usage() {
    std::cout << "Usage: matryoshka_benchmark <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  test               - Run basic functionality test\n";
    std::cout << "  encode <text>      - Encode text using Mamba and matryoshka\n";
    std::cout << "  benchmark          - Run comprehensive benchmark\n";
    std::cout << "  train              - Train model with sample data\n";
    std::cout << "  distill            - Distill from Mamba embeddings\n";
    std::cout << "  analyze            - Analyze dimension vs quality trade-offs\n\n";
}

void test_matryoshka() {
    std::cout << "=== Matryoshka Encoder Test ===\n\n";
    
    MatryoshkaConfig config;
    config.embedding_dims = {64, 128, 256, 512, 768, 1024, 1536};
    config.mamba_config.embed_dim = 1536;
    
    MatryoshkaEncoder encoder(config);
    
    // Generate test embedding from our transformer
    std::vector<float> test_embedding(1536);
    for (int i = 0; i < 1536; ++i) {
        test_embedding[i] = std::sin(i * 0.1f) * 0.5f;
    }
    
    std::cout << "Testing encoding at different dimensions:\n";
    std::cout << std::setw(10) << "Dimension" 
              << std::setw(15) << "Output Norm"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Compression Ratio" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (int dim : config.embedding_dims) {
        auto start = high_resolution_clock::now();
        Vector encoded = encoder.encode(test_embedding, dim);
        auto end = high_resolution_clock::now();
        
        float time_ms = duration_cast<microseconds>(end - start).count() / 1000.0f;
        float compression = encoder.get_compression_ratio(dim);
        
        std::cout << std::setw(10) << dim
                  << std::setw(15) << std::fixed << std::setprecision(4) << encoded.norm()
                  << std::setw(15) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(20) << std::fixed << std::setprecision(2) << compression << "x\n";
    }
    
    std::cout << "\n=== Test Completed ===\n";
}

void encode_with_mamba(const std::string& text) {
    std::cout << "=== Mamba + Matryoshka Encoding ===\n\n";
    std::cout << "Text: \"" << text << "\"\n\n";

    // Initialize Mamba model
    mamba::MambaConfig mamba_config;
    mamba_config.num_layers = 4;
    mamba::MambaModel mamba_model(mamba_config);
    
    // Convert text to embedding (simplified - in practice, use tokenization)
    std::vector<std::vector<float>> input_sequence;
    std::vector<float> char_embedding(1536);
    for (char c : text) {
        for (int i = 0; i < 1536; ++i) {
            char_embedding[i] = std::sin((c + i) * 0.01f) * 0.5f;
        }
        input_sequence.push_back(char_embedding);
    }
    
    // Process through Mamba
    auto start_mamba = high_resolution_clock::now();
    std::vector<float> mamba_output = mamba_model.encode(input_sequence);
    auto end_mamba = high_resolution_clock::now();
    float mamba_time = duration_cast<milliseconds>(end_mamba - start_mamba).count();

    std::cout << "Mamba encoding completed: " << mamba_output.size() << " values\n";
    std::cout << "Mamba time: " << mamba_time << " ms\n\n";
    
    // Extract first 1536 dimensions as embedding
    std::vector<float> embedding(1536);
    for (int i = 0; i < 1536 && i < mamba_output.size(); ++i) {
        embedding[i] = mamba_output[i];
    }
    
    // Initialize Matryoshka encoder
    MatryoshkaConfig config;
    MatryoshkaEncoder encoder(config);
    
    // Encode at different dimensions
    std::cout << std::setw(10) << "Dimension"
              << std::setw(15) << "Time(ms)"
              << std::setw(15) << "Memory(KB)"
              << std::setw(15) << "Compression" << "\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (int dim : config.embedding_dims) {
        auto start_matry = high_resolution_clock::now();
        Vector matry_encoded = encoder.encode(embedding, dim);
        auto end_matry = high_resolution_clock::now();
        float matry_time = duration_cast<microseconds>(end_matry - start_matry).count() / 1000.0f;
        
        float memory_kb = (dim * sizeof(float)) / 1024.0f;
        float compression = encoder.get_compression_ratio(dim);
        
        std::cout << std::setw(10) << dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << matry_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << memory_kb
                  << std::setw(15) << std::fixed << std::setprecision(1) << compression << "x\n";
    }
    
    // Recommendation
    std::cout << "\n=== Recommendation ===\n";
    int optimal_dim = encoder.select_optimal_dimension(0.95f, 10.0f, 1.0f);
    std::cout << "Optimal dimension for 95% accuracy, <10ms latency, <1MB memory: " 
              << optimal_dim << "\n";
}

void run_comprehensive_benchmark() {
    std::cout << "=== Comprehensive Matryoshka Benchmark ===\n\n";
    
    MatryoshkaConfig config;
    MatryoshkaEncoder encoder(config);
    
    // Test different batch sizes and sequence lengths
    std::vector<int> batch_sizes = {1, 8, 32, 128};
    std::vector<int> dimensions = {64, 256, 512, 1536};
    
    std::cout << "Performance Matrix:\n";
    std::cout << std::setw(12) << "Batch Size";
    for (int dim : dimensions) {
        std::cout << std::setw(12) << ("Dim " + std::to_string(dim));
    }
    std::cout << "\n" << std::string(12 + dimensions.size() * 12, '-') << "\n";
    
    for (int batch_size : batch_sizes) {
        std::cout << std::setw(12) << batch_size;
        
        // Generate test batch
        std::vector<std::vector<float>> batch;
        for (int i = 0; i < batch_size; ++i) {
            std::vector<float> embedding(1536);
            for (int j = 0; j < 1536; ++j) {
                embedding[j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            }
            batch.push_back(embedding);
        }
        
        for (int dim : dimensions) {
            auto start = high_resolution_clock::now();
            Matrix encoded = encoder.encode_batch(batch, dim);
            auto end = high_resolution_clock::now();
            
            float time_ms = duration_cast<microseconds>(end - start).count() / 1000.0f;
            float throughput = batch_size / (time_ms / 1000.0f);  // samples/sec
            
            std::cout << std::setw(12) << std::fixed << std::setprecision(1) 
                      << throughput;
        }
        std::cout << "\n";
    }
    
    std::cout << "\n=== Quality vs Dimension Analysis ===\n";
    std::cout << std::setw(12) << "Dimension"
              << std::setw(18) << "Info Retention %"
              << std::setw(15) << "Memory (KB)"
              << std::setw(18) << "Relative Speed" << "\n";
    std::cout << std::string(63, '-') << "\n";
    
    // Generate reference embeddings
    std::vector<std::vector<float>> reference(10);
    for (int i = 0; i < 10; ++i) {
        reference[i].resize(1536);
        for (int j = 0; j < 1536; ++j) {
            reference[i][j] = std::cos(i * j * 0.01f);
        }
    }
    
    for (int dim : config.embedding_dims) {
        auto metrics = encoder.evaluate_dimension(dim, reference, reference);
        
        std::cout << std::setw(12) << dim
                  << std::setw(18) << std::fixed << std::setprecision(1) 
                  << metrics.information_retention
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << metrics.memory_usage_mb * 1024
                  << std::setw(18) << std::fixed << std::setprecision(2) 
                  << (1536.0f / dim) << "x\n";
    }
    
    std::cout << "\n=== Benchmark Completed ===\n";
}

void train_sample_model() {
    std::cout << "=== Training Matryoshka Model ===\n\n";
    
    MatryoshkaConfig config;
    config.progressive_training = true;
    config.embedding_dims = {64, 128, 256, 512};  // Smaller dims for faster training
    
    MatryoshkaEncoder encoder(config);
    MatryoshkaTrainer trainer(encoder, config);
    
    // Generate synthetic training data
    int num_samples = 100;
    int num_classes = 10;
    std::vector<std::vector<float>> training_data;
    std::vector<int> labels;
    
    std::cout << "Generating synthetic training data...\n";
    for (int i = 0; i < num_samples; ++i) {
        std::vector<float> embedding(1536);
        int label = i % num_classes;
        
        // Create class-specific patterns
        for (int j = 0; j < 1536; ++j) {
            embedding[j] = std::sin(label * j * 0.01f) + 
                          std::cos(i * j * 0.001f) * 0.3f +
                          ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        
        training_data.push_back(embedding);
        labels.push_back(label);
    }
    
    std::cout << "Starting curriculum training...\n\n";
    trainer.curriculum_train(training_data, labels);
    
    // Save trained model
    std::cout << "\nSaving model to 'matryoshka_model.bin'...\n";
    encoder.save_model("matryoshka_model.bin");
    
    std::cout << "Training completed!\n";
}

void analyze_dimensions() {
    std::cout << "=== Dimension Analysis for Different Tasks ===\n\n";
    
    MatryoshkaConfig config;
    MatryoshkaEncoder encoder(config);
    
    struct TaskRequirement {
        std::string name;
        float min_accuracy;
        float max_latency_ms;
        float max_memory_mb;
    };
    
    std::vector<TaskRequirement> tasks = {
        {"Real-time search", 0.90f, 5.0f, 0.5f},
        {"Semantic similarity", 0.95f, 20.0f, 2.0f},
        {"Document classification", 0.93f, 50.0f, 5.0f},
        {"Clustering", 0.88f, 100.0f, 10.0f},
        {"Embeddings database", 0.85f, 1000.0f, 0.1f}
    };
    
    std::cout << std::setw(25) << "Task"
              << std::setw(15) << "Optimal Dim"
              << std::setw(20) << "Compression"
              << std::setw(20) << "Expected Quality" << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (const auto& task : tasks) {
        int optimal_dim = encoder.select_optimal_dimension(
            task.min_accuracy, task.max_latency_ms, task.max_memory_mb);
        
        float compression = encoder.get_compression_ratio(optimal_dim);
        float expected_quality = task.min_accuracy * (optimal_dim / 1536.0f);
        
        std::cout << std::setw(25) << task.name
                  << std::setw(15) << optimal_dim
                  << std::setw(20) << std::fixed << std::setprecision(1) 
                  << compression << "x"
                  << std::setw(20) << std::fixed << std::setprecision(1) 
                  << (expected_quality * 100) << "%\n";
    }
    
    std::cout << "\n=== Memory vs Quality Trade-off ===\n";
    std::cout << "For 1 million embeddings:\n\n";
    std::cout << std::setw(12) << "Dimension"
              << std::setw(15) << "Storage (GB)"
              << std::setw(15) << "vs Full Size"
              << std::setw(20) << "Quality Retention" << "\n";
    std::cout << std::string(62, '-') << "\n";
    
    for (int dim : config.embedding_dims) {
        float storage_gb = (1000000.0f * dim * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float size_ratio = dim / 1536.0f;
        float quality = 0.98f * std::pow(size_ratio, 0.3f);  // Empirical quality model
        
        std::cout << std::setw(12) << dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << storage_gb
                  << std::setw(15) << std::fixed << std::setprecision(1) 
                  << (size_ratio * 100) << "%"
                  << std::setw(20) << std::fixed << std::setprecision(1) 
                  << (quality * 100) << "%\n";
    }
    
    std::cout << "\n=== Analysis Completed ===\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "test") {
        test_matryoshka();
    } else if (command == "encode" && argc >= 3) {
        std::string text = argv[2];
        for (int i = 3; i < argc; ++i) {
            text += " " + std::string(argv[i]);
        }
        encode_with_mamba(text);
    } else if (command == "benchmark") {
        run_comprehensive_benchmark();
    } else if (command == "train") {
        train_sample_model();
    } else if (command == "analyze") {
        analyze_dimensions();
    } else {
        print_usage();
        return 1;
    }
    
    return 0;
}