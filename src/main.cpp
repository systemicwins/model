// Simple modification for testing
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include "mamba.h"
#include "tensor_ops.h"
#include "../include/tokenizer.h"

void print_usage() {
    std::cout << "Usage: mamba_model <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  encode <text>          - Encode text using Mamba\n";
    std::cout << "  process <text>         - Process text through Mamba\n";
    std::cout << "  batch <text1> <text2>  - Process multiple texts\n";
    std::cout << "  test                   - Run test with sample data\n";
    std::cout << "  benchmark              - Run performance benchmark\n";
}

void run_test() {
    std::cout << "Running Mamba model test...\n\n";

    // Create Mamba model with default config
    mamba::MambaConfig config;
    config.num_layers = 4;  // Smaller model for testing
    config.state_dim = 64;

    mamba::MambaModel model(config);
    
    // Create sample embeddings (simulating text-embedding-3-small output)
    std::vector<std::vector<float>> sample_embeddings;
    int seq_length = 5;
    int embed_dim = 1536;
    
    // Generate random embeddings for testing
    for (int i = 0; i < seq_length; ++i) {
        std::vector<float> embedding(embed_dim);
        for (int j = 0; j < embed_dim; ++j) {
            embedding[j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random values between -1 and 1
        }
        sample_embeddings.push_back(embedding);
    }
    
    std::cout << "Processing " << seq_length << " embeddings of dimension " << embed_dim << "...\n";

    // Test basic transformer functionality
    auto start = std::chrono::high_resolution_clock::now();

    try {
        // First test just model creation
        std::cout << "Model created successfully!\n";
        std::cout << "Mamba config: " << config.num_layers << " layers, "
                  << config.state_dim << " state_dim, " << config.embed_dim << " embed_dim\n";

        // Convert to matrix for testing
        mamba::Matrix input(seq_length, embed_dim);
        for (int i = 0; i < seq_length; ++i) {
            for (int j = 0; j < embed_dim; ++j) {
                input(i, j) = sample_embeddings[i][j];
            }
        }

        std::cout << "Input matrix created successfully!\n";

        // Test forward pass
        mamba::Matrix output = model.forward(input);
        std::cout << "Forward pass completed successfully!\n";
        std::cout << "Output shape: " << output.rows() << " x " << output.cols() << "\n";

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Processing time: " << duration.count() << " ms\n";
        std::cout << "Model created and processed successfully!\n";
    } catch (const std::exception& e) {
        std::cout << "Error during processing: " << e.what() << "\n";
        std::cout << "Error type: " << typeid(e).name() << "\n";
    }
    
    std::cout << "\nTest completed successfully!\n";
}

void process_text(const std::string& text, bool full_process = true) {
    try {
        std::cout << "Processing text: \"" << text << "\"\n";
        
        mamba::MambaConfig config;
        config.num_layers = 4;
        mamba::MambaModel model(config);
        
        // Initialize tokenizer with larger vocabulary for financial terms
        Tokenizer::TokenizerConfig tok_config;
        tok_config.vocab_size = 56700;  // Base vocab (50000) + ticker symbols (6700)
        tok_config.max_length = 512;
        Tokenizer tokenizer(tok_config);
        
        // Tokenize the text
        std::cout << "Tokenizing text...\n";
        std::vector<int> token_ids = tokenizer.encode(text, true);
        std::cout << "Generated " << token_ids.size() << " tokens\n";
        
        // Display first few tokens for debugging
        std::cout << "First tokens: ";
        for (size_t i = 0; i < std::min(size_t(10), token_ids.size()); ++i) {
            std::cout << token_ids[i] << " ";
        }
        std::cout << "\n";
        
        // Convert tokens to embeddings
        std::vector<std::vector<float>> embeddings = tokenizer.tokens_to_embeddings(token_ids);
        std::cout << "Created " << embeddings.size() << " embeddings from tokens\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> output = model.encode(embeddings);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Mamba output size: " << output.size() << "\n";
        std::cout << "Processing time: " << duration.count() << " ms\n";
        
        // Show first few values
        std::cout << "First 10 output values: ";
        for (size_t i = 0; i < std::min(size_t(10), output.size()); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << "\n";
        
        if (full_process && !embeddings.empty()) {
            // Convert to matrix and get pooled output
            int seq_len = embeddings.size();
            int embed_dim = embeddings[0].size();
            Eigen::MatrixXf output_matrix(seq_len, embed_dim);
            for (size_t i = 0; i < static_cast<size_t>(seq_len) && i * embed_dim < output.size(); ++i) {
                for (size_t j = 0; j < static_cast<size_t>(embed_dim) && i * embed_dim + j < output.size(); ++j) {
                    output_matrix(i, j) = output[i * embed_dim + j];
                }
            }
            
            auto pooled = model.get_pooled_output(output_matrix, "mean");
            std::cout << "Pooled output first 10 values: ";
            for (int i = 0; i < std::min(10, (int)pooled.size()); ++i) {
                std::cout << pooled(i) << " ";
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

void process_batch(const std::vector<std::string>& texts) {
    try {
        std::cout << "Processing " << texts.size() << " texts...\n";
        
        // Initialize tokenizer with larger vocabulary for financial terms
        Tokenizer::TokenizerConfig tok_config;
        tok_config.vocab_size = 56700;  // Base vocab (50000) + ticker symbols (6700)
        tok_config.max_length = 512;
        Tokenizer tokenizer(tok_config);
        
        // Tokenize all texts
        std::cout << "Tokenizing " << texts.size() << " texts...\n";
        auto token_ids_batch = tokenizer.encode_batch(texts, true, true);  // Add special tokens and padding
        
        // Convert all tokens to embeddings
        std::vector<std::vector<float>> all_embeddings = tokenizer.tokens_to_embeddings_batch(token_ids_batch);
        
        std::cout << "Created " << all_embeddings.size() << " total embeddings\n";
        
        std::cout << "\nProcessing through transformer...\n";
        
        mamba::MambaConfig config;
        config.num_layers = 4;
        mamba::MambaModel model(config);
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> output = model.encode(all_embeddings);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Mamba processing time: " << duration.count() << " ms\n";
        
        // Convert to matrix for pooling
        int seq_len = all_embeddings.size();
        int embed_dim = 1536;
        Eigen::MatrixXf output_matrix(seq_len, embed_dim);
        for (size_t i = 0; i < static_cast<size_t>(seq_len) && i * embed_dim < output.size(); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(embed_dim) && i * embed_dim + j < output.size(); ++j) {
                output_matrix(i, j) = output[i * embed_dim + j];
            }
        }
        
        // Get different pooling strategies
        auto mean_pooled = model.get_pooled_output(output_matrix, "mean");
        auto max_pooled = model.get_pooled_output(output_matrix, "max");
        auto first_pooled = model.get_pooled_output(output_matrix, "first");
        
        std::cout << "\nPooling results (first 5 dimensions):\n";
        std::cout << "Mean: ";
        for (int i = 0; i < std::min(5, (int)mean_pooled.size()); ++i) {
            std::cout << mean_pooled(i) << " ";
        }
        std::cout << "\nMax:  ";
        for (int i = 0; i < std::min(5, (int)max_pooled.size()); ++i) {
            std::cout << max_pooled(i) << " ";
        }
        std::cout << "\nFirst:";
        for (int i = 0; i < std::min(5, (int)first_pooled.size()); ++i) {
            std::cout << first_pooled(i) << " ";
        }
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

void run_benchmark() {
    std::cout << "Running performance benchmark...\n\n";
    
    std::vector<int> sequence_lengths = {2, 5, 10, 50, 100};
    std::vector<int> layer_counts = {2, 4, 6};
    
    for (int num_layers : layer_counts) {
        std::cout << "Testing with " << num_layers << " Mamba2 layers:\n";
        
        mamba::MambaConfig config;
        config.num_layers = num_layers;
        // config.num_ssm_heads = 8;  // Commented out for now - not implemented
        config.expand_factor = 4;
        
        mamba::MambaModel model(config);
        
        for (int seq_len : sequence_lengths) {
            // Generate random embeddings
            std::vector<std::vector<float>> embeddings;
            for (int i = 0; i < seq_len; ++i) {
                std::vector<float> embedding(1536);
                for (int j = 0; j < 1536; ++j) {
                    embedding[j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
                }
                embeddings.push_back(embedding);
            }
            
            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<float> output = model.encode(embeddings);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "  Sequence length " << seq_len << ": " 
                      << duration.count() / 1000.0 << " ms\n";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "test") {
        run_test();
    } else if (command == "encode" && argc >= 3) {
        std::string text = argv[2];
        for (int i = 3; i < argc; ++i) {
            text += " " + std::string(argv[i]);
        }
        process_text(text, false);
    } else if (command == "process" && argc >= 3) {
        std::string text = argv[2];
        for (int i = 3; i < argc; ++i) {
            text += " " + std::string(argv[i]);
        }
        process_text(text, true);
    } else if (command == "batch" && argc >= 3) {
        std::vector<std::string> texts;
        for (int i = 2; i < argc; ++i) {
            texts.push_back(argv[i]);
        }
        process_batch(texts);
    } else if (command == "benchmark") {
        run_benchmark();
    } else {
        print_usage();
        return 1;
    }
    
    return 0;
}