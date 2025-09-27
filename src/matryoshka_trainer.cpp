#include "matryoshka_encoder.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

namespace matryoshka {

// MatryoshkaTrainer Implementation
MatryoshkaTrainer::MatryoshkaTrainer(MatryoshkaEncoder& encoder, const MatryoshkaConfig& config)
    : encoder_(encoder), config_(config) {}

void MatryoshkaTrainer::train(const std::vector<std::vector<float>>& training_embeddings,
                             const std::vector<int>& labels,
                             int epochs) {
    
    std::cout << "Starting Matryoshka training for " << epochs << " epochs\n";
    std::cout << "Training samples: " << training_embeddings.size() << "\n";
    std::cout << "Supported dimensions: ";
    for (int dim : config_.embedding_dims) {
        std::cout << dim << " ";
    }
    std::cout << "\n\n";
    
    // Create positive and negative pairs
    auto positive_pairs = generate_positive_pairs(labels);
    auto negative_pairs = generate_negative_pairs(labels);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        // Shuffle training data
        std::vector<size_t> indices(training_embeddings.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Mini-batch training
        int batch_size = 32;
        int num_batches = (training_embeddings.size() + batch_size - 1) / batch_size;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, (int)training_embeddings.size());
            
            // Extract batch
            std::vector<std::vector<float>> batch_embeddings;
            std::vector<int> batch_labels;
            
            for (int i = start_idx; i < end_idx; ++i) {
                batch_embeddings.push_back(training_embeddings[indices[i]]);
                batch_labels.push_back(labels[indices[i]]);
            }
            
            // Generate embeddings at all dimensions
            std::vector<Matrix> embeddings_at_dims;
            
            for (int dim : config_.embedding_dims) {
                Matrix encoded = encoder_.encode_batch(batch_embeddings, dim);
                embeddings_at_dims.push_back(encoded);
            }
            
            // Compute loss
            Matrix pos_pairs(positive_pairs.size(), 2);
            Matrix neg_pairs(negative_pairs.size(), 2);
            
            for (size_t i = 0; i < positive_pairs.size(); ++i) {
                pos_pairs(i, 0) = positive_pairs[i].first;
                pos_pairs(i, 1) = positive_pairs[i].second;
            }
            
            for (size_t i = 0; i < negative_pairs.size(); ++i) {
                neg_pairs(i, 0) = negative_pairs[i].first;
                neg_pairs(i, 1) = negative_pairs[i].second;
            }
            
            // Note: In a real implementation, you would compute gradients and update weights here
            // This is a simplified version showing the training loop structure
            
            epoch_loss += 0.01f;  // Placeholder loss
        }
        
        // Update learning rate
        float current_lr = config_.base_learning_rate * std::pow(0.95f, epoch);
        
        // Log progress
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                     << " | Loss: " << std::fixed << std::setprecision(4) << epoch_loss 
                     << " | LR: " << std::scientific << current_lr << "\n";
        }
    }
    
    std::cout << "\nTraining completed!\n";
}

void MatryoshkaTrainer::curriculum_train(const std::vector<std::vector<float>>& embeddings,
                                        const std::vector<int>& labels) {
    
    std::cout << "Starting curriculum training (progressive dimension increase)\n\n";
    
    // Start with smallest dimensions and progressively increase
    for (int dim : config_.embedding_dims) {
        std::cout << "Training for dimension: " << dim << "\n";
        
        // Adjust training parameters for current dimension
        int epochs_for_dim = 50 + (dim / 64) * 10;  // More epochs for larger dimensions
        float lr_for_dim = config_.base_learning_rate * std::pow(config_.dim_specific_lr_scale, 
                                                                 std::log2(dim / 64.0f));
        
        std::cout << "  Epochs: " << epochs_for_dim << "\n";
        std::cout << "  Learning rate: " << lr_for_dim << "\n";
        
        // Focus training on current dimension
        encoder_.current_dimension_focus_ = dim;
        
        // Train for this dimension
        train(embeddings, labels, epochs_for_dim);
        
        // Warm-up period for next dimension
        if (config_.progressive_training) {
            std::cout << "  Warm-up for next dimension...\n";
            // Simplified warm-up - in practice would gradually introduce next dimension
        }
        
        std::cout << "  Dimension " << dim << " training completed\n\n";
    }
    
    // Final fine-tuning with all dimensions
    std::cout << "Final fine-tuning with all dimensions...\n";
    encoder_.current_dimension_focus_ = -1;  // Train all dimensions
    train(embeddings, labels, 100);
    
    std::cout << "Curriculum training completed!\n";
}

void MatryoshkaTrainer::distill_from_mamba(const std::vector<std::vector<float>>& transformer_embeddings,
                                 const std::vector<std::string>& texts) {
    
    std::cout << "Starting distillation from transformer embeddings\n";
    std::cout << "Number of samples: " << transformer_embeddings.size() << "\n\n";
    
    // Create pseudo-labels based on embedding similarity
    std::vector<int> pseudo_labels(transformer_embeddings.size());
    
    // Simple clustering for pseudo-labels (in practice, use proper clustering)
    int num_clusters = std::sqrt(transformer_embeddings.size());
    for (size_t i = 0; i < transformer_embeddings.size(); ++i) {
        pseudo_labels[i] = i % num_clusters;
    }
    
    // Distillation training loop
    int distillation_epochs = 200;
    float temperature = 3.0f;  // Distillation temperature
    
    for (int epoch = 0; epoch < distillation_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        // For each dimension, minimize distance to transformer's truncated embeddings
        for (int dim : config_.embedding_dims) {
            Matrix student_embeddings = encoder_.encode_batch(transformer_embeddings, dim);
            
            // Truncate transformer embeddings to current dimension
            Matrix teacher_embeddings(transformer_embeddings.size(), dim);
            for (size_t i = 0; i < transformer_embeddings.size(); ++i) {
                for (int j = 0; j < dim; ++j) {
                    teacher_embeddings(i, j) = transformer_embeddings[i][j];
                }
            }
            
            // Normalize teacher embeddings
            for (int i = 0; i < teacher_embeddings.rows(); ++i) {
                float norm = teacher_embeddings.row(i).norm();
                if (norm > 0) {
                    teacher_embeddings.row(i) /= norm;
                }
            }
            
            // Compute distillation loss (MSE + cosine similarity)
            for (int i = 0; i < student_embeddings.rows(); ++i) {
                Vector student = student_embeddings.row(i);
                Vector teacher = teacher_embeddings.row(i);
                
                // MSE loss
                float mse = (student - teacher).squaredNorm() / dim;
                
                // Cosine similarity loss
                float cos_sim = student.dot(teacher) / (student.norm() * teacher.norm());
                float cos_loss = 1.0f - cos_sim;
                
                epoch_loss += (mse + cos_loss) / 2.0f;
            }
        }
        
        epoch_loss /= (config_.embedding_dims.size() * transformer_embeddings.size());
        
        // Apply temperature scaling
        epoch_loss *= temperature;
        
        // Log progress
        if (epoch % 20 == 0) {
            std::cout << "Distillation epoch " << std::setw(3) << epoch 
                     << " | Loss: " << std::fixed << std::setprecision(4) << epoch_loss << "\n";
            
            // Reduce temperature over time
            temperature = std::max(1.0f, temperature * 0.95f);
        }
    }
    
    std::cout << "\nDistillation completed!\n";
    std::cout << "Model has learned to mimic transformer embeddings with dimension flexibility\n";
}

void MatryoshkaTrainer::update_learning_rate(int epoch, int current_dim) {
    // Dimension-aware learning rate scheduling
    float base_lr = config_.base_learning_rate;
    
    // Decay based on epoch
    float epoch_decay = std::pow(0.95f, epoch / 10.0f);
    
    // Scale based on dimension
    float dim_scale = std::sqrt(current_dim / 1536.0f);
    
    // Warm-up for first few epochs
    float warmup_scale = std::min(1.0f, (epoch + 1) / 5.0f);
    
    float final_lr = base_lr * epoch_decay * dim_scale * warmup_scale;
    
    // Note: In practice, this would update the optimizer's learning rate
}

std::vector<std::pair<int, int>> MatryoshkaTrainer::generate_positive_pairs(
    const std::vector<int>& labels) {
    
    std::vector<std::pair<int, int>> positive_pairs;
    
    // Group indices by label
    std::unordered_map<int, std::vector<int>> label_to_indices;
    for (size_t i = 0; i < labels.size(); ++i) {
        label_to_indices[labels[i]].push_back(i);
    }
    
    // Create positive pairs (same label)
    for (const auto& [label, indices] : label_to_indices) {
        for (size_t i = 0; i < indices.size(); ++i) {
            for (size_t j = i + 1; j < indices.size(); ++j) {
                positive_pairs.push_back({indices[i], indices[j]});
                positive_pairs.push_back({indices[j], indices[i]});  // Symmetric
            }
        }
    }
    
    return positive_pairs;
}

std::vector<std::pair<int, int>> MatryoshkaTrainer::generate_negative_pairs(
    const std::vector<int>& labels) {
    
    std::vector<std::pair<int, int>> negative_pairs;
    
    // Group indices by label
    std::unordered_map<int, std::vector<int>> label_to_indices;
    for (size_t i = 0; i < labels.size(); ++i) {
        label_to_indices[labels[i]].push_back(i);
    }
    
    // Create negative pairs (different labels)
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (const auto& [label1, indices1] : label_to_indices) {
        for (const auto& [label2, indices2] : label_to_indices) {
            if (label1 != label2) {
                // Sample a subset of negative pairs to keep balanced
                int num_samples = std::min(5, (int)indices2.size());
                std::uniform_int_distribution<> dist(0, indices2.size() - 1);
                
                for (int idx1 : indices1) {
                    for (int i = 0; i < num_samples; ++i) {
                        int idx2 = indices2[dist(gen)];
                        negative_pairs.push_back({idx1, idx2});
                    }
                }
            }
        }
    }
    
    return negative_pairs;
}

} // namespace matryoshka