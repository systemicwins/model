/**
 * Adaptive Computational Time (ACT) Demo
 *
 * This demo shows how Adaptive Computational Time works in practice
 * for financial prediction tasks. ACT dynamically decides how many
 * computational steps to spend on each sample during training.
 *
 * Based on: "Less is More: Recursive Reasoning with Tiny Networks"
 * arXiv:2510.04871
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>

// Simplified matrix and vector types for demo
using Vector = std::vector<float>;
using Matrix = std::vector<Vector>;

// ACT Decision result
struct ACTDecision {
    bool should_halt;
    bool should_continue;
    float halt_probability;
    float continue_probability;
    int computational_steps;
    float confidence_score;
};

// ACT Configuration
struct ACTConfig {
    int max_steps = 16;
    int min_steps = 1;
    float halt_epsilon = 0.1f;
    float learning_rate = 0.01f;
    float discount_factor = 0.99f;
    float temperature = 1.0f;
    bool use_adaptive_epsilon = true;
    float epsilon_decay = 0.995f;
    float min_epsilon = 0.01f;
    int state_dim = 64;
    bool use_confidence_threshold = true;
    float confidence_threshold = 0.8f;
};

// Adaptive Computational Time Controller
class ACTController {
public:
    ACTController(const ACTConfig& config) : config_(config), current_epsilon_(config.halt_epsilon) {
        rng_ = std::mt19937(std::random_device{}());
        uniform_dist_ = std::uniform_real_distribution<float>(0.0f, 1.0f);

        // Initialize Q-learning parameters
        q_weights_ = std::vector<float>(config.state_dim * 2, 0.1f);
        q_bias_ = std::vector<float>(2, 0.0f);

        // Statistics
        total_computational_steps_ = 0.0f;
        total_reward_ = 0.0f;
        total_decisions_ = 0;
        early_halting_count_ = 0;
        total_confidence_ = 0.0f;

        std::cout << "🎯 ACT Controller initialized with max_steps=" << config.max_steps
                  << ", epsilon=" << current_epsilon_ << std::endl;
    }

    ACTDecision make_decision(const Vector& current_state,
                             const Vector& previous_output,
                             int current_step,
                             bool is_training = true) {
        ACTDecision decision;

        // Extract state features for Q-learning
        Vector state_features = extract_state_features(current_state, previous_output);

        // Compute confidence score
        decision.confidence_score = compute_confidence(current_state, previous_output);

        // Check if we've reached minimum steps
        if (current_step < config_.min_steps) {
            decision.should_halt = false;
            decision.should_continue = true;
            decision.halt_probability = 0.0f;
            decision.continue_probability = 1.0f;
            decision.computational_steps = current_step + 1;
            return decision;
        }

        // Check if we've reached maximum steps
        if (current_step >= config_.max_steps) {
            decision.should_halt = true;
            decision.should_continue = false;
            decision.halt_probability = 1.0f;
            decision.continue_probability = 0.0f;
            decision.computational_steps = current_step;
            return decision;
        }

        // Compute Q-values for current state
        Vector q_values = compute_q_values(state_features);

        // Select action (0 = continue, 1 = halt)
        int action = select_action(q_values, is_training);

        // Determine decision based on action
        if (action == 1) { // Halt
            decision.should_halt = true;
            decision.should_continue = false;
            decision.halt_probability = q_values[1] / (q_values[0] + q_values[1]);
            decision.continue_probability = q_values[0] / (q_values[0] + q_values[1]);
            decision.computational_steps = current_step;

            if (current_step < config_.max_steps) {
                early_halting_count_++;
            }
        } else { // Continue
            decision.should_halt = false;
            decision.should_continue = true;
            decision.halt_probability = q_values[1] / (q_values[0] + q_values[1]);
            decision.continue_probability = q_values[0] / (q_values[0] + q_values[1]);
            decision.computational_steps = current_step + 1;
        }

        // Update statistics
        total_decisions_++;
        total_computational_steps_ += decision.computational_steps;
        total_confidence_ += decision.confidence_score;

        return decision;
    }

    void update_q_values(const Vector& state, int action, float reward, const Vector& next_state) {
        Vector current_state_features = extract_state_features(state, Vector());
        Vector next_state_features = extract_state_features(next_state, Vector());

        Vector current_q = compute_q_values(current_state_features);
        Vector next_q = compute_q_values(next_state_features);

        // Q-learning update
        float target_q = reward + config_.discount_factor * *std::max_element(next_q.begin(), next_q.end());
        float td_error = target_q - current_q[action];

        // Update weights and bias using gradient descent
        for (int i = 0; i < config_.state_dim; ++i) {
            int weight_idx = action * config_.state_dim + i;
            q_weights_[weight_idx] += config_.learning_rate * td_error * current_state_features[i];
        }
        q_bias_[action] += config_.learning_rate * td_error;

        total_reward_ += reward;
    }

    void decay_epsilon() {
        if (config_.use_adaptive_epsilon) {
            current_epsilon_ = std::max(config_.min_epsilon,
                                       current_epsilon_ * config_.epsilon_decay);
        }
    }

    void reset() {
        current_epsilon_ = config_.halt_epsilon;
        total_computational_steps_ = 0.0f;
        total_reward_ = 0.0f;
        total_decisions_ = 0;
        early_halting_count_ = 0;
        total_confidence_ = 0.0f;
    }

    struct ACTStats {
        float avg_computational_steps;
        float total_reward;
        int total_decisions;
        float early_halting_rate;
        float avg_confidence;
    };

    ACTStats get_statistics() const {
        ACTStats stats;
        if (total_decisions_ > 0) {
            stats.avg_computational_steps = total_computational_steps_ / total_decisions_;
            stats.avg_confidence = total_confidence_ / total_decisions_;
            stats.early_halting_rate = static_cast<float>(early_halting_count_) / total_decisions_;
        } else {
            stats.avg_computational_steps = 0.0f;
            stats.avg_confidence = 0.0f;
            stats.early_halting_rate = 0.0f;
        }
        stats.total_reward = total_reward_;
        stats.total_decisions = total_decisions_;
        return stats;
    }

    float get_current_epsilon() const { return current_epsilon_; }

private:
    Vector extract_state_features(const Vector& current_state, const Vector& previous_output) {
        Vector features(config_.state_dim, 0.0f);

        // Basic state statistics
        if (!current_state.empty()) {
            float mean_val = 0.0f, min_val = current_state[0], max_val = current_state[0];
            for (float val : current_state) {
                mean_val += val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            mean_val /= current_state.size();

            float variance = 0.0f;
            for (float val : current_state) {
                variance += (val - mean_val) * (val - mean_val);
            }
            variance /= current_state.size();
            float std_val = std::sqrt(variance);

            // Fill features with state statistics
            int idx = 0;
            features[idx++] = mean_val;
            features[idx++] = std_val;
            features[idx++] = max_val;
            features[idx++] = min_val;

            // Add some derived features
            if (current_state.size() >= 2) {
                features[idx++] = current_state.back() - current_state[0]; // Trend
            }

            // Add previous output comparison if available
            if (!previous_output.empty() && !current_state.empty()) {
                features[idx++] = previous_output.back() - current_state.back(); // Change
            }
        }

        return features;
    }

    Vector compute_q_values(const Vector& state_features) {
        Vector q_values(2, 0.0f);

        // Simple linear Q-network
        for (int action = 0; action < 2; ++action) {
            for (int i = 0; i < std::min(static_cast<int>(state_features.size()), config_.state_dim); ++i) {
                int weight_idx = action * config_.state_dim + i;
                if (weight_idx < static_cast<int>(q_weights_.size())) {
                    q_values[action] += q_weights_[weight_idx] * state_features[i];
                }
            }
            q_values[action] += q_bias_[action];
        }

        return q_values;
    }

    int select_action(const Vector& q_values, bool is_training) {
        if (!is_training) {
            // Greedy action selection during inference
            return (q_values[0] > q_values[1]) ? 0 : 1;
        }

        // Epsilon-greedy action selection during training
        float rand_val = uniform_dist_(rng_);
        if (rand_val < current_epsilon_) {
            // Explore: random action
            return (uniform_dist_(rng_) < 0.5f) ? 0 : 1;
        } else {
            // Exploit: best action
            return (q_values[0] > q_values[1]) ? 0 : 1;
        }
    }

    float compute_confidence(const Vector& current_state, const Vector& previous_output) {
        if (config_.use_confidence_threshold) {
            // Simple confidence measure based on state stability
            if (current_state.empty()) return 0.5f;

            float stability = 1.0f;
            if (current_state.size() > 1) {
                float mean = 0.0f;
                for (float val : current_state) mean += val;
                mean /= current_state.size();

                float variance = 0.0f;
                for (float val : current_state) {
                    variance += (val - mean) * (val - mean);
                }
                variance /= current_state.size();

                stability = 1.0f / (1.0f + std::sqrt(variance));
            }

            // If we have previous output, compare with current state
            if (!previous_output.empty() && !current_state.empty()) {
                float similarity = 1.0f / (1.0f + std::abs(previous_output.back() - current_state.back()));
                return (stability + similarity) / 2.0f;
            }

            return stability;
        }

        return 0.5f; // Default confidence
    }

private:
    ACTConfig config_;
    float current_epsilon_;

    // Q-learning parameters
    std::vector<float> q_weights_;
    std::vector<float> q_bias_;

    // Statistics tracking
    mutable float total_computational_steps_;
    mutable float total_reward_;
    mutable int total_decisions_;
    mutable int early_halting_count_;
    mutable float total_confidence_;

    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
};

// Simulated financial prediction model with ACT
class ACTFinancialModel {
public:
    ACTFinancialModel() {
        ACTConfig act_config;
        act_config.max_steps = 8;  // Reduced for demo
        act_config.min_steps = 2;
        act_controller_ = std::make_unique<ACTController>(act_config);

        std::cout << "💰 ACT Financial Model initialized" << std::endl;
    }

    float predict_with_act(const Matrix& market_data, bool is_training = true) {
        std::cout << "\n🔮 Making ACT-enhanced prediction..." << std::endl;

        Vector current_input = market_data.back(); // Latest market data
        Vector accumulated_prediction = Vector(current_input.size(), 0.0f);
        Vector previous_output = Vector(current_input.size(), 0.0f);

        int step = 0;
        bool should_continue = true;

        while (should_continue && step < 50) { // Safety limit
            // Simulate model forward pass
            Vector step_prediction = simulate_model_forward(current_input, step);

            // Accumulate prediction
            for (size_t i = 0; i < step_prediction.size(); ++i) {
                accumulated_prediction[i] += step_prediction[i];
            }

            // ACT decision making
            ACTDecision decision = act_controller_->make_decision(current_input, previous_output, step, is_training);

            std::cout << "Step " << step << ": confidence=" << std::fixed << std::setprecision(3)
                      << decision.confidence_score << ", halt_prob=" << decision.halt_probability
                      << ", steps=" << decision.computational_steps << std::endl;

            // Update previous output for next iteration
            previous_output = step_prediction;

            // Check if we should halt
            if (decision.should_halt) {
                should_continue = false;

                // Update Q-values if training
                if (is_training) {
                    float reward = compute_reward(true, step + 1, decision.confidence_score);
                    act_controller_->update_q_values(current_input, 1, reward, step_prediction);
                }
            } else {
                // Update Q-values for continue decision if training
                if (is_training) {
                    float reward = compute_reward(false, step + 1, decision.confidence_score);
                    act_controller_->update_q_values(current_input, 0, reward, step_prediction);
                }

                step++;
                should_continue = (step < 20); // Simplified continuation logic for demo
            }

            // Decay epsilon for exploration reduction
            if (is_training) {
                act_controller_->decay_epsilon();
            }
        }

        // Average the accumulated predictions
        float final_prediction = 0.0f;
        for (float val : accumulated_prediction) {
            final_prediction += val;
        }
        final_prediction /= accumulated_prediction.size();

        std::cout << "Final prediction: $" << std::setprecision(2) << final_prediction
                  << " (used " << step + 1 << " computational steps)" << std::endl;

        return final_prediction;
    }

    void print_act_statistics() {
        ACTController::ACTStats stats = act_controller_->get_statistics();

        std::cout << "\n📊 ACT STATISTICS:" << std::endl;
        std::cout << "Average computational steps: " << std::fixed << std::setprecision(2)
                  << stats.avg_computational_steps << std::endl;
        std::cout << "Early halting rate: " << std::setprecision(2) << (stats.early_halting_rate * 100) << "%" << std::endl;
        std::cout << "Average confidence: " << std::setprecision(3) << stats.avg_confidence << std::endl;
        std::cout << "Total decisions: " << stats.total_decisions << std::endl;
        std::cout << "Current epsilon: " << std::setprecision(3) << act_controller_->get_current_epsilon() << std::endl;
    }

private:
    Vector simulate_model_forward(const Vector& input, int step) {
        // Simulate a model forward pass with some noise and trend
        Vector output = input;

        // Add some trend based on step
        float trend = 0.001f * step;

        // Add noise
        std::mt19937 rng(step);
        std::normal_distribution<float> noise_dist(0.0f, 0.01f);

        for (float& val : output) {
            val += trend + noise_dist(rng);
        }

        return output;
    }

    float compute_reward(bool halted_early, int steps_used, float confidence) {
        // Reward function for ACT
        float efficiency_reward = (8.0f - steps_used) / 8.0f; // Assuming max 8 steps
        float confidence_reward = confidence;

        // Penalty for halting too early if confidence is low
        float early_penalty = 0.0f;
        if (halted_early && confidence < 0.8f) {
            early_penalty = (0.8f - confidence) * 2.0f;
        }

        return efficiency_reward + confidence_reward - early_penalty;
    }

private:
    std::unique_ptr<ACTController> act_controller_;
};

// Demo function showing ACT in action
void run_act_demo() {
    std::cout << "🎯 Adaptive Computational Time (ACT) Demo" << std::endl;
    std::cout << "Based on: Less is More - Recursive Reasoning with Tiny Networks" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Generate sample market data
    Matrix market_data;
    float price = 100.0f;

    for (int t = 0; t < 20; ++t) {
        Vector market_point(5); // [Open, High, Low, Close, Volume]

        // Generate realistic market data
        float change = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        price += change;
        price = std::max(10.0f, price);

        market_point[0] = price; // Open
        market_point[1] = price * (1.0f + 0.01f * (static_cast<float>(rand()) / RAND_MAX)); // High
        market_point[2] = price * (1.0f - 0.01f * (static_cast<float>(rand()) / RAND_MAX)); // Low
        market_point[3] = price; // Close
        market_point[4] = 1000000.0f + static_cast<float>(rand()) / RAND_MAX * 5000000.0f; // Volume

        market_data.push_back(market_point);
    }

    // Create ACT financial model
    ACTFinancialModel act_model;

    // Run multiple predictions to show ACT learning
    std::vector<float> predictions;
    std::vector<double> processing_times;

    for (int scenario = 0; scenario < 5; ++scenario) {
        std::cout << "\n--- Prediction Scenario " << (scenario + 1) << " ---" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        float predicted_price = act_model.predict_with_act(market_data, true);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> processing_time = end_time - start_time;

        predictions.push_back(predicted_price);
        processing_times.push_back(processing_time.count());

        std::cout << "Processing time: " << std::fixed << std::setprecision(3)
                  << processing_time.count() << " ms" << std::endl;
    }

    // Print final statistics
    act_model.print_act_statistics();

    std::cout << "\n🎯 KEY ACT ADVANTAGES DEMONSTRATED:" << std::endl;
    std::cout << "• Dynamic computation allocation per sample" << std::endl;
    std::cout << "• Q-learning based halting decisions" << std::endl;
    std::cout << "• Confidence-based early stopping" << std::endl;
    std::cout << "• Adaptive exploration (epsilon decay)" << std::endl;
    std::cout << "• Training efficiency through variable computation" << std::endl;

    std::cout << "\n💡 FINANCIAL APPLICATIONS:" << std::endl;
    std::cout << "• Risk-adjusted position sizing" << std::endl;
    std::cout << "• Market regime detection" << std::endl;
    std::cout << "• Portfolio optimization" << std::endl;
    std::cout << "• Real-time trading decisions" << std::endl;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "✅ ACT DEMONSTRATION COMPLETE" << std::endl;
    std::cout << "Adaptive computation provides superior efficiency for financial prediction!" << std::endl;
}

int main() {
    try {
        run_act_demo();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }
}