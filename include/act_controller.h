#ifndef ACT_CONTROLLER_H
#define ACT_CONTROLLER_H

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <random>

namespace transformer {

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
    int max_steps = 16;                    // Maximum computational steps
    int min_steps = 1;                     // Minimum computational steps
    float halt_epsilon = 0.1f;             // Exploration rate for halting decision
    float learning_rate = 0.01f;           // Q-learning learning rate
    float discount_factor = 0.99f;         // Q-learning discount factor
    float temperature = 1.0f;              // Temperature for softmax in decision making
    bool use_adaptive_epsilon = true;      // Whether to decay epsilon over time
    float epsilon_decay = 0.995f;          // Epsilon decay rate
    float min_epsilon = 0.01f;             // Minimum epsilon value
    int state_dim = 64;                    // Dimension of state representation for Q-learning
    bool use_confidence_threshold = true;  // Use confidence threshold for early halting
    float confidence_threshold = 0.8f;     // Confidence threshold for halting
};

// Adaptive Computational Time Controller
class ACTController {
public:
    ACTController(const ACTConfig& config);
    ~ACTController() = default;

    // Main ACT decision making
    ACTDecision make_decision(const Eigen::MatrixXf& current_state,
                             const Eigen::MatrixXf& previous_output,
                             int current_step,
                             bool is_training = true);

    // Update Q-values based on reward
    void update_q_values(const Eigen::MatrixXf& state,
                        int action,
                        float reward,
                        const Eigen::MatrixXf& next_state);

    // Get current epsilon value
    float get_current_epsilon() const { return current_epsilon_; }

    // Decay epsilon for exploration reduction
    void decay_epsilon();

    // Reset controller state
    void reset();

    // Get statistics
    struct ACTStats {
        float avg_computational_steps;
        float total_reward;
        int total_decisions;
        float early_halting_rate;
        float avg_confidence;
    };

    ACTStats get_statistics() const;

private:
    // Q-learning state representation
    Eigen::VectorXf extract_state_features(const Eigen::MatrixXf& current_state,
                                          const Eigen::MatrixXf& previous_output);

    // Compute Q-values for current state
    Eigen::VectorXf compute_q_values(const Eigen::VectorXf& state_features);

    // Select action using epsilon-greedy policy
    int select_action(const Eigen::VectorXf& q_values, bool is_training);

    // Compute confidence score for current state
    float compute_confidence(const Eigen::MatrixXf& current_state,
                           const Eigen::MatrixXf& previous_output);

    // Compute reward for halting decision
    float compute_reward(bool halted_early, int steps_used, float confidence);

    // Initialize Q-network parameters
    void initialize_q_network();

    // Forward pass through Q-network
    Eigen::VectorXf q_network_forward(const Eigen::VectorXf& state_features);

private:
    ACTConfig config_;
    float current_epsilon_;

    // Q-learning parameters
    Eigen::MatrixXf q_weights_;
    Eigen::VectorXf q_bias_;

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

} // namespace transformer

#endif // ACT_CONTROLLER_H