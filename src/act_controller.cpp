#include "../include/act_controller.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace transformer {

// ==========================================
// ACTController Implementation
// ==========================================

ACTController::ACTController(const ACTConfig& config)
    : config_(config), current_epsilon_(config.halt_epsilon) {

    // Initialize random number generator
    rng_ = std::mt19937(std::random_device{}());
    uniform_dist_ = std::uniform_real_distribution<float>(0.0f, 1.0f);

    // Initialize Q-network
    initialize_q_network();

    // Initialize statistics
    total_computational_steps_ = 0.0f;
    total_reward_ = 0.0f;
    total_decisions_ = 0;
    early_halting_count_ = 0;
    total_confidence_ = 0.0f;

    std::cout << "🎯 ACT Controller initialized with max_steps=" << config.max_steps
              << ", epsilon=" << current_epsilon_ << std::endl;
}

ACTDecision ACTController::make_decision(const Eigen::MatrixXf& current_state,
                                       const Eigen::MatrixXf& previous_output,
                                       int current_step,
                                       bool is_training) {

    ACTDecision decision;

    // Extract state features for Q-learning
    Eigen::VectorXf state_features = extract_state_features(current_state, previous_output);

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
    Eigen::VectorXf q_values = compute_q_values(state_features);

    // Select action (0 = continue, 1 = halt)
    int action = select_action(q_values, is_training);

    // Determine decision based on action
    if (action == 1) { // Halt
        decision.should_halt = true;
        decision.should_continue = false;
        decision.halt_probability = q_values(1) / (q_values(0) + q_values(1));
        decision.continue_probability = q_values(0) / (q_values(0) + q_values(1));
        decision.computational_steps = current_step;

        if (current_step < config_.max_steps) {
            early_halting_count_++;
        }
    } else { // Continue
        decision.should_halt = false;
        decision.should_continue = true;
        decision.halt_probability = q_values(1) / (q_values(0) + q_values(1));
        decision.continue_probability = q_values(0) / (q_values(0) + q_values(1));
        decision.computational_steps = current_step + 1;
    }

    // Update statistics
    total_decisions_++;
    total_computational_steps_ += decision.computational_steps;
    total_confidence_ += decision.confidence_score;

    return decision;
}

void ACTController::update_q_values(const Eigen::MatrixXf& state,
                                  int action,
                                  float reward,
                                  const Eigen::MatrixXf& next_state) {

    Eigen::VectorXf current_state_features = extract_state_features(state, Eigen::MatrixXf());
    Eigen::VectorXf next_state_features = extract_state_features(next_state, Eigen::MatrixXf());

    Eigen::VectorXf current_q = compute_q_values(current_state_features);
    Eigen::VectorXf next_q = compute_q_values(next_state_features);

    // Q-learning update
    float target_q = reward + config_.discount_factor * next_q.maxCoeff();
    float td_error = target_q - current_q(action);

    // Update weights and bias using gradient descent
    Eigen::VectorXf gradient = current_state_features;
    if (action == 0) {
        gradient = -gradient; // Negative gradient for continue action
    }

    q_weights_.col(action) += config_.learning_rate * td_error * gradient;
    q_bias_(action) += config_.learning_rate * td_error;

    total_reward_ += reward;
}

void ACTController::decay_epsilon() {
    if (config_.use_adaptive_epsilon) {
        current_epsilon_ = std::max(config_.min_epsilon,
                                   current_epsilon_ * config_.epsilon_decay);
    }
}

void ACTController::reset() {
    current_epsilon_ = config_.halt_epsilon;
    total_computational_steps_ = 0.0f;
    total_reward_ = 0.0f;
    total_decisions_ = 0;
    early_halting_count_ = 0;
    total_confidence_ = 0.0f;
}

ACTController::ACTStats ACTController::get_statistics() const {
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

// Private methods implementation

Eigen::VectorXf ACTController::extract_state_features(const Eigen::MatrixXf& current_state,
                                                    const Eigen::MatrixXf& previous_output) {
    Eigen::VectorXf features(config_.state_dim);

    // Basic state statistics
    float mean_val = current_state.mean();
    float std_val = std::sqrt((current_state.array() - mean_val).square().sum() / current_state.size());
    float max_val = current_state.maxCoeff();
    float min_val = current_state.minCoeff();

    // Fill features with state statistics
    int idx = 0;
    features(idx++) = mean_val;
    features(idx++) = std_val;
    features(idx++) = max_val;
    features(idx++) = min_val;

    // Add row-wise statistics if we have enough space
    if (current_state.rows() > 0 && idx < config_.state_dim) {
        Eigen::VectorXf row_means = current_state.rowwise().mean();
        features(idx++) = row_means.mean();
        features(idx++) = row_means.maxCoeff();
        features(idx++) = row_means.minCoeff();
    }

    // Add column-wise statistics if we have enough space
    if (current_state.cols() > 0 && idx < config_.state_dim) {
        Eigen::VectorXf col_means = current_state.colwise().mean();
        features(idx++) = col_means.mean();
        features(idx++) = col_means.maxCoeff();
        features(idx++) = col_means.minCoeff();
    }

    // Fill remaining features with normalized values
    while (idx < config_.state_dim) {
        features(idx++) = 0.0f;
    }

    return features;
}

Eigen::VectorXf ACTController::compute_q_values(const Eigen::VectorXf& state_features) {
    return q_network_forward(state_features);
}

int ACTController::select_action(const Eigen::VectorXf& q_values, bool is_training) {
    if (!is_training) {
        // Greedy action selection during inference
        return (q_values(0) > q_values(1)) ? 0 : 1;
    }

    // Epsilon-greedy action selection during training
    float rand_val = uniform_dist_(rng_);
    if (rand_val < current_epsilon_) {
        // Explore: random action
        return (uniform_dist_(rng_) < 0.5f) ? 0 : 1;
    } else {
        // Exploit: best action
        return (q_values(0) > q_values(1)) ? 0 : 1;
    }
}

float ACTController::compute_confidence(const Eigen::MatrixXf& current_state,
                                      const Eigen::MatrixXf& previous_output) {
    if (config_.use_confidence_threshold) {
        // Simple confidence measure based on state stability
        float stability = 1.0f / (1.0f + current_state.norm());

        // If we have previous output, compare with current state
        if (previous_output.size() > 0) {
            float similarity = 1.0f / (1.0f + (current_state - previous_output).norm());
            return (stability + similarity) / 2.0f;
        }

        return stability;
    }

    return 0.5f; // Default confidence
}

float ACTController::compute_reward(bool halted_early, int steps_used, float confidence) const {
    // Reward function for ACT
    float efficiency_reward = (config_.max_steps - steps_used) / static_cast<float>(config_.max_steps);
    float confidence_reward = confidence;

    // Penalty for halting too early if confidence is low
    float early_penalty = 0.0f;
    if (halted_early && confidence < config_.confidence_threshold) {
        early_penalty = (config_.confidence_threshold - confidence) * 2.0f;
    }

    return efficiency_reward + confidence_reward - early_penalty;
}

void ACTController::initialize_q_network() {
    // Initialize Q-network weights and bias
    q_weights_ = Eigen::MatrixXf::Random(config_.state_dim, 2) * 0.1f;
    q_bias_ = Eigen::VectorXf::Zero(2);
}

Eigen::VectorXf ACTController::q_network_forward(const Eigen::VectorXf& state_features) {
    Eigen::VectorXf q_values = state_features.transpose() * q_weights_ + q_bias_;
    return q_values;
}

} // namespace transformer