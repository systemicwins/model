#ifndef CONTEXT_FIRST_ARCHITECTURE_H
#define CONTEXT_FIRST_ARCHITECTURE_H

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <string>

/**
 * Context-First Recursive Hybrid Architecture for Financial Prediction
 *
 * Key Design Principle: Economic context (FRED data) is processed FIRST,
 * then used to interpret market data, with recursive refinement between both domains.
 *
 * This corrects the fundamental flaw of processing market data in isolation.
 */

namespace financial {

// Economic context structure - represents the economic foundation
struct EconomicContext {
    Eigen::VectorXf interest_rate_regime;  // Federal Funds Rate, Treasury yields
    Eigen::VectorXf inflation_trend;       // CPI, inflation expectations
    Eigen::VectorXf growth_cycle;          // GDP, industrial production
    Eigen::VectorXf employment_conditions; // Unemployment, labor market
    std::string market_regime;             // Current economic regime

    // Context quality metrics
    float context_confidence;
    float economic_stability;
};

// Market interpretation within economic context
struct MarketInterpretation {
    Eigen::VectorXf price_interpretation;
    Eigen::VectorXf volume_interpretation;
    Eigen::VectorXf directional_bias;
    Eigen::VectorXf volatility_regime;
    float economic_alignment;  // How well market aligns with economic context
};

// Configuration for context-first processing
struct ContextFirstConfig {
    int economic_context_dim = 128;
    int market_interpretation_dim = 128;
    int economic_recursion_depth = 7;    // Deep analysis of economic fundamentals
    int market_recursion_depth = 5;      // Medium analysis of market behavior
    int cross_refinement_depth = 5;      // Refinement between domains
    float ssm_attention_balance = 0.7f;  // Favor SSM for time series
    float base_sparsity_ratio = 0.15f;   // Moderate sparsity for financial data
};

// Economic Context Analyzer - Processes FRED data FIRST
class EconomicContextAnalyzer {
public:
    EconomicContextAnalyzer(const ContextFirstConfig& config);
    ~EconomicContextAnalyzer() = default;

    // Main entry point: Analyze FRED data to establish economic context
    EconomicContext analyze_context(const Eigen::MatrixXf& fred_data);

    // Deep recursive analysis of economic indicators
    Eigen::MatrixXf deep_economic_analysis(const Eigen::MatrixXf& economic_data);

private:
    // Extract specific economic components
    Eigen::VectorXf extract_interest_rate_regime(const Eigen::MatrixXf& context_state);
    Eigen::VectorXf extract_inflation_trend(const Eigen::MatrixXf& context_state);
    Eigen::VectorXf extract_growth_cycle(const Eigen::MatrixXf& context_state);
    Eigen::VectorXf extract_employment_conditions(const Eigen::MatrixXf& context_state);

    // Determine overall market regime based on economic indicators
    std::string determine_market_regime(const Eigen::VectorXf& interest,
                                       const Eigen::VectorXf& inflation,
                                       const Eigen::VectorXf& growth,
                                       const Eigen::VectorXf& employment);

    ContextFirstConfig config_;
    int recursion_depth_;
};

// Market Data Interpreter - Uses economic context to interpret market behavior
class MarketDataInterpreter {
public:
    MarketDataInterpreter(const ContextFirstConfig& config);
    ~MarketDataInterpreter() = default;

    // Interpret market data within established economic context
    MarketInterpretation interpret_market(const Eigen::MatrixXf& market_data,
                                        const EconomicContext& economic_context);

private:
    // Context-guided market analysis
    Eigen::VectorXf context_guided_price_analysis(const Eigen::MatrixXf& price_data,
                                                 const EconomicContext& context);
    Eigen::VectorXf context_guided_volume_analysis(const Eigen::MatrixXf& volume_data,
                                                  const EconomicContext& context);

    ContextFirstConfig config_;
    int recursion_depth_;
};

// Cross-Domain Refinement - Recursive reasoning between economic and market domains
class CrossDomainRefinement {
public:
    CrossDomainRefinement(const ContextFirstConfig& config);
    ~CrossDomainRefinement() = default;

    // Multi-step refinement between economic context and market interpretation
    Eigen::MatrixXf refine_understanding(const EconomicContext& economic_context,
                                        const MarketInterpretation& market_insights);

private:
    // Cross-attention between economic and market understanding
    Eigen::MatrixXf cross_domain_attention(const Eigen::MatrixXf& economic_features,
                                          const Eigen::MatrixXf& market_features);

    ContextFirstConfig config_;
    int refinement_depth_;
};

// Main Context-First Recursive Hybrid Model
class ContextFirstRecursiveModel {
public:
    ContextFirstRecursiveModel(const ContextFirstConfig& config);
    ~ContextFirstRecursiveModel() = default;

    // Complete context-first reasoning pipeline
    Eigen::VectorXf predict(const Eigen::MatrixXf& fred_data,
                           const Eigen::MatrixXf& market_data);

    // Get economic context for analysis
    EconomicContext get_economic_context(const Eigen::MatrixXf& fred_data);

    // Interpret market data within context
    MarketInterpretation interpret_market_context(const Eigen::MatrixXf& market_data,
                                                 const EconomicContext& context);

private:
    ContextFirstConfig config_;

    std::unique_ptr<EconomicContextAnalyzer> economic_analyzer_;
    std::unique_ptr<MarketDataInterpreter> market_interpreter_;
    std::unique_ptr<CrossDomainRefinement> cross_refinement_;

    // Final prediction layer
    Eigen::MatrixXf prediction_weights_;
    Eigen::VectorXf prediction_bias_;
};

// FRED Data Loader - Loads economic indicators from collected data
class FREDDataLoader {
public:
    FREDDataLoader(const std::string& data_directory);
    ~FREDDataLoader() = default;

    // Load specific economic indicators
    Eigen::MatrixXf load_economic_indicators(const std::vector<std::string>& indicators,
                                           int sequence_length);

    // Load combined training dataset
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> load_training_data(int sequence_length);

    // Get available indicators
    std::vector<std::string> get_available_indicators();

private:
    std::string data_directory_;
    std::vector<std::string> available_indicators_;
};

// Financial Prediction Pipeline - Complete context-first prediction system
class FinancialPredictionPipeline {
public:
    FinancialPredictionPipeline(const std::string& fred_data_path);
    ~FinancialPredictionPipeline() = default;

    // Run complete context-first prediction
    Eigen::VectorXf predict_market_movement(const std::vector<std::string>& economic_indicators,
                                          const Eigen::MatrixXf& market_data);

    // Multi-step prediction with recursive reasoning
    std::vector<Eigen::VectorXf> predict_multi_step(int prediction_horizon);

    // Analyze economic regime and market implications
    std::string analyze_economic_regime(const std::vector<std::string>& key_indicators);

private:
    ContextFirstConfig config_;
    std::unique_ptr<FREDDataLoader> fred_loader_;
    std::unique_ptr<ContextFirstRecursiveModel> model_;

    // Current economic context (maintained across predictions)
    EconomicContext current_economic_context_;
};

} // namespace financial

#endif // CONTEXT_FIRST_ARCHITECTURE_H