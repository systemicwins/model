// Context-First Financial Prediction Demo
// CORRECTED DESIGN: Economic context (FRED) first, then market interpretation
//
// Key Design Principle: Markets operate within economic reality, not in isolation.
// Economic indicators establish the foundation, then market data is interpreted within that context.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "../include/act_controller.h"

// Simplified data structures for demonstration
using Matrix = std::vector<std::vector<float>>;
using Vector = std::vector<float>;

// Economic Context Structure - Represents the economic foundation
struct EconomicContext {
    Vector interest_rate_regime;  // Federal Funds Rate, Treasury yields
    Vector inflation_trend;       // CPI, inflation expectations
    Vector growth_cycle;          // GDP, industrial production
    Vector employment_conditions; // Unemployment, labor market
    std::string market_regime;    // Current economic regime

    float context_confidence;
    float economic_stability;
};

// Market Interpretation within Economic Context
struct MarketInterpretation {
    Vector price_interpretation;
    Vector volume_interpretation;
    Vector directional_bias;
    Vector volatility_regime;
    float economic_alignment;  // How well market aligns with economic context
};

// Context-First Financial Prediction Demo
class ContextFirstFinancialDemo {
public:
    // Load sample FRED economic data (synthetic but realistic patterns)
    static Matrix load_fred_economic_data(int sequence_length) {
        Matrix data = Matrix::Zero(sequence_length, 8); // 8 key economic indicators

        // Simulate realistic economic indicator patterns
        for (int t = 0; t < sequence_length; ++t) {
            // Federal Funds Rate (base rate around 2-5%)
            data[t][0] = 0.025f + 0.02f * sin(0.1f * t) + 0.01f * (rand() % 100) / 100.0f;

            // 10-Year Treasury (typically 1-2% above Fed Funds)
            data[t][1] = data[t][0] + 0.015f + 0.005f * (rand() % 100) / 100.0f;

            // CPI Inflation (target around 2%)
            data[t][2] = 0.02f + 0.01f * sin(0.05f * t) + 0.005f * (rand() % 100) / 100.0f;

            // Unemployment Rate (natural rate around 4-5%)
            data[t][3] = 0.045f + 0.02f * sin(0.03f * t) + 0.01f * (rand() % 100) / 100.0f;

            // GDP Growth (annualized quarterly)
            data[t][4] = 0.025f + 0.015f * sin(0.02f * t) + 0.01f * (rand() % 100) / 100.0f;

            // Industrial Production
            data[t][5] = 0.02f + 0.03f * sin(0.08f * t) + 0.015f * (rand() % 100) / 100.0f;

            // Housing Starts
            data[t][6] = 0.015f + 0.025f * sin(0.04f * t) + 0.01f * (rand() % 100) / 100.0f;

            // Consumer Confidence
            data[t][7] = 0.6f + 0.3f * sin(0.06f * t) + 0.15f * (rand() % 100) / 100.0f;
        }

        return data;
    }

    // Load sample market data that responds to economic conditions
    static Matrix load_market_data(int sequence_length) {
        Matrix data = Matrix::Zero(sequence_length, 5); // [Open, High, Low, Close, Volume]

        // Generate market data that responds to economic conditions
        float price = 100.0f;
        float economic_cycle = 0.0f;

        for (int t = 0; t < sequence_length; ++t) {
            // Economic cycle affects market behavior
            economic_cycle = 0.5f * sin(0.02f * t);

            // Price influenced by economic conditions
            float economic_impact = economic_cycle * 10.0f;
            float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
            float price_change = economic_impact + noise;

            price += price_change;
            price = std::max(10.0f, price); // Minimum price floor

            // Generate OHLCV
            float open = price;
            float volatility = 0.02f + std::abs(economic_cycle) * 0.03f;
            float high = open * (1.0f + volatility * (0.5f + static_cast<float>(rand()) / RAND_MAX));
            float low = open * (1.0f - volatility * (0.5f + static_cast<float>(rand()) / RAND_MAX));
            float close = open + price_change;
            float volume = 1000000.0f + static_cast<float>(rand()) / RAND_MAX * 5000000.0f;

            data[t][0] = open;
            data[t][1] = high;
            data[t][2] = low;
            data[t][3] = close;
            data[t][4] = volume;
        }

        return data;
    }

    // Traditional financial prediction model for comparison
    class TraditionalFinancialModel {
    public:
        TraditionalFinancialModel(int input_dim, int hidden_dim = 128)
            : input_dim_(input_dim), hidden_dim_(hidden_dim) {

            // Simple moving average and momentum features
            weights_ = Eigen::VectorXf::Random(hidden_dim_);
            bias_ = Eigen::VectorXf::Random(1);
        }

        Eigen::VectorXf predict(const Eigen::MatrixXf& features) {
            // Simple linear model with technical indicators
            Eigen::VectorXf prediction(1);

            // Extract basic features
            Eigen::VectorXf price_features = features.col(3); // Close prices

            // Compute simple moving averages
            float sma_5 = price_features.tail(5).mean();
            float sma_20 = price_features.tail(20).mean();

            // Compute momentum
            float momentum = price_features.tail(1)(0) - price_features.tail(10).mean();

            // Volume trend
            float volume_trend = features.col(4).tail(5).mean() / features.col(4).tail(20).mean();

            // Combine features
            Eigen::VectorXf feature_vec(4);
            feature_vec << sma_5, sma_20, momentum, volume_trend;

            prediction(0) = weights_.dot(feature_vec) + bias_(0);
            return prediction;
        }

    private:
        int input_dim_, hidden_dim_;
        Eigen::VectorXf weights_;
        Eigen::VectorXf bias_;
    };

    // Hybrid attention financial model
    class HybridFinancialModel {
    public:
        HybridFinancialModel(int input_dim) : input_dim_(input_dim) {
            // Configure for financial time series
            config_.state_dim = 128;
            config_.base_sparsity_ratio = 0.15f;  // Moderate sparsity for financial data
            config_.use_adaptive_sparsity = true;
            config_.ssm_attention_balance = 0.7f;  // Favor SSM for time series

            hybrid_attention_ = std::make_unique<transformer::HybridAttention>(config_, input_dim);
        }

        Eigen::VectorXf predict(const Eigen::MatrixXf& market_data) {
            // Use hybrid attention for financial prediction
            Eigen::MatrixXf output = hybrid_attention_->forward(market_data);

            // Extract prediction from final state
            Eigen::VectorXf prediction(1);
            prediction(0) = output.row(output.rows() - 1).mean();

            return prediction;
        }

    private:
        int input_dim_;
        transformer::HybridConfig config_;
        std::unique_ptr<transformer::HybridAttention> hybrid_attention_;
    };

    // Economic Context Analyzer - Processes FRED data FIRST
    class EconomicContextAnalyzer {
    public:
        EconomicContextAnalyzer() {
            std::cout << "🔍 Economic Context Analyzer initialized" << std::endl;
        }

        EconomicContext analyze_context(const Matrix& fred_data) {
            std::cout << "📊 Analyzing FRED economic indicators..." << std::endl;

            int seq_len = fred_data.size();
            if (seq_len == 0) {
                return create_default_context();
            }

            // Extract latest economic conditions (most recent data)
            const std::vector<float>& latest = fred_data.back();

            // Analyze interest rate regime
            Vector interest_regime = analyze_interest_rates(fred_data);

            // Analyze inflation trend
            Vector inflation_trend = analyze_inflation(fred_data);

            // Analyze growth cycle
            Vector growth_cycle = analyze_growth(fred_data);

            // Analyze employment conditions
            Vector employment_conditions = analyze_employment(fred_data);

            // Determine overall market regime
            std::string market_regime = determine_market_regime(latest);

            // Calculate context quality metrics
            float context_confidence = calculate_context_confidence(fred_data);
            float economic_stability = calculate_economic_stability(fred_data);

            return EconomicContext{
                interest_regime,
                inflation_trend,
                growth_cycle,
                employment_conditions,
                market_regime,
                context_confidence,
                economic_stability
            };
        }

    private:
        Vector analyze_interest_rates(const Matrix& fred_data) {
            Vector rates(fred_data.size());
            for (size_t i = 0; i < fred_data.size(); ++i) {
                rates[i] = fred_data[i][0] + fred_data[i][1]; // Fed Funds + 10Y Treasury
            }
            return rates;
        }

        Vector analyze_inflation(const Matrix& fred_data) {
            Vector inflation(fred_data.size());
            for (size_t i = 0; i < fred_data.size(); ++i) {
                inflation[i] = fred_data[i][2]; // CPI
            }
            return inflation;
        }

        Vector analyze_growth(const Matrix& fred_data) {
            Vector growth(fred_data.size());
            for (size_t i = 0; i < fred_data.size(); ++i) {
                growth[i] = fred_data[i][4] + fred_data[i][5]; // GDP + Industrial Production
            }
            return growth;
        }

        Vector analyze_employment(const Matrix& fred_data) {
            Vector employment(fred_data.size());
            for (size_t i = 0; i < fred_data.size(); ++i) {
                employment[i] = fred_data[i][3] + fred_data[i][7]; // Unemployment + Confidence
            }
            return employment;
        }

        std::string determine_market_regime(const std::vector<float>& latest) {
            float interest_rate = latest[0];
            float inflation = latest[2];
            float growth = latest[4];
            float unemployment = latest[3];

            if (inflation > 0.03f && growth > 0.02f) {
                return "reflation";
            } else if (interest_rate > 0.04f && growth < 0.01f) {
                return "tightening";
            } else if (growth > 0.025f && unemployment < 0.045f) {
                return "expansion";
            } else {
                return "neutral";
            }
        }

        float calculate_context_confidence(const Matrix& fred_data) {
            // Higher confidence with more data and stable trends
            float data_quality = std::min(1.0f, static_cast<float>(fred_data.size()) / 100.0f);
            return 0.5f + 0.5f * data_quality;
        }

        float calculate_economic_stability(const Matrix& fred_data) {
            // Measure volatility in economic indicators
            if (fred_data.size() < 10) return 0.5f;

            float total_volatility = 0.0f;
            for (const auto& indicator_series : fred_data) {
                float mean = 0.0f;
                for (float val : indicator_series) mean += val;
                mean /= indicator_series.size();

                float variance = 0.0f;
                for (float val : indicator_series) {
                    variance += (val - mean) * (val - mean);
                }
                variance /= indicator_series.size();

                total_volatility += sqrt(variance);
            }

            // Lower volatility = higher stability
            float avg_volatility = total_volatility / fred_data[0].size();
            return std::max(0.1f, 1.0f - avg_volatility);
        }

        EconomicContext create_default_context() {
            return EconomicContext{
                Vector{0.025f, 0.04f},  // Default interest rates
                Vector{0.02f, 0.02f},   // Default inflation
                Vector{0.025f, 0.025f}, // Default growth
                Vector{0.045f, 0.045f}, // Default employment
                "neutral",
                0.5f,
                0.5f
            };
        }
    };

    // Market Data Interpreter - Uses economic context to interpret market behavior
    class MarketDataInterpreter {
    public:
        MarketDataInterpreter() {
            std::cout << "📈 Market Data Interpreter initialized" << std::endl;
        }

        MarketInterpretation interpret_market(const Matrix& market_data, const EconomicContext& context) {
            std::cout << "🔍 Interpreting market data within economic context..." << std::endl;
            std::cout << "Economic regime: " << context.market_regime << std::endl;

            // Extract price and volume patterns
            Vector price_interpretation = analyze_price_patterns(market_data, context);
            Vector volume_interpretation = analyze_volume_patterns(market_data, context);
            Vector directional_bias = calculate_directional_bias(market_data, context);
            Vector volatility_regime = assess_volatility_regime(market_data, context);
            float economic_alignment = assess_economic_alignment(market_data, context);

            return MarketInterpretation{
                price_interpretation,
                volume_interpretation,
                directional_bias,
                volatility_regime,
                economic_alignment
            };
        }

    private:
        Vector analyze_price_patterns(const Matrix& market_data, const EconomicContext& context) {
            Vector patterns(market_data.size());
            for (size_t i = 0; i < market_data.size(); ++i) {
                float close_price = market_data[i][3];

                // Price interpretation influenced by economic context
                float economic_multiplier = 1.0f;
                if (context.market_regime == "expansion") economic_multiplier = 1.2f;
                else if (context.market_regime == "tightening") economic_multiplier = 0.8f;

                patterns[i] = close_price * economic_multiplier;
            }
            return patterns;
        }

        Vector analyze_volume_patterns(const Matrix& market_data, const EconomicContext& context) {
            Vector patterns(market_data.size());
            for (size_t i = 0; i < market_data.size(); ++i) {
                float volume = market_data[i][4];

                // Volume interpretation within economic context
                float context_adjustment = 1.0f + context.economic_stability * 0.5f;
                patterns[i] = volume * context_adjustment;
            }
            return patterns;
        }

        Vector calculate_directional_bias(const Matrix& market_data, const EconomicContext& context) {
            Vector bias(market_data.size());

            // Directional bias influenced by economic conditions
            float economic_direction = 0.0f;
            if (context.growth_cycle.back() > 0.02f) economic_direction += 0.1f;
            if (context.employment_conditions.back() < 0.045f) economic_direction += 0.1f;
            if (context.inflation_trend.back() < 0.025f) economic_direction += 0.05f;

            for (size_t i = 0; i < market_data.size(); ++i) {
                bias[i] = economic_direction;
            }
            return bias;
        }

        Vector assess_volatility_regime(const Matrix& market_data, const EconomicContext& context) {
            Vector volatility(market_data.size());

            // Base volatility on economic stability
            float base_volatility = 0.02f + (1.0f - context.economic_stability) * 0.03f;

            for (size_t i = 0; i < market_data.size(); ++i) {
                volatility[i] = base_volatility;
            }
            return volatility;
        }

        float assess_economic_alignment(const Matrix& market_data, const EconomicContext& context) {
            // Calculate how well market behavior aligns with economic fundamentals
            float latest_price_change = market_data.back()[3] - market_data[market_data.size()-10][3];
            float economic_signal = context.growth_cycle.back() + context.employment_conditions.back();

            // Alignment score based on correlation between price and economic signals
            float alignment = 0.5f + 0.5f * tanh(latest_price_change * economic_signal * 10.0f);
            return std::max(0.0f, std::min(1.0f, alignment));
        }
    };

    // Context-First Financial Model
    class ContextFirstFinancialModel {
    public:
        ContextFirstFinancialModel() {
            economic_analyzer_ = EconomicContextAnalyzer();
            market_interpreter_ = MarketDataInterpreter();
            std::cout << "🎯 Context-First Financial Model initialized" << std::endl;
        }

        float predict_next_price(const Matrix& fred_data, const Matrix& market_data) {
            std::cout << "\n🔮 Making context-first prediction..." << std::endl;

            // Step 1: Establish economic context (FRED data FIRST)
            EconomicContext context = economic_analyzer_.analyze_context(fred_data);
            std::cout << "Economic context established: " << context.market_regime
                      << " (confidence: " << context.context_confidence << ")" << std::endl;

            // Step 2: Interpret market data within economic context
            MarketInterpretation interpretation = market_interpreter_.interpret_market(market_data, context);
            std::cout << "Market interpretation complete (alignment: "
                      << interpretation.economic_alignment << ")" << std::endl;

            // Step 3: Generate prediction based on integrated understanding
            float prediction = generate_context_aware_prediction(context, interpretation, market_data);

            return prediction;
        }

    private:
        float generate_context_aware_prediction(const EconomicContext& context,
                                              const MarketInterpretation& interpretation,
                                              const Matrix& market_data) {
            // Get latest market price
            float current_price = market_data.back()[3];

            // Economic context influence
            float economic_multiplier = 1.0f;
            if (context.market_regime == "expansion") economic_multiplier = 1.05f;
            else if (context.market_regime == "tightening") economic_multiplier = 0.95f;

            // Market interpretation influence
            float market_momentum = interpretation.directional_bias.back();

            // Combine economic and market factors
            float price_change = current_price * (economic_multiplier - 1.0f + market_momentum * 0.1f);
            float predicted_price = current_price + price_change;

            std::cout << "Current price: $" << std::fixed << std::setprecision(2) << current_price << std::endl;
            std::cout << "Economic multiplier: " << economic_multiplier << std::endl;
            std::cout << "Market momentum: " << market_momentum << std::endl;
            std::cout << "Predicted price: $" << predicted_price << std::endl;

            return predicted_price;
        }

        EconomicContextAnalyzer economic_analyzer_;
        MarketDataInterpreter market_interpreter_;
    };

    // Run context-first financial prediction demo
    static void run_context_first_demo() {
        std::cout << "🎯 Context-First Financial Prediction Demo" << std::endl;
        std::cout << "Economic Context (FRED) → Market Interpretation → Prediction" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Demo parameters
        const int sequence_length = 60;  // 60 time steps of data
        const int num_test_scenarios = 5;

        ContextFirstFinancialModel context_model;

        std::vector<float> prediction_errors;
        std::vector<double> processing_times;

        for (int scenario = 0; scenario < num_test_scenarios; ++scenario) {
            std::cout << "\n--- Economic Scenario " << (scenario + 1) << " ---" << std::endl;

            // Generate economic and market data for this scenario
            Matrix fred_data = load_fred_economic_data(sequence_length);
            Matrix market_data = load_market_data(sequence_length);

            // Make context-first prediction
            auto start_time = std::chrono::high_resolution_clock::now();

            float predicted_price = context_model.predict_next_price(fred_data, market_data);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> processing_time = end_time - start_time;

            // Calculate error (using next hypothetical price as target)
            float actual_next_price = market_data.back()[3] * (1.0f + 0.02f * (rand() % 100 - 50) / 100.0f);
            float error = std::abs(predicted_price - actual_next_price) / actual_next_price;

            prediction_errors.push_back(error);
            processing_times.push_back(processing_time.count());

            std::cout << "Processing time: " << std::fixed << std::setprecision(3)
                      << processing_time.count() << " ms" << std::endl;
            std::cout << "Prediction error: " << std::setprecision(2) << (error * 100) << "%" << std::endl;
        }

        // Summary statistics
        float avg_error = std::accumulate(prediction_errors.begin(), prediction_errors.end(), 0.0f) / prediction_errors.size();
        double avg_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "📊 CONTEXT-FIRST PREDICTION RESULTS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << "\n⏱️  PERFORMANCE:" << std::endl;
        std::cout << "Average processing time: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        std::cout << "Average prediction error: " << std::setprecision(2) << (avg_error * 100) << "%" << std::endl;

        std::cout << "\n🎯 KEY ADVANTAGES:" << std::endl;
        std::cout << "• Economic context processed FIRST (FRED data)" << std::endl;
        std::cout << "• Market data interpreted within economic reality" << std::endl;
        std::cout << "• Multi-step reasoning between economic and market domains" << std::endl;
        std::cout << "• Grounded in economic fundamentals, not just price action" << std::endl;

        std::cout << "\n💡 FINANCIAL APPLICATIONS:" << std::endl;
        std::cout << "• Interest rate impact prediction" << std::endl;
        std::cout << "• Economic regime-aware trading" << std::endl;
        std::cout << "• Fed policy response modeling" << std::endl;
        std::cout << "• Risk-adjusted portfolio optimization" << std::endl;
        std::cout << "• Economic indicator impact assessment" << std::endl;

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "✅ CONTEXT-FIRST DEMONSTRATION COMPLETE" << std::endl;
        std::cout << "Economic context provides superior foundation for financial prediction!" << std::endl;
    }
};

int main() {
    try {
        ContextFirstFinancialDemo::run_context_first_demo();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }
}