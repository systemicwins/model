#ifndef TICKER_VOCABULARY_H
#define TICKER_VOCABULARY_H

#include <vector>
#include <string>
#include <unordered_set>

namespace tokenizer {

// Stock ticker vocabulary extension
// This adds 6,700+ tradeable ticker symbols to the tokenizer vocabulary
class TickerVocabulary {
public:
    static const std::vector<std::string>& getTickers();
    static bool isTicker(const std::string& token);
    static void loadFromFile(const std::string& filepath);
    
private:
    static std::vector<std::string> tickers_;
    static std::unordered_set<std::string> ticker_set_;
    static bool initialized_;
    
    static void initialize();
};

// Updated vocabulary size to include tickers
constexpr size_t EXPANDED_VOCAB_SIZE = 56700;  // 50,000 base + 6,700 tickers

} // namespace tokenizer

#endif // TICKER_VOCABULARY_H