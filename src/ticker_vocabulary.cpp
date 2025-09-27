#include "ticker_vocabulary.h"
#include <fstream>
#include <algorithm>
#include <iostream>

namespace tokenizer {

std::vector<std::string> TickerVocabulary::tickers_;
std::unordered_set<std::string> TickerVocabulary::ticker_set_;
bool TickerVocabulary::initialized_ = false;

void TickerVocabulary::initialize() {
    if (initialized_) return;
    
    // Default path to ticker file
    std::string ticker_file = "../data/all_tickers.txt";
    loadFromFile(ticker_file);
    
    initialized_ = true;
}

void TickerVocabulary::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open ticker file: " << filepath << std::endl;
        return;
    }
    
    std::string ticker;
    tickers_.clear();
    ticker_set_.clear();
    
    while (std::getline(file, ticker)) {
        if (!ticker.empty()) {
            // Normalize to uppercase
            std::transform(ticker.begin(), ticker.end(), ticker.begin(), ::toupper);
            tickers_.push_back(ticker);
            ticker_set_.insert(ticker);
            
            // Also add with $ prefix (common in financial text)
            std::string dollar_ticker = "$" + ticker;
            tickers_.push_back(dollar_ticker);
            ticker_set_.insert(dollar_ticker);
        }
    }
    
    file.close();
    
    std::cout << "Loaded " << tickers_.size() / 2 << " tickers (" 
              << tickers_.size() << " total with $ variants)" << std::endl;
}

const std::vector<std::string>& TickerVocabulary::getTickers() {
    if (!initialized_) {
        initialize();
    }
    return tickers_;
}

bool TickerVocabulary::isTicker(const std::string& token) {
    if (!initialized_) {
        initialize();
    }
    
    // Check both uppercase and with $ prefix
    std::string upper_token = token;
    std::transform(upper_token.begin(), upper_token.end(), upper_token.begin(), ::toupper);
    
    return ticker_set_.find(upper_token) != ticker_set_.end();
}

} // namespace tokenizer