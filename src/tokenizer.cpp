#include "../include/tokenizer.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

Tokenizer::Tokenizer(const TokenizerConfig& config) 
    : config_(config), vocab_size_(config.vocab_size), embedding_dim_(1536) {
    
    initialize_special_tokens();
    
    if (!config.vocab_file.empty()) {
        if (!load_vocab(config.vocab_file)) {
            std::cerr << "Warning: Failed to load vocabulary from " << config.vocab_file 
                      << ". Using default vocabulary.\n";
            build_default_vocab();
        }
    } else {
        build_default_vocab();
    }
    
    // Initialize embeddings with default dimension
    initialize_embeddings(embedding_dim_);
    
    // Try to load trading symbols if file exists
    std::string symbols_file = "/Users/alex/relentless/model/data/trading_symbols.txt";
    std::ifstream test_file(symbols_file);
    if (test_file.good()) {
        test_file.close();
        int loaded = load_additional_symbols(symbols_file);
        if (loaded > 0) {
            std::cout << "Loaded " << loaded << " trading symbols into vocabulary\n";
        }
    }
    
    // Try to load SEC filing vocabulary if file exists
    std::string sec_vocab_file = "/Users/alex/relentless/model/data/sec_filing_vocab.txt";
    std::ifstream sec_test_file(sec_vocab_file);
    if (sec_test_file.good()) {
        sec_test_file.close();
        int loaded = load_additional_symbols(sec_vocab_file);
        if (loaded > 0) {
            std::cout << "Loaded " << loaded << " SEC filing terms into vocabulary\n";
        }
    }
}

Tokenizer::~Tokenizer() = default;

void Tokenizer::initialize_special_tokens() {
    // Add special tokens to vocabulary
    token_to_id_[PAD_TOKEN] = PAD_TOKEN_ID;
    token_to_id_[UNK_TOKEN] = UNK_TOKEN_ID;
    token_to_id_[BOS_TOKEN] = BOS_TOKEN_ID;
    token_to_id_[EOS_TOKEN] = EOS_TOKEN_ID;
    token_to_id_[CLS_TOKEN] = CLS_TOKEN_ID;
    token_to_id_[SEP_TOKEN] = SEP_TOKEN_ID;
    token_to_id_[MASK_TOKEN] = MASK_TOKEN_ID;
    
    id_to_token_[PAD_TOKEN_ID] = PAD_TOKEN;
    id_to_token_[UNK_TOKEN_ID] = UNK_TOKEN;
    id_to_token_[BOS_TOKEN_ID] = BOS_TOKEN;
    id_to_token_[EOS_TOKEN_ID] = EOS_TOKEN;
    id_to_token_[CLS_TOKEN_ID] = CLS_TOKEN;
    id_to_token_[SEP_TOKEN_ID] = SEP_TOKEN;
    id_to_token_[MASK_TOKEN_ID] = MASK_TOKEN;
}

void Tokenizer::build_default_vocab() {
    // Start with special tokens (already added)
    int current_id = 7;  // After special tokens
    
    // Add common single characters
    for (char c = ' '; c <= '~'; ++c) {
        std::string token(1, c);
        if (token_to_id_.find(token) == token_to_id_.end()) {
            token_to_id_[token] = current_id;
            id_to_token_[current_id] = token;
            current_id++;
        }
    }
    
    // Add common subwords and words
    std::vector<std::string> common_tokens = {
        // Common programming tokens
        "def", "class", "function", "return", "if", "else", "for", "while",
        "import", "from", "const", "let", "var", "public", "private", "static",
        "void", "int", "float", "string", "bool", "true", "false", "null",
        "self", "this", "new", "delete", "try", "catch", "finally", "throw",
        
        // Financial Market Terms
        "market", "stock", "bond", "equity", "asset", "security", "derivative",
        "option", "future", "forward", "swap", "commodity", "currency", "forex",
        "exchange", "NYSE", "NASDAQ", "SP500", "S&P500", "DOW", "DJIA", "FTSE",
        "DAX", "Nikkei", "HSI", "SSE", "SZSE", "TSX", "ASX", "LSE",
        
        // Trading Terms
        "buy", "sell", "bid", "ask", "spread", "volume", "liquidity", "volatility",
        "trade", "trading", "trader", "order", "limit", "market", "stop", "fill",
        "execution", "settlement", "clearing", "broker", "dealer", "maker", "taker",
        "long", "short", "position", "leverage", "margin", "collateral", "hedge",
        
        // Financial Instruments
        "share", "dividend", "yield", "coupon", "maturity", "duration", "convexity",
        "ETF", "ETN", "mutual", "fund", "index", "benchmark", "portfolio", "allocation",
        "call", "put", "strike", "expiration", "exercise", "assignment", "premium",
        "underlying", "delta", "gamma", "theta", "vega", "rho", "implied", "historical",
        
        // Price and Valuation
        "price", "value", "valuation", "quote", "tick", "pip", "basis", "point",
        "open", "high", "low", "close", "OHLC", "VWAP", "TWAP", "midpoint",
        "fair", "intrinsic", "market", "book", "NAV", "AUM", "P/E", "P/B",
        "EPS", "PE", "PB", "PS", "PEG", "EV", "EBITDA", "DCF", "NPV", "IRR",
        
        // Risk and Analytics
        "risk", "return", "alpha", "beta", "sharpe", "sortino", "treynor", "jensen",
        "correlation", "covariance", "variance", "deviation", "standard", "VaR", "CVaR",
        "drawdown", "maximum", "recovery", "downside", "upside", "systematic", "idiosyncratic",
        "specific", "factor", "exposure", "sensitivity", "stress", "scenario", "backtest",
        
        // Corporate Finance
        "revenue", "profit", "loss", "income", "expense", "earnings", "EBIT", "EBITDA",
        "cash", "flow", "FCF", "working", "capital", "debt", "equity", "leverage",
        "ratio", "margin", "gross", "operating", "net", "ROE", "ROA", "ROI", "ROIC",
        "balance", "sheet", "statement", "financial", "quarterly", "annual", "10K", "10Q",
        
        // Market Structure
        "exchange", "OTC", "dark", "pool", "ECN", "ATS", "SIP", "NBBO", "BBO",
        "maker", "taker", "rebate", "fee", "commission", "slippage", "impact",
        "HFT", "algo", "algorithmic", "quant", "quantitative", "systematic", "discretionary",
        "arbitrage", "statistical", "pairs", "momentum", "reversal", "trend", "breakout",
        
        // Economic Indicators
        "GDP", "CPI", "PPI", "PCE", "unemployment", "payroll", "NFP", "jobless",
        "inflation", "deflation", "stagflation", "recession", "expansion", "cycle",
        "Fed", "FOMC", "ECB", "BOJ", "BOE", "PBOC", "central", "bank", "policy",
        "rate", "interest", "discount", "prime", "LIBOR", "SOFR", "EFFR", "repo",
        
        // Technical Analysis
        "support", "resistance", "trendline", "channel", "breakout", "breakdown",
        "reversal", "continuation", "pattern", "flag", "pennant", "wedge", "triangle",
        "head", "shoulders", "double", "triple", "top", "bottom", "cup", "handle",
        "MA", "SMA", "EMA", "WMA", "MACD", "RSI", "stochastic", "bollinger", "band",
        "fibonacci", "retracement", "extension", "pivot", "candlestick", "doji", "hammer",
        
        // Options Greeks and Strategies
        "straddle", "strangle", "spread", "butterfly", "condor", "iron", "collar",
        "covered", "naked", "protective", "synthetic", "combo", "roll", "calendar",
        "diagonal", "vertical", "horizontal", "ratio", "backspread", "risk", "reversal",
        
        // Crypto and Digital Assets
        "crypto", "cryptocurrency", "bitcoin", "BTC", "ethereum", "ETH", "blockchain",
        "DeFi", "NFT", "token", "coin", "altcoin", "stablecoin", "USDT", "USDC",
        "mining", "staking", "yield", "farming", "liquidity", "pool", "AMM", "DEX",
        "wallet", "address", "hash", "block", "chain", "ledger", "consensus", "proof",
        
        // Regulatory and Compliance
        "SEC", "CFTC", "FINRA", "regulation", "compliance", "audit", "filing",
        "disclosure", "insider", "material", "nonpublic", "MNPI", "restricted",
        "blackout", "window", "quiet", "period", "prospectus", "registration",
        
        // SEC Filing Types and Forms
        "10-K", "10-Q", "8-K", "DEF14A", "S-1", "S-3", "S-4", "S-8", "424B",
        "424B2", "424B3", "424B4", "424B5", "DEFM14A", "PREM14A", "PRE14A",
        "Form4", "Form5", "Schedule13D", "Schedule13G", "13F", "Schedule14A",
        "20-F", "40-F", "6-K", "F-1", "F-3", "F-4", "N-CSR", "N-Q", "485APOS",
        "485BPOS", "497", "497K", "NT10-K", "NT10-Q", "SC13D", "SC13G", "144",
        
        // SEC Filing Sections and Items
        "Item1", "Item1A", "Item1B", "Item2", "Item3", "Item4", "Item5", "Item6",
        "Item7", "Item7A", "Item8", "Item9", "Item9A", "Item9B", "Item10", "Item11",
        "Item12", "Item13", "Item14", "Item15", "MD&A", "Management's", "Discussion",
        "Analysis", "Risk", "Factors", "Legal", "Proceedings", "Controls", "Procedures",
        
        // Financial Statement Components
        "Consolidated", "Statements", "Operations", "Comprehensive", "Stockholders",
        "Shareholders", "Cash", "Flows", "Notes", "Unaudited", "Audited", "Interim",
        "Restated", "Restatement", "As-Reported", "Pro-Forma", "Non-GAAP", "GAAP",
        "Adjusted", "Normalized", "Recurring", "Non-Recurring", "One-Time",
        
        // Accounting Terms in Filings
        "Goodwill", "Intangible", "Tangible", "Depreciation", "Amortization",
        "Impairment", "Write-down", "Write-off", "Provision", "Allowance", "Reserve",
        "Accrual", "Deferred", "Prepaid", "Accumulated", "Retained", "Treasury",
        "Common", "Preferred", "Par", "Additional", "Paid-In", "Contributed",
        "Surplus", "Deficit", "Unrealized", "Realized", "Gain", "Loss",
        
        // Revenue Recognition Terms
        "ASC606", "ASC842", "ASU", "FASB", "PCAOB", "SOX", "Sarbanes-Oxley",
        "Revenue", "Recognition", "Performance", "Obligation", "Contract", "Customer",
        "Transaction", "Price", "Allocation", "Satisfaction", "Point-in-Time",
        "Over-Time", "Deferred", "Unbilled", "Backlog", "Bookings", "Billings",
        
        // Corporate Governance Terms
        "Board", "Directors", "Executive", "Officers", "Compensation", "Committee",
        "Audit", "Nominating", "Governance", "Independent", "Chairman", "CEO",
        "CFO", "COO", "President", "Secretary", "Treasurer", "Controller",
        "Principal", "Accounting", "Officer", "Named", "NEO", "Proxy", "Solicitation",
        
        // Compensation and Benefits
        "Base", "Salary", "Bonus", "Incentive", "Stock-Based", "Options", "RSUs",
        "PSUs", "Restricted", "Units", "Performance", "Vesting", "Cliff", "Graded",
        "Forfeiture", "Clawback", "Change-in-Control", "CIC", "Severance", "Perquisites",
        "Pension", "401k", "Deferred", "Compensation", "SERP", "Rabbi", "Trust",
        
        // M&A and Corporate Actions
        "Merger", "Acquisition", "Combination", "Reorganization", "Restructuring",
        "Divestiture", "Disposition", "Spin-off", "Split-off", "Carve-out",
        "Joint", "Venture", "Strategic", "Alliance", "Partnership", "Collaboration",
        "License", "Royalty", "Earnout", "Escrow", "Indemnification", "Representation",
        "Warranty", "Covenant", "Condition", "Precedent", "Subsequent", "Closing",
        
        // Legal and Litigation Terms
        "Litigation", "Lawsuit", "Claim", "Dispute", "Proceeding", "Investigation",
        "Subpoena", "Settlement", "Judgment", "Verdict", "Appeal", "Arbitration",
        "Mediation", "Class", "Action", "Derivative", "Plaintiff", "Defendant",
        "Damages", "Injunction", "Remedy", "Liability", "Contingency", "Probable",
        "Reasonably", "Possible", "Remote", "Estimable", "Accrued", "Disclosed",
        
        // Risk Disclosure Language
        "May", "Could", "Might", "Should", "Would", "Expect", "Anticipate",
        "Believe", "Estimate", "Intend", "Plan", "Predict", "Potential", "Seek",
        "Forward-Looking", "Statements", "Safe", "Harbor", "Cautionary", "Statement",
        "Uncertainties", "Assumptions", "Projections", "Outlook", "Guidance",
        
        // Debt and Financing Terms
        "Indebtedness", "Borrowings", "Credit", "Facility", "Revolver", "Term",
        "Loan", "Notes", "Bonds", "Debentures", "Senior", "Subordinated", "Secured",
        "Unsecured", "Guarantee", "Guarantor", "Indenture", "Trustee", "Covenant",
        "Default", "Event", "Acceleration", "Prepayment", "Redemption", "Refinancing",
        "Maturity", "Principal", "Interest", "LIBOR", "SOFR", "Base", "Rate",
        
        // Segment and Geographic Reporting
        "Segment", "Operating", "Reportable", "Geographic", "Domestic", "Foreign",
        "International", "Americas", "EMEA", "APAC", "Constant", "Currency",
        "Organic", "Inorganic", "Same-Store", "Comparable", "Like-for-Like",
        
        // Tax Terms
        "Income", "Tax", "Provision", "Benefit", "Current", "Deferred", "Asset",
        "Liability", "Valuation", "Allowance", "NOL", "Carryforward", "Carryback",
        "Effective", "Rate", "ETR", "Statutory", "Permanent", "Temporary", "Difference",
        "Uncertain", "Position", "FIN48", "ASC740", "Transfer", "Pricing", "Repatriation",
        
        // Related Party Terms
        "Related", "Party", "Affiliate", "Subsidiary", "Parent", "Holding",
        "Wholly-Owned", "Majority-Owned", "Minority", "Interest", "Equity", "Method",
        "Consolidation", "VIE", "Variable", "Interest", "Entity", "Control",
        "Significant", "Influence", "Joint", "Venture", "Associate",
        
        // Internal Controls Language
        "Internal", "Control", "Financial", "Reporting", "ICFR", "Disclosure",
        "Controls", "DC&P", "Material", "Weakness", "Significant", "Deficiency",
        "Remediation", "Management", "Assessment", "Evaluation", "Effective",
        "Ineffective", "COSO", "Framework", "Design", "Operating", "Effectiveness",
        
        // Auditor Terms
        "Independent", "Registered", "Public", "Accounting", "Firm", "Auditor",
        "Opinion", "Unqualified", "Qualified", "Adverse", "Disclaimer", "Going",
        "Concern", "Emphasis", "Matter", "Critical", "Audit", "Matters", "CAM",
        "PCAOB", "Standards", "GAAS", "Reasonable", "Assurance", "Misstatement",
        
        // Industry-Specific SEC Terms
        "FDA", "Clinical", "Trial", "Phase", "Approval", "Pipeline", "Patent",
        "Intellectual", "Property", "Regulatory", "Milestone", "Royalty", "License",
        "Biosimilar", "Generic", "Branded", "Formulary", "Reimbursement", "Medicare",
        "Medicaid", "Commercial", "Payor", "PBM", "Rebate", "Discount", "Chargeback",
        
        // Environmental and Social Terms
        "ESG", "Environmental", "Social", "Governance", "Sustainability", "Climate",
        "Carbon", "Emissions", "Scope1", "Scope2", "Scope3", "GHG", "Renewable",
        "Energy", "Diversity", "Inclusion", "DEI", "Human", "Capital", "Safety",
        "OSHA", "Recordable", "Incident", "Rate", "TRIR", "Fatality",
        
        // Quantitative Disclosures
        "Millions", "Billions", "Thousands", "Percent", "Percentage", "Basis",
        "Points", "bps", "Year-over-Year", "YoY", "Quarter-over-Quarter", "QoQ",
        "Sequential", "Trailing", "Twelve", "Months", "TTM", "Run-Rate", "Annualized",
        "Normalized", "Adjusted", "Excluding", "Including", "Incremental", "Decremental",
        
        // Asset Classes
        "equity", "fixed", "income", "commodity", "real", "estate", "REIT",
        "alternative", "private", "public", "emerging", "developed", "frontier",
        "domestic", "international", "global", "sovereign", "corporate", "municipal",
        "treasury", "gilt", "bund", "JGB", "convertible", "preferred", "common",
        
        // Financial Metrics Abbreviations
        "YTD", "QTD", "MTD", "YoY", "QoQ", "MoM", "TTM", "LTM", "CAGR",
        "bps", "bp", "pct", "percent", "percentage", "annualized", "compounded",
        
        // Common subwords
        "##ing", "##ed", "##er", "##est", "##ly", "##tion", "##ness",
        "##ment", "##able", "##ful", "##less", "##ize", "##ise",
        
        // Common words
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        
        // Numbers as strings
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "100", "1000", "10000", "100000", "1000000",
        
        // Common bigrams
        "th", "he", "in", "er", "an", "re", "ed", "on", "es", "st",
        "en", "at", "to", "nt", "ha", "nd", "ou", "ea", "ng", "as",
        
        // Byte-level tokens for handling unknown characters
        "<0x00>", "<0x01>", "<0x02>", "<0x03>", "<0x04>", "<0x05>",
        "<0x06>", "<0x07>", "<0x08>", "<0x09>", "<0x0A>", "<0x0B>",
        "<0x0C>", "<0x0D>", "<0x0E>", "<0x0F>"
    };
    
    for (const auto& token : common_tokens) {
        if (current_id >= vocab_size_) break;
        if (token_to_id_.find(token) == token_to_id_.end()) {
            token_to_id_[token] = current_id;
            id_to_token_[current_id] = token;
            current_id++;
        }
    }
    
    // Fill remaining vocabulary with generic tokens
    while (current_id < vocab_size_) {
        std::string token = "<token_" + std::to_string(current_id) + ">";
        token_to_id_[token] = current_id;
        id_to_token_[current_id] = token;
        current_id++;
    }
}

void Tokenizer::initialize_embeddings(int embedding_dim, bool random) {
    embedding_dim_ = embedding_dim;
    embeddings_.resize(vocab_size_);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);  // Xavier initialization
    
    for (int i = 0; i < vocab_size_; ++i) {
        embeddings_[i].resize(embedding_dim_);
        
        if (i == PAD_TOKEN_ID) {
            // Padding token gets zero embedding
            std::fill(embeddings_[i].begin(), embeddings_[i].end(), 0.0f);
        } else if (random) {
            // Random initialization for other tokens
            for (int j = 0; j < embedding_dim_; ++j) {
                embeddings_[i][j] = dist(gen);
            }
        } else {
            // Deterministic initialization based on token ID
            for (int j = 0; j < embedding_dim_; ++j) {
                embeddings_[i][j] = std::sin((i * embedding_dim_ + j) * 0.01f) * 0.1f;
            }
        }
    }
}

std::string Tokenizer::normalize_text(const std::string& text) {
    std::string normalized = text;
    
    if (config_.lowercase) {
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    }
    
    // Replace multiple spaces with single space
    size_t pos = 0;
    while ((pos = normalized.find("  ", pos)) != std::string::npos) {
        normalized.replace(pos, 2, " ");
    }
    
    return normalized;
}

std::vector<std::string> Tokenizer::split_on_whitespace(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string token;
    
    while (stream >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<std::string> Tokenizer::split_on_punctuation(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : text) {
        if (std::ispunct(c) && c != '#' && c != '_') {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(std::string(1, c));
        } else {
            current_token += c;
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

std::vector<std::string> Tokenizer::basic_tokenize(const std::string& text) {
    std::string normalized = normalize_text(text);
    std::vector<std::string> words = split_on_whitespace(normalized);
    std::vector<std::string> tokens;
    
    for (const auto& word : words) {
        auto word_tokens = split_on_punctuation(word);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }
    
    return tokens;
}

std::vector<std::string> Tokenizer::wordpiece_tokenize(const std::string& word) {
    std::vector<std::string> tokens;
    std::string remaining = word;
    
    while (!remaining.empty()) {
        std::string longest_match;
        bool found = false;
        
        // Try to find the longest matching subword
        for (size_t len = remaining.length(); len > 0; --len) {
            std::string subword = remaining.substr(0, len);
            
            // Add ## prefix for subwords (except at the beginning)
            if (!tokens.empty() && len < remaining.length()) {
                subword = "##" + subword;
            }
            
            if (token_to_id_.find(subword) != token_to_id_.end()) {
                longest_match = subword;
                found = true;
                remaining = remaining.substr(len);
                break;
            }
        }
        
        if (!found) {
            // If no match found, use unknown token for the first character
            tokens.push_back(UNK_TOKEN);
            remaining = remaining.substr(1);
        } else {
            tokens.push_back(longest_match);
        }
    }
    
    return tokens;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_special_tokens) {
    std::vector<int> token_ids;
    
    if (add_special_tokens && config_.add_bos_token) {
        token_ids.push_back(BOS_TOKEN_ID);
    }
    
    // Basic tokenization
    std::vector<std::string> basic_tokens = basic_tokenize(text);
    
    // Wordpiece tokenization for each basic token
    for (const auto& token : basic_tokens) {
        std::vector<std::string> subword_tokens = wordpiece_tokenize(token);
        
        for (const auto& subword : subword_tokens) {
            auto it = token_to_id_.find(subword);
            if (it != token_to_id_.end()) {
                token_ids.push_back(it->second);
            } else {
                token_ids.push_back(UNK_TOKEN_ID);
            }
        }
    }
    
    if (add_special_tokens && config_.add_eos_token) {
        token_ids.push_back(EOS_TOKEN_ID);
    }
    
    // Truncate if necessary
    if (token_ids.size() > static_cast<size_t>(config_.max_length)) {
        token_ids = truncate_sequence(token_ids, config_.max_length);
    }
    
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids, bool skip_special_tokens) {
    std::string result;
    
    for (int id : token_ids) {
        if (skip_special_tokens && id <= MASK_TOKEN_ID) {
            continue;
        }
        
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            std::string token = it->second;
            
            // Remove ## prefix from subwords
            if (token.substr(0, 2) == "##") {
                result += token.substr(2);
            } else {
                if (!result.empty() && result.back() != ' ') {
                    result += " ";
                }
                result += token;
            }
        }
    }
    
    return result;
}

std::vector<std::vector<int>> Tokenizer::encode_batch(const std::vector<std::string>& texts,
                                                      bool add_special_tokens,
                                                      bool padding) {
    std::vector<std::vector<int>> batch_token_ids;
    size_t max_length = 0;
    
    // Encode all texts
    for (const auto& text : texts) {
        auto token_ids = encode(text, add_special_tokens);
        max_length = std::max(max_length, token_ids.size());
        batch_token_ids.push_back(token_ids);
    }
    
    // Apply padding if requested
    if (padding) {
        for (auto& token_ids : batch_token_ids) {
            token_ids = pad_sequence(token_ids, max_length);
        }
    }
    
    return batch_token_ids;
}

std::vector<std::vector<float>> Tokenizer::tokens_to_embeddings(const std::vector<int>& token_ids) {
    std::vector<std::vector<float>> result;
    result.reserve(token_ids.size());
    
    for (int id : token_ids) {
        if (id >= 0 && id < vocab_size_) {
            result.push_back(embeddings_[id]);
        } else {
            // Return UNK embedding for out-of-vocabulary IDs
            result.push_back(embeddings_[UNK_TOKEN_ID]);
        }
    }
    
    return result;
}

std::vector<std::vector<float>> Tokenizer::tokens_to_embeddings_batch(
    const std::vector<std::vector<int>>& token_ids_batch) {
    
    std::vector<std::vector<float>> all_embeddings;
    
    for (const auto& token_ids : token_ids_batch) {
        auto embeddings = tokens_to_embeddings(token_ids);
        all_embeddings.insert(all_embeddings.end(), embeddings.begin(), embeddings.end());
    }
    
    return all_embeddings;
}

std::vector<int> Tokenizer::pad_sequence(const std::vector<int>& sequence, int target_length) {
    std::vector<int> padded = sequence;
    
    while (padded.size() < static_cast<size_t>(target_length)) {
        padded.push_back(PAD_TOKEN_ID);
    }
    
    return padded;
}

std::vector<int> Tokenizer::truncate_sequence(const std::vector<int>& sequence, int max_length) {
    if (sequence.size() <= static_cast<size_t>(max_length)) {
        return sequence;
    }
    
    std::vector<int> truncated(sequence.begin(), sequence.begin() + max_length);
    
    // Ensure EOS token at the end if it was originally present
    if (config_.add_eos_token && 
        std::find(sequence.begin(), sequence.end(), EOS_TOKEN_ID) != sequence.end()) {
        truncated.back() = EOS_TOKEN_ID;
    }
    
    return truncated;
}

bool Tokenizer::has_token(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

int Tokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : UNK_TOKEN_ID;
}

std::string Tokenizer::id_to_token(int id) const {
    auto it = id_to_token_.find(id);
    return (it != id_to_token_.end()) ? it->second : UNK_TOKEN;
}

bool Tokenizer::load_vocab(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    token_to_id_.clear();
    id_to_token_.clear();
    
    std::string line;
    int id = 0;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            token_to_id_[line] = id;
            id_to_token_[id] = line;
            id++;
        }
    }
    
    vocab_size_ = id;
    return true;
}

bool Tokenizer::save_vocab(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    for (int i = 0; i < vocab_size_; ++i) {
        auto it = id_to_token_.find(i);
        if (it != id_to_token_.end()) {
            file << it->second << "\n";
        }
    }
    
    return true;
}

bool Tokenizer::load_embeddings(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&vocab_size_), sizeof(int));
    file.read(reinterpret_cast<char*>(&embedding_dim_), sizeof(int));
    
    // Read embeddings
    embeddings_.resize(vocab_size_);
    for (int i = 0; i < vocab_size_; ++i) {
        embeddings_[i].resize(embedding_dim_);
        file.read(reinterpret_cast<char*>(embeddings_[i].data()), 
                  embedding_dim_ * sizeof(float));
    }
    
    return true;
}
int Tokenizer::load_additional_symbols(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return 0;
    }
    
    std::string line;
    int added_count = 0;
    int current_id = token_to_id_.size();
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Trim whitespace
        size_t first = line.find_first_not_of(" \t\r\n");
        size_t last = line.find_last_not_of(" \t\r\n");
        if (first != std::string::npos) {
            line = line.substr(first, last - first + 1);
        }
        
        // Add symbol if not already in vocabulary and within size limit
        if (token_to_id_.find(line) == token_to_id_.end() && current_id < vocab_size_) {
            token_to_id_[line] = current_id;
            id_to_token_[current_id] = line;
            
            // Also add embedding for new token
            if (current_id < static_cast<int>(embeddings_.size())) {
                // Use existing embedding slot
            } else {
                // Add new embedding
                std::vector<float> embedding(embedding_dim_);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<float> dist(0.0f, 0.02f);
                for (int i = 0; i < embedding_dim_; ++i) {
                    embedding[i] = dist(gen);
                }
                embeddings_.push_back(embedding);
            }
            
            current_id++;
            added_count++;
        }
    }
    
    return added_count;
}
