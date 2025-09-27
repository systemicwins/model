#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <regex>

class Tokenizer {
public:
    // Special token IDs
    static constexpr int PAD_TOKEN_ID = 0;
    static constexpr int UNK_TOKEN_ID = 1;
    static constexpr int BOS_TOKEN_ID = 2;
    static constexpr int EOS_TOKEN_ID = 3;
    static constexpr int CLS_TOKEN_ID = 4;
    static constexpr int SEP_TOKEN_ID = 5;
    static constexpr int MASK_TOKEN_ID = 6;
    
    // Special tokens
    static constexpr const char* PAD_TOKEN = "<pad>";
    static constexpr const char* UNK_TOKEN = "<unk>";
    static constexpr const char* BOS_TOKEN = "<bos>";
    static constexpr const char* EOS_TOKEN = "<eos>";
    static constexpr const char* CLS_TOKEN = "<cls>";
    static constexpr const char* SEP_TOKEN = "<sep>";
    static constexpr const char* MASK_TOKEN = "<mask>";

    struct TokenizerConfig {
        int vocab_size;
        int max_length;
        bool add_bos_token;
        bool add_eos_token;
        bool lowercase;
        std::string vocab_file;
        
        TokenizerConfig() 
            : vocab_size(50000),
              max_length(8191),
              add_bos_token(true),
              add_eos_token(true),
              lowercase(false),
              vocab_file("") {}
    };

    Tokenizer(const TokenizerConfig& config = TokenizerConfig());
    ~Tokenizer();

    // Main tokenization functions
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true);
    std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = false);
    
    // Batch processing
    std::vector<std::vector<int>> encode_batch(const std::vector<std::string>& texts, 
                                                bool add_special_tokens = true,
                                                bool padding = true);
    
    // Convert tokens to embeddings (returns vocab_size dimensional one-hot or learned embeddings)
    std::vector<std::vector<float>> tokens_to_embeddings(const std::vector<int>& token_ids);
    std::vector<std::vector<float>> tokens_to_embeddings_batch(const std::vector<std::vector<int>>& token_ids_batch);
    
    // Utility functions
    int get_vocab_size() const { return vocab_size_; }
    int get_embedding_dim() const { return embedding_dim_; }
    bool has_token(const std::string& token) const;
    int token_to_id(const std::string& token) const;
    std::string id_to_token(int id) const;
    
    // Padding and truncation
    std::vector<int> pad_sequence(const std::vector<int>& sequence, int target_length);
    std::vector<int> truncate_sequence(const std::vector<int>& sequence, int max_length);
    
    // Load/save vocabulary
    bool load_vocab(const std::string& filepath);
    bool save_vocab(const std::string& filepath) const;
    
    // Load additional symbols (e.g., stock tickers)
    int load_additional_symbols(const std::string& filepath);
    
    // Initialize embedding matrix (random or from file)
    void initialize_embeddings(int embedding_dim, bool random = true);
    bool load_embeddings(const std::string& filepath);

private:
    TokenizerConfig config_;
    int vocab_size_;
    int embedding_dim_;
    
    // Vocabulary mappings
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    
    // BPE merge rules (for subword tokenization)
    std::vector<std::pair<std::string, std::string>> bpe_merges_;
    std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
    
    // Embedding matrix (vocab_size x embedding_dim)
    std::vector<std::vector<float>> embeddings_;
    
    // Internal tokenization methods
    std::vector<std::string> basic_tokenize(const std::string& text);
    std::vector<std::string> wordpiece_tokenize(const std::string& word);
    std::vector<std::string> bpe_tokenize(const std::string& text);
    std::string apply_bpe(const std::string& word);
    
    // Helper functions
    void build_default_vocab();
    void initialize_special_tokens();
    std::string normalize_text(const std::string& text);
    std::vector<std::string> split_on_whitespace(const std::string& text);
    std::vector<std::string> split_on_punctuation(const std::string& text);
};

#endif // TOKENIZER_H