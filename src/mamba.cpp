#include "mamba.h"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace mamba {

MambaModel::MambaModel(const MambaConfig& config)
    : config_(config) {
}

MambaModel::~MambaModel() = default;

transformer::Matrix MambaModel::forward(const transformer::Matrix& embeddings,
                                        const transformer::Matrix* /*mask*/) {
    // Minimal placeholder: pass-through
    return embeddings;
}

std::vector<float> MambaModel::encode(const std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) {
        return {};
    }

    const int seq_len = static_cast<int>(embeddings.size());
    const int embed_dim = static_cast<int>(embeddings.front().size());

    transformer::Matrix input(seq_len, embed_dim);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < embed_dim; ++j) {
            input(i, j) = embeddings[i][j];
        }
    }

    transformer::Matrix output = forward(input);

    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(output.rows()) * static_cast<size_t>(output.cols()));
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            flat.push_back(output(i, j));
        }
    }
    return flat;
}

transformer::Vector MambaModel::get_pooled_output(const transformer::Matrix& encoded,
                                                  const std::string& pooling_method) {
    if (encoded.rows() == 0) {
        return transformer::Vector();
    }

    if (pooling_method == "mean") {
        return encoded.colwise().mean();
    }
    if (pooling_method == "max") {
        return encoded.colwise().maxCoeff();
    }
    if (pooling_method == "first") {
        return encoded.row(0);
    }
    if (pooling_method == "last") {
        return encoded.row(encoded.rows() - 1);
    }
    // Default to mean pooling
    return encoded.colwise().mean();
}

transformer::Matrix MambaModel::get_embeddings_at_dimension(const transformer::Matrix& input,
                                                            int target_dim) {
    const int in_dim = input.cols();
    const int use_dim = std::min(in_dim, target_dim);

    transformer::Matrix out(input.rows(), use_dim);
    out = input.leftCols(use_dim);
    return out;
}

void MambaModel::save_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    // Serialize minimal config
    file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
    file.close();
}

void MambaModel::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }
    // Read minimal config
    MambaConfig loaded{};
    file.read(reinterpret_cast<char*>(&loaded), sizeof(loaded));
    file.close();
    // Keep current config_, but ensure dimensions align if needed
}

} // namespace mamba