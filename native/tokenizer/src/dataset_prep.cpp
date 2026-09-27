#include "dataset_prep.h"
#include "utils.h"

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace tok {

BinaryCompileResult compile_dataset_to_binary(
    const std::vector<std::string>& raw_documents,
    const Tokenizer& tokenizer,
    const std::string& train_bin_path,
    const std::string& val_bin_path,
    double val_ratio,
    uint64_t max_target_tokens,
    const std::string& eos_token) {

    int eos_id = -1;
    if (tokenizer.vocabulary().has_special(eos_token)) {
        eos_id = tokenizer.vocabulary().special_id(eos_token);
    } else {
        throw std::runtime_error("EOS token not found in tokenizer vocabulary: " + eos_token);
    }

    std::vector<uint16_t> train_tokens;
    std::vector<uint16_t> val_tokens;

    uint64_t current_total = 0;

    for (size_t doc_idx = 0; doc_idx < raw_documents.size(); ++doc_idx) {
        if (current_total >= max_target_tokens) {
            break; // Token target budget reached
        }

        std::string cleaned = sanitize_utf8_and_normalize(raw_documents[doc_idx]);
        if (cleaned.empty()) continue;

        auto ids = tokenizer.encode(cleaned);
        ids.push_back(eos_id); // Explicitly append EOS after document

        bool is_val = (val_ratio > 0.0) && ((doc_idx % static_cast<size_t>(1.0 / val_ratio)) == 0);

        uint64_t prev_check = current_total;
        for (int id : ids) {
            if (current_total >= max_target_tokens) break;

            uint16_t u16_id = static_cast<uint16_t>(id & 0xFFFF);
            if (is_val) {
                val_tokens.push_back(u16_id);
            } else {
                train_tokens.push_back(u16_id);
            }
            current_total++;
        }

        if ((current_total / 500000) > (prev_check / 500000)) {
            std::cout << "   [Subsampling Progress] Processed " << current_total << " / " << max_target_tokens << " tokens..." << std::endl;
        }
    }

    // Write train.bin
    std::ofstream train_f(train_bin_path, std::ios::binary);
    if (!train_f) {
        throw std::runtime_error("Could not open output train.bin file: " + train_bin_path);
    }
    DatasetHeader train_header;
    train_header.vocab_size = static_cast<uint32_t>(tokenizer.vocab_size());
    train_header.eos_token_id = static_cast<uint32_t>(eos_id);
    train_header.total_tokens = train_tokens.size();
    train_header.split_type = 0;

    train_f.write(reinterpret_cast<const char*>(&train_header), sizeof(DatasetHeader));
    if (!train_tokens.empty()) {
        train_f.write(reinterpret_cast<const char*>(train_tokens.data()),
                      static_cast<std::streamsize>(train_tokens.size() * sizeof(uint16_t)));
    }
    train_f.close();

    // Write val.bin
    std::ofstream val_f(val_bin_path, std::ios::binary);
    if (!val_f) {
        throw std::runtime_error("Could not open output val.bin file: " + val_bin_path);
    }
    DatasetHeader val_header;
    val_header.vocab_size = static_cast<uint32_t>(tokenizer.vocab_size());
    val_header.eos_token_id = static_cast<uint32_t>(eos_id);
    val_header.total_tokens = val_tokens.size();
    val_header.split_type = 1;

    val_f.write(reinterpret_cast<const char*>(&val_header), sizeof(DatasetHeader));
    if (!val_tokens.empty()) {
        val_f.write(reinterpret_cast<const char*>(val_tokens.data()),
                    static_cast<std::streamsize>(val_tokens.size() * sizeof(uint16_t)));
    }
    val_f.close();

    BinaryCompileResult result;
    result.train_tokens = train_tokens.size();
    result.val_tokens = val_tokens.size();
    result.train_bin_path = train_bin_path;
    result.val_bin_path = val_bin_path;

    return result;
}

} // namespace tok
