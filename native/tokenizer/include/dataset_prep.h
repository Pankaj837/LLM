#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "tokenizer.h"

namespace tok {

#pragma pack(push, 1)
struct DatasetHeader {
    uint32_t magic = 0x544F4B31;   // "TOK1" magic signature
    uint32_t version = 1;          // Format version 1
    uint32_t vocab_size = 0;       // Vocabulary size
    uint32_t eos_token_id = 0;     // EOS Token ID
    uint64_t total_tokens = 0;     // Total token count in dataset
    uint32_t split_type = 0;       // 0 = train, 1 = val
    uint64_t reserved = 0;         // Alignment padding (32 bytes header total)
};
#pragma pack(pop)

struct BinaryCompileResult {
    uint64_t train_tokens = 0;
    uint64_t val_tokens = 0;
    std::string train_bin_path;
    std::string val_bin_path;
};

// Cleans input documents, subsamples tokens up to max_target_tokens, appends
// EOS after each document, and compiles binary dataset files train.bin and val.bin.
BinaryCompileResult compile_dataset_to_binary(
    const std::vector<std::string>& raw_documents,
    const Tokenizer& tokenizer,
    const std::string& train_bin_path,
    const std::string& val_bin_path,
    double val_ratio = 0.05,
    uint64_t max_target_tokens = 250000000ULL,
    const std::string& eos_token = "<|endoftext|>");

} // namespace tok
