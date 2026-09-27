#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include "dataset_prep.h"

namespace tok {

class MMapDataLoader {
public:
    MMapDataLoader();
    ~MMapDataLoader();

    // Opens a compiled binary dataset file (train.bin / val.bin) and verifies header.
    void open(const std::string& binary_file_path);
    void close();

    const DatasetHeader& header() const { return header_; }
    uint64_t total_tokens() const { return header_.total_tokens; }

    // Slices a sequence batch of token IDs starting at token_offset.
    std::vector<uint16_t> get_batch(uint64_t token_offset, size_t sequence_length) const;

private:
    DatasetHeader header_;
    std::vector<uint16_t> tokens_data_; // Packed binary payload view
    bool is_open_ = false;
};

} // namespace tok
