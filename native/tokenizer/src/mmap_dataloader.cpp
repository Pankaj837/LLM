#include "mmap_dataloader.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace tok {

MMapDataLoader::MMapDataLoader() = default;
MMapDataLoader::~MMapDataLoader() {
    close();
}

void MMapDataLoader::open(const std::string& binary_file_path) {
    close();

    std::ifstream f(binary_file_path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Could not open binary dataset file: " + binary_file_path);
    }

    std::streamsize file_size = f.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(DatasetHeader))) {
        throw std::runtime_error("Binary file is smaller than minimum header size: " + binary_file_path);
    }

    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(&header_), sizeof(DatasetHeader));

    if (header_.magic != 0x544F4B31) {
        throw std::runtime_error("Invalid dataset magic number in header: " + binary_file_path);
    }

    size_t expected_payload_bytes = header_.total_tokens * sizeof(uint16_t);
    size_t actual_payload_bytes = static_cast<size_t>(file_size) - sizeof(DatasetHeader);
    if (actual_payload_bytes < expected_payload_bytes) {
        throw std::runtime_error("Binary dataset file is truncated or corrupted: " + binary_file_path);
    }

    tokens_data_.resize(header_.total_tokens);
    if (header_.total_tokens > 0) {
        f.read(reinterpret_cast<char*>(tokens_data_.data()), static_cast<std::streamsize>(expected_payload_bytes));
    }

    is_open_ = true;
}

void MMapDataLoader::close() {
    is_open_ = false;
    tokens_data_.clear();
    header_ = DatasetHeader{};
}

std::vector<uint16_t> MMapDataLoader::get_batch(uint64_t token_offset, size_t sequence_length) const {
    if (!is_open_) {
        throw std::runtime_error("MMapDataLoader is not open.");
    }
    if (token_offset >= header_.total_tokens) {
        return {};
    }

    size_t count = sequence_length;
    if (token_offset + count > header_.total_tokens) {
        count = static_cast<size_t>(header_.total_tokens - token_offset);
    }

    std::vector<uint16_t> batch(count);
    std::memcpy(batch.data(), &tokens_data_[token_offset], count * sizeof(uint16_t));
    return batch;
}

} // namespace tok
