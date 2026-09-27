#include "vocabulary.h"
#include "utils.h"

#include <stdexcept>
#include <sstream>

namespace tok {

Vocabulary::Vocabulary() = default;

void Vocabulary::init_byte_vocab() {
    if (!id_to_token_.empty()) {
        throw std::runtime_error("Vocabulary already initialized");
    }
    id_to_token_.reserve(256);
    for (int b = 0; b < 256; ++b) {
        std::string tok(1, static_cast<char>(static_cast<unsigned char>(b)));
        token_to_id_[tok] = b;
        id_to_token_.push_back(tok);
    }
}

int Vocabulary::add_merge_token(const std::string& token_bytes) {
    if (token_to_id_.count(token_bytes)) {
        throw std::runtime_error("Token already exists in vocabulary");
    }
    int id = static_cast<int>(id_to_token_.size());
    token_to_id_[token_bytes] = id;
    id_to_token_.push_back(token_bytes);
    return id;
}

int Vocabulary::add_special_token(const std::string& text) {
    if (special_to_id_.count(text)) {
        throw std::runtime_error("Special token already exists");
    }
    // Specials live in a numbering space right after the merge vocabulary.
    int id = static_cast<int>(id_to_token_.size() + special_to_id_.size());
    special_to_id_[text] = id;
    special_id_to_text_[id] = text;
    return id;
}

bool Vocabulary::has_token(const std::string& token_bytes) const {
    return token_to_id_.count(token_bytes) != 0;
}

int Vocabulary::id_of(const std::string& token_bytes) const {
    auto it = token_to_id_.find(token_bytes);
    if (it == token_to_id_.end()) {
        throw std::out_of_range("Token not found in vocabulary");
    }
    return it->second;
}

const std::string& Vocabulary::bytes_of(int id) const {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[static_cast<size_t>(id)];
    }
    auto it = special_id_to_text_.find(id);
    if (it != special_id_to_text_.end()) {
        return it->second;
    }
    throw std::out_of_range("Token id not found in vocabulary");
}

bool Vocabulary::is_special_id(int id) const {
    return special_id_to_text_.count(id) != 0;
}

bool Vocabulary::has_special(const std::string& text) const {
    return special_to_id_.count(text) != 0;
}

int Vocabulary::special_id(const std::string& text) const {
    auto it = special_to_id_.find(text);
    if (it == special_to_id_.end()) {
        throw std::out_of_range("Special token not found");
    }
    return it->second;
}

size_t Vocabulary::size() const {
    return id_to_token_.size();
}

size_t Vocabulary::total_size() const {
    return id_to_token_.size() + special_to_id_.size();
}

void Vocabulary::save(const std::string& path) const {
    std::ostringstream ss;
    ss << "# tiktoken-style-vocab\n";
    ss << "V1\n";
    ss << id_to_token_.size() << "\n";
    for (size_t id = 0; id < id_to_token_.size(); ++id) {
        ss << base64_encode(id_to_token_[id]) << " " << id << "\n";
    }
    ss << "SPECIALS\n";
    ss << special_to_id_.size() << "\n";
    for (const auto& [text, id] : special_to_id_) {
        ss << base64_encode(text) << " " << id << "\n";
    }
    write_file(path, ss.str());
}

void Vocabulary::load(const std::string& path) {
    std::string content = read_file(path);
    std::istringstream in(content);

    std::string line;
    std::getline(in, line); // "# tiktoken-style-vocab"
    std::getline(in, line); // "V1"
    if (line != "V1") {
        throw std::runtime_error("Unsupported vocab file version: " + line);
    }

    token_to_id_.clear();
    id_to_token_.clear();
    special_to_id_.clear();
    special_id_to_text_.clear();

    size_t merge_count = 0;
    in >> merge_count;
    in.ignore(); // consume newline
    id_to_token_.resize(merge_count);
    for (size_t i = 0; i < merge_count; ++i) {
        std::string b64;
        int id;
        in >> b64 >> id;
        std::string bytes = base64_decode(b64);
        token_to_id_[bytes] = id;
        id_to_token_[static_cast<size_t>(id)] = bytes;
    }

    std::string marker;
    in >> marker; // "SPECIALS"
    size_t special_count = 0;
    in >> special_count;
    for (size_t i = 0; i < special_count; ++i) {
        std::string b64;
        int id;
        in >> b64 >> id;
        std::string text = base64_decode(b64);
        special_to_id_[text] = id;
        special_id_to_text_[id] = text;
    }
}

} // namespace tok
