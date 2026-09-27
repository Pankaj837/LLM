#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace tok {

// A Vocabulary maps byte-sequences ("tokens") <-> integer ids.
//
//  - Ids 0..255            : the 256 raw byte tokens (always present).
//  - Ids 256..N-1           : merge tokens learned during training, stored
//                             in the exact order they were merged. Because
//                             lower ids were merged earlier, the numeric id
//                             doubles as the BPE "rank" -- exactly the
//                             scheme tiktoken itself uses.
//  - Ids N..N+specials-1    : special tokens (e.g. "<|endoftext|>"),
//                             appended after the merge vocabulary. These
//                             are matched as whole strings and are never
//                             split by the BPE merge loop.
class Vocabulary {
public:
    Vocabulary();

    // Populate ids 0..255 with the raw single-byte tokens. Must be called
    // once before any merge tokens are added.
    void init_byte_vocab();

    // Adds a merged token (bytes = concatenation of two existing tokens).
    // Returns the new token's id (== its rank). Throws if the token
    // already exists.
    int add_merge_token(const std::string& token_bytes);

    // Adds a special token (matched as a whole string, not built from
    // byte merges). Returns its id. Throws if it already exists.
    int add_special_token(const std::string& text);

    bool has_token(const std::string& token_bytes) const;
    int id_of(const std::string& token_bytes) const;   // throws if missing
    const std::string& bytes_of(int id) const;          // throws if missing

    bool is_special_id(int id) const;
    bool has_special(const std::string& text) const;
    int special_id(const std::string& text) const;       // throws if missing

    size_t size() const;        // size of the byte + merge vocabulary only
    size_t total_size() const;  // byte + merge vocabulary + specials

    const std::unordered_map<std::string, int>& specials() const { return special_to_id_; }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    std::unordered_map<std::string, int> special_to_id_;
    std::unordered_map<int, std::string> special_id_to_text_;
};

} // namespace tok
