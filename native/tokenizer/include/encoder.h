#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include "vocabulary.h"

namespace tok {

// Encodes raw text into token ids using a trained Vocabulary, following
// tiktoken's byte-pair-merge algorithm:
//
//   1. Pre-tokenize text into word-like chunks.
//   2. For each chunk, start with one symbol per raw byte.
//   3. Repeatedly merge the adjacent symbol-pair whose concatenation has
//      the lowest rank (== lowest token id) in the vocabulary, until no
//      further merge is possible.
//   4. Map the final symbols to their token ids.
class Encoder {
public:
    explicit Encoder(const Vocabulary& vocab);

    // Encodes plain text with no special-token handling: any special-token
    // substrings are treated as ordinary text and BPE-split like anything
    // else.
    std::vector<int> encode_ordinary(const std::string& text) const;

    // Encodes text, additionally recognizing any of the strings in
    // `allowed_special` (which must already exist in the vocabulary as
    // special tokens) as single atomic tokens wherever they appear.
    std::vector<int> encode(const std::string& text,
                             const std::unordered_set<std::string>& allowed_special = {}) const;

private:
    const Vocabulary& vocab_;

    // Runs the core byte-pair-merge loop on one pre-token's raw bytes and
    // returns the resulting token ids.
    std::vector<int> bpe_encode_piece(const std::string& piece) const;
};

} // namespace tok
