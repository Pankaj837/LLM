#pragma once
#include <string>
#include <vector>
#include "vocabulary.h"

namespace tok {

// Decodes a sequence of token ids back into raw text by concatenating each
// token's byte sequence, then interpreting the result as UTF-8.
class Decoder {
public:
    explicit Decoder(const Vocabulary& vocab);

    std::string decode(const std::vector<int>& ids) const;

private:
    const Vocabulary& vocab_;
};

} // namespace tok
