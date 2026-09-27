#include "decoder.h"

namespace tok {

Decoder::Decoder(const Vocabulary& vocab) : vocab_(vocab) {}

std::string Decoder::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        out += vocab_.bytes_of(id);
    }
    return out;
}

} // namespace tok
