#include "encoder.h"
#include "utils.h"

#include <climits>
#include <limits>

namespace tok {

Encoder::Encoder(const Vocabulary& vocab) : vocab_(vocab) {}

std::vector<int> Encoder::bpe_encode_piece(const std::string& piece) const {
    if (piece.empty()) return {};

    // Start with one symbol per raw byte.
    std::vector<std::string> parts;
    parts.reserve(piece.size());
    for (unsigned char c : piece) {
        parts.emplace_back(1, static_cast<char>(c));
    }

    // Repeatedly merge the adjacent pair with the lowest token id (rank),
    // exactly mirroring the training merge order.
    while (parts.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = std::string::npos;

        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            std::string merged = parts[i] + parts[i + 1];
            if (vocab_.has_token(merged)) {
                int rank = vocab_.id_of(merged);
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                }
            }
        }

        if (best_idx == std::string::npos) break; // no mergeable pair left

        parts[best_idx] = parts[best_idx] + parts[best_idx + 1];
        parts.erase(parts.begin() + static_cast<long>(best_idx) + 1);
    }

    std::vector<int> ids;
    ids.reserve(parts.size());
    for (const auto& p : parts) {
        ids.push_back(vocab_.id_of(p)); // every part is guaranteed to be a known token
    }
    return ids;
}

std::vector<int> Encoder::encode_ordinary(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& chunk : pretokenize(text)) {
        auto chunk_ids = bpe_encode_piece(chunk);
        ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
    }
    return ids;
}

std::vector<int> Encoder::encode(const std::string& text,
                                  const std::unordered_set<std::string>& allowed_special) const {
    if (allowed_special.empty()) {
        return encode_ordinary(text);
    }

    std::vector<int> ids;
    size_t pos = 0;
    const size_t n = text.size();

    while (pos < n) {
        // Find the earliest occurrence of any allowed special token.
        size_t best_at = std::string::npos;
        std::string best_special;

        for (const auto& special : allowed_special) {
            if (special.empty()) continue;
            size_t found = text.find(special, pos);
            if (found != std::string::npos &&
                (best_at == std::string::npos || found < best_at ||
                 (found == best_at && special.size() > best_special.size()))) {
                best_at = found;
                best_special = special;
            }
        }

        if (best_at == std::string::npos) {
            // No more special tokens ahead: encode the remaining text plainly.
            auto rest = encode_ordinary(text.substr(pos));
            ids.insert(ids.end(), rest.begin(), rest.end());
            break;
        }

        if (best_at > pos) {
            auto before = encode_ordinary(text.substr(pos, best_at - pos));
            ids.insert(ids.end(), before.begin(), before.end());
        }

        ids.push_back(vocab_.special_id(best_special));
        pos = best_at + best_special.size();
    }

    return ids;
}

} // namespace tok
