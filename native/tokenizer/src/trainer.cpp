#include "trainer.h"
#include "utils.h"

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace tok {

namespace {

using Pair = std::pair<std::string, std::string>;

struct Word {
    std::vector<std::string> symbols; // current symbol sequence for this unique pre-token
    long long freq = 0;               // how many times this pre-token occurred in the corpus
};

} // namespace

BPETrainer::BPETrainer(TrainerConfig config) : config_(std::move(config)) {}

Vocabulary BPETrainer::train(const std::vector<std::string>& corpus) {
    if (config_.vocab_size < 256) {
        throw std::runtime_error("vocab_size must be >= 256 (256 raw byte tokens are always kept)");
    }

    // 1) Pre-tokenize the whole corpus and collect unique chunks with their frequency.
    std::unordered_map<std::string, long long> chunk_freq;
    for (const auto& doc : corpus) {
        for (const auto& chunk : pretokenize(doc)) {
            chunk_freq[chunk]++;
        }
    }

    std::vector<Word> words;
    words.reserve(chunk_freq.size());
    for (const auto& [chunk, freq] : chunk_freq) {
        Word w;
        w.freq = freq;
        w.symbols.reserve(chunk.size());
        for (unsigned char c : chunk) {
            w.symbols.emplace_back(1, static_cast<char>(c));
        }
        words.push_back(std::move(w));
    }

    // 2) Build initial pair frequencies and inverted index (pair -> word indices).
    std::map<Pair, long long> pair_counts;
    std::map<Pair, std::unordered_set<size_t>> pair_to_words;

    for (size_t w_idx = 0; w_idx < words.size(); ++w_idx) {
        const auto& syms = words[w_idx].symbols;
        long long freq = words[w_idx].freq;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            Pair p{syms[i], syms[i + 1]};
            pair_counts[p] += freq;
            pair_to_words[p].insert(w_idx);
        }
    }

    // 3) Initialize vocabulary with the 256 raw byte tokens.
    Vocabulary vocab;
    vocab.init_byte_vocab();

    // 4) Iteratively merge the most frequent adjacent pair incrementally.
    const int target = config_.vocab_size;
    int merges_done = 0;

    while (static_cast<int>(vocab.size()) < target) {
        if (pair_counts.empty()) {
            if (config_.verbose) {
                std::cerr << "No more pairs to merge; stopping early at vocab size "
                          << vocab.size() << "\n";
            }
            break;
        }

        // Find best pair with maximum frequency.
        auto best_it = std::max_element(
            pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second < b.second;
                return a.first < b.first; // deterministic tie-breaker
            });

        if (best_it == pair_counts.end() || best_it->second <= 0) {
            break;
        }

        Pair best_pair = best_it->first;
        long long best_freq = best_it->second;
        std::string merged = best_pair.first + best_pair.second;

        // Remove best_pair from active pair_counts so it won't be picked again.
        pair_counts.erase(best_it);

        if (vocab.has_token(merged)) {
            continue;
        }

        vocab.add_merge_token(merged);
        ++merges_done;

        if (config_.verbose) {
            std::cerr << "merge " << merges_done << ": (" << best_pair.first.size()
                      << "b + " << best_pair.second.size() << "b) -> " << merged.size()
                      << "b tokens, freq=" << best_freq
                      << ", vocab_size=" << vocab.size() << "\n";
        }

        // Incrementally update only the words containing best_pair.
        auto it_words = pair_to_words.find(best_pair);
        if (it_words == pair_to_words.end()) continue;

        std::unordered_set<size_t> word_indices = std::move(it_words->second);
        pair_to_words.erase(it_words);

        for (size_t w_idx : word_indices) {
            Word& w = words[w_idx];
            long long w_freq = w.freq;
            auto& syms = w.symbols;

            if (syms.size() < 2) continue;

            std::vector<std::string> new_syms;
            new_syms.reserve(syms.size());

            size_t i = 0;
            while (i < syms.size()) {
                if (i + 1 < syms.size() && syms[i] == best_pair.first && syms[i + 1] == best_pair.second) {
                    // Decrement pair before this merge if i > 0
                    if (i > 0) {
                        Pair prev_p{syms[i - 1], syms[i]};
                        pair_counts[prev_p] -= w_freq;
                        if (pair_counts[prev_p] <= 0) pair_counts.erase(prev_p);
                    }
                    // Decrement pair after this merge if i + 2 < syms.size()
                    if (i + 2 < syms.size()) {
                        Pair next_p{syms[i + 1], syms[i + 2]};
                        pair_counts[next_p] -= w_freq;
                        if (pair_counts[next_p] <= 0) pair_counts.erase(next_p);
                    }

                    new_syms.push_back(merged);

                    // Add new pair with previous symbol if new_syms has > 1 symbol
                    if (new_syms.size() > 1) {
                        Pair new_prev_p{new_syms[new_syms.size() - 2], merged};
                        pair_counts[new_prev_p] += w_freq;
                        pair_to_words[new_prev_p].insert(w_idx);
                    }
                    // Add new pair with next symbol if i + 2 < syms.size()
                    if (i + 2 < syms.size()) {
                        Pair new_next_p{merged, syms[i + 2]};
                        pair_counts[new_next_p] += w_freq;
                        pair_to_words[new_next_p].insert(w_idx);
                    }

                    i += 2;
                } else {
                    new_syms.push_back(syms[i]);
                    i += 1;
                }
            }

            w.symbols = std::move(new_syms);
        }
    }

    // 5) Add special tokens after the merge vocabulary.
    for (const auto& special : config_.special_tokens) {
        if (!vocab.has_special(special)) {
            vocab.add_special_token(special);
        }
    }

    return vocab;
}

} // namespace tok

