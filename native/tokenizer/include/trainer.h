#pragma once
#include <string>
#include <vector>
#include "vocabulary.h"

namespace tok {

struct TrainerConfig {
    int vocab_size = 500;                      // target size of the merge vocabulary (>= 256)
    std::vector<std::string> special_tokens;    // e.g. {"<|endoftext|>"}
    bool verbose = false;                       // print merge progress to stderr
};

// Learns a byte-pair-encoding merge vocabulary from a text corpus, following
// the same algorithm used by GPT-2 / tiktoken:
//
//   1. Pre-tokenize the corpus into "word-like" chunks.
//   2. Represent each unique chunk as a sequence of raw bytes.
//   3. Repeatedly find the most frequent adjacent byte-pair across the whole
//      corpus, merge it into a new token, and repeat until the target
//      vocab size is reached.
class BPETrainer {
public:
    explicit BPETrainer(TrainerConfig config);

    // Trains on the given corpus (each string is treated as one document /
    // line) and returns a fully populated Vocabulary (256 byte tokens +
    // learned merges + special tokens).
    Vocabulary train(const std::vector<std::string>& corpus);

private:
    TrainerConfig config_;
};

} // namespace tok
