#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <memory>

#include "vocabulary.h"
#include "encoder.h"
#include "decoder.h"
#include "trainer.h"

namespace tok {

// High-level facade tying Vocabulary + Encoder + Decoder + Trainer together,
// similar in spirit to Python's `tiktoken.Encoding`.
//
// Typical usage:
//
//   tok::Tokenizer tokenizer;
//   tokenizer.train(corpus, /*vocab_size=*/1000, {"<|endoftext|>"});
//   auto ids = tokenizer.encode("Hello, world!");
//   std::string text = tokenizer.decode(ids);
//   tokenizer.save("vocab.tok");
//
//   tok::Tokenizer loaded;
//   loaded.load("vocab.tok");
class Tokenizer {
public:
    Tokenizer();

    // Trains a brand-new tokenizer from a text corpus. Replaces any
    // previously trained/loaded vocabulary.
    void train(const std::vector<std::string>& corpus,
               int vocab_size,
               const std::vector<std::string>& special_tokens = {});

    // Encodes text. This overload treats every special token registered in
    // the vocabulary as "allowed" (i.e. it will be recognized and emitted
    // as a single token wherever it appears in the input).
    std::vector<int> encode(const std::string& text) const;

    // Encodes text, only recognizing the given subset of special tokens.
    std::vector<int> encode(const std::string& text,
                             const std::unordered_set<std::string>& allowed_special) const;

    // Decodes a sequence of token ids back to text.
    std::string decode(const std::vector<int>& ids) const;

    // Persistence.
    void save(const std::string& path) const;
    void load(const std::string& path);

    size_t vocab_size() const;
    const Vocabulary& vocabulary() const { return vocab_; }

private:
    Vocabulary vocab_;
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Decoder> decoder_;

    // (Re)builds encoder_/decoder_ views over the current vocab_. Must be
    // called after train()/load() replace the vocabulary.
    void rebuild_views();
};

} // namespace tok
