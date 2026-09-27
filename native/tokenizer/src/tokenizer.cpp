#include "tokenizer.h"

#include <stdexcept>

namespace tok {

Tokenizer::Tokenizer() {
    // Start with an (empty-of-merges) byte vocabulary so encode()/decode()
    // are always safe to call, even before train()/load().
    vocab_.init_byte_vocab();
    rebuild_views();
}

void Tokenizer::rebuild_views() {
    encoder_ = std::make_unique<Encoder>(vocab_);
    decoder_ = std::make_unique<Decoder>(vocab_);
}

void Tokenizer::train(const std::vector<std::string>& corpus,
                       int vocab_size,
                       const std::vector<std::string>& special_tokens) {
    TrainerConfig config;
    config.vocab_size = vocab_size;
    config.special_tokens = special_tokens;

    BPETrainer trainer(config);
    vocab_ = trainer.train(corpus);
    rebuild_views();
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::unordered_set<std::string> all_special;
    for (const auto& [text_key, id] : vocab_.specials()) {
        (void)id;
        all_special.insert(text_key);
    }
    return encoder_->encode(text, all_special);
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                    const std::unordered_set<std::string>& allowed_special) const {
    return encoder_->encode(text, allowed_special);
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    return decoder_->decode(ids);
}

void Tokenizer::save(const std::string& path) const {
    vocab_.save(path);
}

void Tokenizer::load(const std::string& path) {
    vocab_.load(path);
    rebuild_views();
}

size_t Tokenizer::vocab_size() const {
    return vocab_.total_size();
}

} // namespace tok
