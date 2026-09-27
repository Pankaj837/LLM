// Lightweight hand-rolled test suite (no gtest dependency, so the project
// builds with nothing but the standard library + CMake).
//
// Run via:  ./test_tokenizer   or   ctest --test-dir build

#include "tokenizer.h"
#include "vocabulary.h"
#include "encoder.h"
#include "decoder.h"
#include "trainer.h"
#include "utils.h"
#include "dataset_prep.h"
#include "mmap_dataloader.h"

#include <iostream>
#include <cassert>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& description) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << "[FAIL] " << description << "\n";
    } else {
        std::cout << "[ OK ] " << description << "\n";
    }
}

// ---------------------------------------------------------------------
// utils tests
// ---------------------------------------------------------------------
void test_base64_roundtrip() {
    std::vector<std::string> samples = {
        "", "a", "ab", "abc", "abcd",
        std::string("\x00\x01\x02\xff", 4),
        "The quick brown fox jumps over the lazy dog."
    };
    for (const auto& s : samples) {
        std::string encoded = tok::base64_encode(s);
        std::string decoded = tok::base64_decode(encoded);
        check(decoded == s, "base64 roundtrip for string of length " + std::to_string(s.size()));
    }
}

void test_pretokenize_basic() {
    auto tokens = tok::pretokenize("Hello, world!");
    // Expect roughly: "Hello" "," " world" "!"
    check(!tokens.empty(), "pretokenize produces non-empty output");
    std::string joined = tok::join_strings(tokens, "|");
    std::cout << "    pretokenize(\"Hello, world!\") -> " << joined << "\n";

    // Re-joining all pieces (without the pipe separators) must reconstruct
    // the original string exactly -- this is the property that actually
    // matters for correctness.
    std::string rebuilt;
    for (auto& t : tokens) rebuilt += t;
    check(rebuilt == "Hello, world!", "pretokenize output concatenates back to the original text");
}

void test_pretokenize_contraction() {
    auto tokens = tok::pretokenize("don't stop");
    std::string rebuilt;
    for (auto& t : tokens) rebuilt += t;
    check(rebuilt == "don't stop", "pretokenize preserves contractions when rejoined");

    bool found_apostrophe_t = false;
    for (auto& t : tokens) if (t == "'t") found_apostrophe_t = true;
    check(found_apostrophe_t, "pretokenize splits off \"'t\" as its own chunk");
}

// ---------------------------------------------------------------------
// Vocabulary tests
// ---------------------------------------------------------------------
void test_vocabulary_byte_init() {
    tok::Vocabulary vocab;
    vocab.init_byte_vocab();
    check(vocab.size() == 256, "byte vocabulary has exactly 256 entries");
    check(vocab.id_of(std::string(1, 'A')) == 65, "byte token 'A' maps to id 65");
    check(vocab.bytes_of(65) == std::string(1, 'A'), "id 65 maps back to byte 'A'");
}

void test_vocabulary_merges_and_specials() {
    tok::Vocabulary vocab;
    vocab.init_byte_vocab();
    int id = vocab.add_merge_token("th");
    check(id == 256, "first merge token gets id 256");
    check(vocab.has_token("th"), "vocabulary recognizes the new merge token");

    int special_id = vocab.add_special_token("<|endoftext|>");
    check(special_id == static_cast<int>(vocab.size()), "special token id starts right after merge vocab");
    check(vocab.is_special_id(special_id), "special id is flagged as special");
    check(vocab.total_size() == vocab.size() + 1, "total_size accounts for specials");
}

void test_vocabulary_save_load(const std::string& tmp_path) {
    tok::Vocabulary vocab;
    vocab.init_byte_vocab();
    vocab.add_merge_token("th");
    vocab.add_merge_token("the");
    vocab.add_special_token("<|endoftext|>");
    vocab.save(tmp_path);

    tok::Vocabulary loaded;
    loaded.load(tmp_path);

    check(loaded.size() == vocab.size(), "loaded vocab has same merge-vocab size");
    check(loaded.total_size() == vocab.total_size(), "loaded vocab has same total size");
    check(loaded.has_token("the"), "loaded vocab contains merge token 'the'");
    check(loaded.has_special("<|endoftext|>"), "loaded vocab contains special token");
    check(loaded.id_of("the") == vocab.id_of("the"), "loaded merge token id matches original");
}

// ---------------------------------------------------------------------
// Encoder / Decoder roundtrip tests
// ---------------------------------------------------------------------
void test_encode_decode_roundtrip_no_merges() {
    // With only the 256 raw-byte tokens (no merges learned), encoding must
    // still work correctly -- every pretoken just falls back to one token
    // per byte.
    tok::Vocabulary vocab;
    vocab.init_byte_vocab();
    tok::Encoder encoder(vocab);
    tok::Decoder decoder(vocab);

    std::string text = "Hello, world! 123";
    auto ids = encoder.encode_ordinary(text);
    std::string decoded = decoder.decode(ids);
    check(decoded == text, "encode/decode roundtrip with byte-only vocab reproduces original text");
}

void test_trainer_and_roundtrip() {
    std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "the dog barks at the fox",
        "the quick fox runs away from the dog",
        "hello world, hello there",
    };

    tok::TrainerConfig config;
    config.vocab_size = 300; // 256 bytes + 44 learned merges
    config.special_tokens = {"<|endoftext|>"};

    tok::BPETrainer trainer(config);
    tok::Vocabulary vocab = trainer.train(corpus);

    check(vocab.size() <= 300, "trained vocab does not exceed target vocab_size");
    check(vocab.size() > 256, "trainer learned at least one merge");
    check(vocab.has_special("<|endoftext|>"), "trained vocab contains the requested special token");

    tok::Encoder encoder(vocab);
    tok::Decoder decoder(vocab);

    for (const auto& doc : corpus) {
        auto ids = encoder.encode_ordinary(doc);
        std::string decoded = decoder.decode(ids);
        check(decoded == doc, "roundtrip preserves text: \"" + doc + "\"");
    }

    // A frequent word like "the" should now typically compress to fewer
    // tokens than its raw byte length (3 bytes).
    auto the_ids = encoder.encode_ordinary("the");
    check(the_ids.size() <= 3, "\"the\" encodes to at most 3 tokens after training");
    std::cout << "    \"the\" -> " << the_ids.size() << " token(s)\n";
}

void test_special_token_handling() {
    std::vector<std::string> corpus = {"hello world", "goodbye world"};
    tok::TrainerConfig config;
    config.vocab_size = 260;
    config.special_tokens = {"<|endoftext|>"};
    tok::BPETrainer trainer(config);
    tok::Vocabulary vocab = trainer.train(corpus);

    tok::Encoder encoder(vocab);
    tok::Decoder decoder(vocab);

    std::string text = "hello world<|endoftext|>goodbye world";
    auto ids = encoder.encode(text, {"<|endoftext|>"});

    int special_id = vocab.special_id("<|endoftext|>");
    bool contains_special = false;
    for (int id : ids) if (id == special_id) contains_special = true;
    check(contains_special, "encode() emits the special token id when allowed");

    std::string decoded = decoder.decode(ids);
    check(decoded == text, "roundtrip preserves text containing a special token");
}

void test_high_level_tokenizer_api(const std::string& tmp_path) {
    tok::Tokenizer tokenizer;
    std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
    };
    tokenizer.train(corpus, 300, {"<|endoftext|>"});

    auto ids = tokenizer.encode("the quick fox");
    std::string decoded = tokenizer.decode(ids);
    check(decoded == "the quick fox", "Tokenizer::encode/decode roundtrip works");

    tokenizer.save(tmp_path);

    tok::Tokenizer loaded;
    loaded.load(tmp_path);
    auto ids2 = loaded.encode("the quick fox");
    check(ids == ids2, "loaded Tokenizer produces identical encoding to the original");
}

void test_text_cleaning_and_latex_preservation() {
    std::string raw = "Math formula:\r\n\\frac{1}{2} + \\sum_{i=1}^n x_i = 0\0\rCheck\r\n";
    raw.push_back('\0'); // add explicit NUL byte

    std::string cleaned = tok::sanitize_utf8_and_normalize(raw);

    check(cleaned.find("\r") == std::string::npos, "sanitize_utf8_and_normalize removes all \\r line endings");
    check(cleaned.find(std::string(1, '\0')) == std::string::npos, "sanitize_utf8_and_normalize removes NUL bytes");
    check(cleaned.find("\\frac{1}{2}") != std::string::npos, "sanitize_utf8_and_normalize preserves LaTeX commands");
    check(cleaned.find("\\sum_{i=1}^n") != std::string::npos, "sanitize_utf8_and_normalize preserves math subscripts/superscripts");
}

void test_binary_dataset_compilation_and_mmap_dataloader(const std::string& tmp_dir) {
    tok::Tokenizer tokenizer;
    std::vector<std::string> corpus = {
        "Document 1: \\frac{a}{b} + c = d.",
        "Document 2: OpenWebMath sample text for testing binary dataset creation.",
        "Document 3: Third document with math formula \\int_0^1 f(x) dx.",
    };
    tokenizer.train(corpus, 280, {"<|endoftext|>"});

    std::string train_path = tmp_dir + "_train.bin";
    std::string val_path = tmp_dir + "_val.bin";

    auto res = tok::compile_dataset_to_binary(corpus, tokenizer, train_path, val_path, 0.33, 1000, "<|endoftext|>");

    check(res.train_tokens > 0, "compile_dataset_to_binary generates train tokens");
    check(res.val_tokens > 0, "compile_dataset_to_binary generates val tokens");

    tok::MMapDataLoader loader;
    loader.open(train_path);

    check(loader.header().magic == 0x544F4B31, "MMapDataLoader verifies header magic 0x544F4B31");
    check(loader.total_tokens() == res.train_tokens, "MMapDataLoader matches compiled token count");

    auto batch = loader.get_batch(0, 5);
    check(batch.size() == 5, "MMapDataLoader get_batch retrieves requested sequence batch size");
}

} // namespace

int main() {
    test_base64_roundtrip();
    test_pretokenize_basic();
    test_pretokenize_contraction();

    test_vocabulary_byte_init();
    test_vocabulary_merges_and_specials();
    test_vocabulary_save_load("test_vocab.tok");

    test_encode_decode_roundtrip_no_merges();
    test_trainer_and_roundtrip();
    test_special_token_handling();

    test_high_level_tokenizer_api("test_tokenizer.tok");

    test_text_cleaning_and_latex_preservation();
    test_binary_dataset_compilation_and_mmap_dataloader("test_dataset");

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed.\n";
    if (g_failures > 0) {
        std::cerr << g_failures << " check(s) FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}

