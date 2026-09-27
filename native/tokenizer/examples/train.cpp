// Example: train a BPE tokenizer on a text corpus and try it out.
//
// Usage:
//   ./train_example                       trains on a small built-in corpus
//   ./train_example corpus.txt             trains on the given text file (one line per document)
//   ./train_example corpus.txt 1000        also sets the target vocab size (default 500)
//   ./train_example corpus.txt 1000 out.tok  also sets the output vocab file path

#include "tokenizer.h"
#include "utils.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

namespace {

std::vector<std::string> default_corpus() {
    return {
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "The five boxing wizards jump quickly.",
        "Sphinx of black quartz, judge my vow.",
        "The quick brown fox jumps over the lazy dog again and again.",
        "Tokenizers turn text into tokens, and tokens back into text.",
        "Byte pair encoding merges the most frequent adjacent pair of bytes.",
        "This tiny example corpus is intentionally small and repetitive.",
    };
}

std::vector<std::string> load_corpus_from_file(const std::string& path) {
    std::string content = tok::read_file(path);
    std::vector<std::string> lines;
    std::istringstream in(content);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

} // namespace

void print_usage(const char* prog_name) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " [corpus.txt] [vocab_size] [out_path.tok]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  corpus.txt   : Path to input text file (one line per document, default: built-in sample)\n";
    std::cout << "  vocab_size   : Target vocabulary size > 256 (default: 500)\n";
    std::cout << "  out_path.tok : Path to save trained vocabulary (default: vocab.tok)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << "\n";
    std::cout << "  " << prog_name << " corpus.txt\n";
    std::cout << "  " << prog_name << " corpus.txt 1000 out.tok\n";
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help" || arg1 == "help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::vector<std::string> corpus;
    int vocab_size = 500;
    std::string out_path = "vocab.tok";

    if (argc >= 2) {
        std::cout << "Loading corpus from: " << argv[1] << "\n";
        try {
            corpus = load_corpus_from_file(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "[Error] Could not read corpus file '" << argv[1] << "': " << e.what() << "\n\n";
            print_usage(argv[0]);
            return 1;
        }
    } else {
        std::cout << "No corpus file given; using the small built-in example corpus.\n";
        corpus = default_corpus();
    }

    if (argc >= 3) {
        vocab_size = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        out_path = argv[3];
    }

    std::cout << "Training BPE tokenizer: " << corpus.size() << " document(s), "
              << "target vocab_size=" << vocab_size << "\n";

    tok::Tokenizer tokenizer;
    tokenizer.train(corpus, vocab_size, {"<|endoftext|>"});

    std::cout << "Trained vocabulary size: " << tokenizer.vocab_size() << "\n";

    tokenizer.save(out_path);
    std::cout << "Saved vocabulary to: " << out_path << "\n\n";

    // Try it out.
    std::string sample = "The quick brown fox jumps over the lazy dog.";
    auto ids = tokenizer.encode(sample);

    std::cout << "Sample text: \"" << sample << "\"\n";
    std::cout << "Encoded (" << ids.size() << " tokens): ";
    for (size_t i = 0; i < ids.size(); ++i) {
        std::cout << ids[i] << (i + 1 < ids.size() ? " " : "\n");
    }

    std::string decoded = tokenizer.decode(ids);
    std::cout << "Decoded: \"" << decoded << "\"\n";
    std::cout << "Roundtrip OK: " << (decoded == sample ? "yes" : "no") << "\n\n";

    // Demonstrate special-token handling.
    std::string with_special = sample + "<|endoftext|>Another sentence follows.";
    auto ids_special = tokenizer.encode(with_special);
    std::cout << "With special token, encoded to " << ids_special.size() << " tokens.\n";
    std::cout << "Decoded: \"" << tokenizer.decode(ids_special) << "\"\n";

    // Reload from disk to prove persistence works end to end.
    tok::Tokenizer reloaded;
    reloaded.load(out_path);
    auto ids_reloaded = reloaded.encode(sample);
    std::cout << "\nReloaded tokenizer produces identical ids: "
              << (ids_reloaded == ids ? "yes" : "no") << "\n";

    return 0;
}
