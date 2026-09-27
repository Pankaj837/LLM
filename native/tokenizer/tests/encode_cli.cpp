// Batch encode harness for C++/Python tokenizer parity testing.
// Usage: encode_cli <vocab.tok>
// Reads one line of text per stdin line, prints the space-separated token ids for that line
// (allowed_special = every special token registered in the vocabulary, matching the Python
// port's `BPETokenizer.encode(text, allowed_special=None)` default).
#include "tokenizer.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: encode_cli <vocab.tok>\n";
        return 1;
    }
    tok::Tokenizer tokenizer;
    try {
        tokenizer.load(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "failed to load vocab: " << e.what() << "\n";
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        auto ids = tokenizer.encode(line);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << ids[i];
        }
        std::cout << '\n';
    }
    return 0;
}
