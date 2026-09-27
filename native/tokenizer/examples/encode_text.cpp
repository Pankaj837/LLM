// Utility tool to encode any sample text string using a saved .tok vocabulary file
// and print a detailed token-by-token breakdown (ID <-> Decoded String).
//
// Usage:
//   ./encode_text my_vocab_thing.tok "The quick brown fox jumps over the lazy dog."

#include "tokenizer.h"
#include "utils.h"

#include <iostream>
#include <iomanip>
#include <string>

namespace {

std::string format_token_bytes(const std::string& bytes) {
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char c : bytes) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c == ' ') out += " ";
        else if (c >= 32 && c <= 126) out.push_back(static_cast<char>(c));
        else {
            char hex_buf[8];
            snprintf(hex_buf, sizeof(hex_buf), "\\x%02X", c);
            out += hex_buf;
        }
    }
    return out;
}

} // namespace

void print_usage(const char* prog_name) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " [vocab_path.tok] [\"sample text to encode\"]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  vocab_path.tok : Path to saved vocabulary file (default: my_vocab_thing.tok)\n";
    std::cout << "  sample text    : Text string to encode and analyze\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog_name << " vocab_math.tok \"2 + 2 = 4\"\n";
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help" || arg1 == "help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::string vocab_path = "my_vocab_thing.tok";
    std::string sample_text = "The quick brown fox jumps over the lazy dog.";

    if (argc >= 2) {
        vocab_path = argv[1];
    }
    if (argc >= 3) {
        sample_text = argv[2];
    }

    tok::Tokenizer tokenizer;
    try {
        tokenizer.load(vocab_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading vocabulary from " << vocab_path << ": " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Loaded Vocabulary: " << vocab_path << " (size: " << tokenizer.vocab_size() << ")\n";
    std::cout << "Input Text: \"" << sample_text << "\"\n\n";

    auto ids = tokenizer.encode(sample_text);

    std::cout << std::left 
              << std::setw(8)  << "Seq #"
              << std::setw(12) << "Token ID"
              << "Decoded Subword Token String\n";
    std::cout << std::string(50, '-') << "\n";

    for (size_t i = 0; i < ids.size(); ++i) {
        int id = ids[i];
        std::string token_str = tokenizer.vocabulary().bytes_of(id);
        std::cout << std::left 
                  << std::setw(8)  << i
                  << std::setw(12) << id
                  << "\"" << format_token_bytes(token_str) << "\"\n";
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << "Total Encoded Tokens: " << ids.size() << "\n";

    std::string decoded = tokenizer.decode(ids);
    std::cout << "Reconstructed Decoded Text: \"" << decoded << "\"\n";
    std::cout << "Roundtrip Exact Match: " << (decoded == sample_text ? "YES" : "NO") << "\n";

    return 0;
}
