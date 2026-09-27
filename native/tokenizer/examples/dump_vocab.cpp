// Utility tool to inspect and dump human-readable ID <-> Token mapping from a saved .tok file.
//
// Usage:
//   ./dump_vocab vocab.tok

#include "vocabulary.h"
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
        else if (c == ' ') out += " "; // space
        else if (c >= 32 && c <= 126) out.push_back(static_cast<char>(c));
        else {
            // Hex escape non-printable byte
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
    std::cout << "  " << prog_name << " [vocab_path.tok]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  vocab_path.tok : Path to saved vocabulary file (default: vocab.tok)\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog_name << " vocab_math.tok\n";
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help" || arg1 == "help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::string vocab_path = "vocab.tok";
    if (argc >= 2) {
        vocab_path = argv[1];
    }

    std::cout << "Loading vocabulary from: " << vocab_path << "\n\n";

    tok::Vocabulary vocab;
    try {
        vocab.load(vocab_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading vocabulary from '" << vocab_path << "': " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << std::left 
              << std::setw(8)  << "ID" 
              << std::setw(12) << "Type" 
              << std::setw(16) << "Base64" 
              << "Decoded Token String" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (size_t id = 0; id < vocab.total_size(); ++id) {
        int int_id = static_cast<int>(id);
        std::string bytes = vocab.bytes_of(int_id);
        std::string b64 = tok::base64_encode(bytes);
        std::string type_str = (int_id < 256) ? "Byte" : (vocab.is_special_id(int_id) ? "Special" : "Merge");

        std::cout << std::left 
                  << std::setw(8)  << int_id 
                  << std::setw(12) << type_str 
                  << std::setw(16) << b64 
                  << "\"" << format_token_bytes(bytes) << "\"" << "\n";
    }

    std::cout << std::string(60, '-') << "\n";
    std::cout << "Total vocabulary size: " << vocab.total_size() << "\n";

    return 0;
}
