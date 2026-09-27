#include "utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <array>

namespace tok {

// =======================================================================
// File I/O
// =======================================================================

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open file for reading: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open file for writing: " + path);
    }
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
}

// =======================================================================
// Base64
// =======================================================================

namespace {
const char kB64Chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::array<int, 256> make_decode_table() {
    std::array<int, 256> table{};
    table.fill(-1);
    for (int i = 0; i < 64; ++i) {
        table[static_cast<unsigned char>(kB64Chars[i])] = i;
    }
    return table;
}
} // namespace

std::string base64_encode(const std::string& data) {
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    const size_t n = data.size();
    while (i + 3 <= n) {
        uint32_t chunk = (static_cast<unsigned char>(data[i]) << 16) |
                          (static_cast<unsigned char>(data[i + 1]) << 8) |
                          (static_cast<unsigned char>(data[i + 2]));
        out.push_back(kB64Chars[(chunk >> 18) & 0x3F]);
        out.push_back(kB64Chars[(chunk >> 12) & 0x3F]);
        out.push_back(kB64Chars[(chunk >> 6) & 0x3F]);
        out.push_back(kB64Chars[chunk & 0x3F]);
        i += 3;
    }

    const size_t rem = n - i;
    if (rem == 1) {
        uint32_t chunk = static_cast<unsigned char>(data[i]) << 16;
        out.push_back(kB64Chars[(chunk >> 18) & 0x3F]);
        out.push_back(kB64Chars[(chunk >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        uint32_t chunk = (static_cast<unsigned char>(data[i]) << 16) |
                          (static_cast<unsigned char>(data[i + 1]) << 8);
        out.push_back(kB64Chars[(chunk >> 18) & 0x3F]);
        out.push_back(kB64Chars[(chunk >> 12) & 0x3F]);
        out.push_back(kB64Chars[(chunk >> 6) & 0x3F]);
        out.push_back('=');
    }
    return out;
}

std::string base64_decode(const std::string& data) {
    static const std::array<int, 256> decode_table = make_decode_table();

    std::string out;
    out.reserve((data.size() / 4) * 3);

    int val = 0;
    int valb = -8;
    for (unsigned char c : data) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        int d = decode_table[c];
        if (d == -1) continue; // skip unknown chars defensively
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// =======================================================================
// Pre-tokenization
// =======================================================================

namespace {

enum class CharClass { Space, Letter, Digit, Other };

CharClass classify(unsigned char c) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
        return CharClass::Space;
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) {
        // Treat any UTF-8 continuation/lead byte as "letter" so multi-byte
        // unicode characters stay grouped with adjacent letters.
        return CharClass::Letter;
    }
    if (c >= '0' && c <= '9') {
        return CharClass::Digit;
    }
    return CharClass::Other;
}

// Returns the length of a contraction suffix ('s, 't, 're, 've, 'm, 'll, 'd)
// starting at position i, or 0 if none matches.
size_t match_contraction(const std::string& text, size_t i) {
    static const std::vector<std::string> kContractions = {
        "'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
    for (const auto& c : kContractions) {
        if (text.compare(i, c.size(), c) == 0) return c.size();
    }
    return 0;
}

} // namespace

std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<std::string> tokens;
    const size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        // 1) Contractions, e.g. "'ve", "'ll" - matched as their own token.
        if (text[i] == '\'') {
            size_t len = match_contraction(text, i);
            if (len > 0) {
                tokens.push_back(text.substr(i, len));
                i += len;
                continue;
            }
        }

        CharClass cls = classify(static_cast<unsigned char>(text[i]));

        if (cls == CharClass::Space) {
            size_t j = i;
            while (j < n && classify(static_cast<unsigned char>(text[j])) == CharClass::Space) {
                ++j;
            }
            size_t run_len = j - i;

            if (j < n) {
                // Whitespace run is followed by more content: keep all but
                // the last space as a standalone whitespace token, and let
                // the final space become the leading space of the next
                // word/number/punctuation token (mirrors " word" style
                // tokens in GPT-2/tiktoken).
                if (run_len > 1) {
                    tokens.push_back(text.substr(i, run_len - 1));
                }
                CharClass next_cls = classify(static_cast<unsigned char>(text[j]));
                size_t k = j + 1;
                while (k < n && classify(static_cast<unsigned char>(text[k])) == next_cls) {
                    ++k;
                }
                // substr starting at (j - 1) includes the one leading space.
                tokens.push_back(text.substr(j - 1, k - (j - 1)));
                i = k;
            } else {
                // Trailing whitespace run at end of string: emit as-is.
                tokens.push_back(text.substr(i, run_len));
                i = j;
            }
        } else {
            size_t j = i + 1;
            while (j < n && classify(static_cast<unsigned char>(text[j])) == cls) {
                ++j;
            }
            tokens.push_back(text.substr(i, j - i));
            i = j;
        }
    }

    return tokens;
}

// =======================================================================
// Fine-Grained Text Cleaning & Sanitization
// =======================================================================

std::string sanitize_utf8_and_normalize(const std::string& input) {
    std::string out;
    out.reserve(input.size());

    size_t i = 0;
    const size_t n = input.size();

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(input[i]);

        // 1) Strip out NUL bytes
        if (c == 0x00) {
            i++;
            continue;
        }

        // 2) Normalize Windows CRLF (\r\n) or lone \r to \n
        if (c == '\r') {
            if (i + 1 < n && input[i + 1] == '\n') {
                i++; // Skip \r in \r\n
            }
            out.push_back('\n');
            i++;
            continue;
        }

        // 3) Validate ASCII (0x01..0x7F)
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
            i++;
            continue;
        }

        // 4) Validate UTF-8 multi-byte sequences
        size_t len = 0;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        if (len == 0 || i + len > n) {
            // Invalid UTF-8 lead byte or truncated sequence: skip byte
            i++;
            continue;
        }

        bool valid = true;
        for (size_t j = 1; j < len; ++j) {
            unsigned char continuation = static_cast<unsigned char>(input[i + j]);
            if ((continuation & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }

        if (valid) {
            out.append(input, i, len);
            i += len;
        } else {
            i++; // skip invalid byte
        }
    }

    return out;
}

// =======================================================================
// Misc string helpers
// =======================================================================

std::vector<std::string> split_string(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == delim) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::string join_strings(const std::vector<std::string>& parts, const std::string& sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += sep;
        out += parts[i];
    }
    return out;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

} // namespace tok

