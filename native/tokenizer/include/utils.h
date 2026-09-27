#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace tok {

// ---------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------
std::string read_file(const std::string& path);
void write_file(const std::string& path, const std::string& content);

// ---------------------------------------------------------------------
// Base64 (used to serialize raw token bytes in the saved vocab file,
// the same way OpenAI's .tiktoken files store base64-encoded tokens)
// ---------------------------------------------------------------------
std::string base64_encode(const std::string& data);
std::string base64_decode(const std::string& data);

// ---------------------------------------------------------------------
// Pre-tokenization
// ---------------------------------------------------------------------
// Splits raw UTF-8 text into "word-like" chunks, approximating the
// GPT-2 / tiktoken pre-tokenizer regex:
//
//   's|'t|'re|'ve|'m|'ll|'d
//   | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
//
// This is implemented by hand (no <regex> unicode-property support,
// which would require ICU/Boost) so the project has zero external
// dependencies. Bytes >= 0x80 (UTF-8 continuation/lead bytes) are
// treated as "letter" characters so multi-byte UTF-8 sequences stay
// grouped together. Good enough for a learning/demo BPE tokenizer.
std::vector<std::string> pretokenize(const std::string& text);

// ---------------------------------------------------------------------
// Fine-Grained Text Cleaning & Sanitization
// ---------------------------------------------------------------------
// Normalizes line breaks (\r\n -> \n), strips invalid UTF-8 bytes and NUL
// characters, while preserving LaTeX/math formatting (no lowercasing).
std::string sanitize_utf8_and_normalize(const std::string& input);

// ---------------------------------------------------------------------
// Misc string helpers
// ---------------------------------------------------------------------
std::vector<std::string> split_string(const std::string& s, char delim);
std::string join_strings(const std::vector<std::string>& parts, const std::string& sep);
bool starts_with(const std::string& s, const std::string& prefix);

} // namespace tok

