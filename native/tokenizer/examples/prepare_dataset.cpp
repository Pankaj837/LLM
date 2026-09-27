#include "tokenizer.h"
#include "dataset_prep.h"
#include "mmap_dataloader.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

namespace {

std::string extract_text_from_line(const std::string& line) {
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start != std::string::npos && line[start] == '{') {
        static const std::vector<std::string> kKeys = {"\"text\":", "\"content\":", "\"document\":", "\"body\":"};
        for (const auto& key : kKeys) {
            size_t pos = line.find(key);
            if (pos != std::string::npos) {
                size_t val_start = line.find('"', pos + key.size());
                if (val_start != std::string::npos) {
                    std::string out;
                    out.reserve(line.size());
                    size_t i = val_start + 1;
                    while (i < line.size()) {
                        if (line[i] == '\\' && i + 1 < line.size()) {
                            char next = line[i + 1];
                            if (next == '"') { out.push_back('"'); i += 2; }
                            else if (next == '\\') { out.push_back('\\'); i += 2; }
                            else if (next == '/') { out.push_back('/'); i += 2; }
                            else if (next == 'n') { out.push_back('\n'); i += 2; }
                            else if (next == 'r') { out.push_back('\r'); i += 2; }
                            else if (next == 't') { out.push_back('\t'); i += 2; }
                            else if (next == 'u' && i + 5 < line.size()) {
                                std::string hex = line.substr(i + 2, 4);
                                try {
                                    uint32_t cp = std::stoul(hex, nullptr, 16);
                                    if (cp < 0x80) {
                                        out.push_back(static_cast<char>(cp));
                                    } else if (cp < 0x800) {
                                        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                                    } else {
                                        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                                    }
                                } catch (...) {
                                    out.append(line, i, 6);
                                }
                                i += 6;
                            } else {
                                out.push_back(line[i]);
                                i++;
                            }
                        } else if (line[i] == '"') {
                            break;
                        } else {
                            out.push_back(line[i]);
                            i++;
                        }
                    }
                    if (!out.empty()) return out;
                }
            }
        }
    }
    return line;
}

std::vector<std::string> load_lines(const std::string& path) {
    std::string content = tok::read_file(path);
    std::vector<std::string> lines;
    std::istringstream in(content);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            std::string text = extract_text_from_line(line);
            if (!text.empty()) {
                lines.push_back(text);
            }
        }
    }
    return lines;
}

} // namespace

void print_usage(const char* prog_name) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " [bpe_sample_path] [vocab_out_path] [target_vocab_size]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  bpe_sample_path   : Path to text corpus file (default: Mathematics_Dataset-20260825T044106Z-1-001/Mathematics_Dataset/bpe_sample/train_50mb.txt)\n";
    std::cout << "  vocab_out_path    : Output path for trained vocabulary token file (default: vocab_math.tok)\n";
    std::cout << "  target_vocab_size : Target vocabulary size > 256 (default: 4000)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << "\n";
    std::cout << "  " << prog_name << " dataset.txt\n";
    std::cout << "  " << prog_name << " dataset.txt vocab_math.tok 4000\n";
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help" || arg1 == "help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "===========================================================\n";
    std::cout << " Tokenizer Optimization & Dataset Compilation Pipeline\n";
    std::cout << "===========================================================\n\n";

    std::string bpe_sample_path = "Mathematics_Dataset-20260825T044106Z-1-001/Mathematics_Dataset/bpe_sample/train_50mb.txt";
    if (argc >= 2) {
        bpe_sample_path = argv[1];
    }

    std::string vocab_out_path = "vocab_math.tok";
    if (argc >= 3) {
        vocab_out_path = argv[2];
    }

    int target_vocab_size = 4000; // Default 4000
    if (argc >= 4) {
        try {
            target_vocab_size = std::stoi(argv[3]);
            if (target_vocab_size <= 256) {
                std::cerr << "[Error] target_vocab_size must be greater than 256 (byte-level vocabulary requires at least 256 base bytes).\n\n";
                print_usage(argv[0]);
                return 1;
            }
        } catch (const std::exception&) {
            std::cerr << "[Error] Invalid target_vocab_size argument '" << argv[3] << "'. Expected an integer.\n\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Step 1: Train BPE Tokenizer on Sample
    std::cout << "[Step 1/3] Loading BPE Training Sample from: " << bpe_sample_path << std::endl;
    std::vector<std::string> bpe_corpus;
    try {
        bpe_corpus = load_lines(bpe_sample_path);
    } catch (const std::exception& e) {
        std::cerr << "\n[Error] " << e.what() << "\n";
        std::cerr << "Could not open or read the specified dataset file.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Loaded " << bpe_corpus.size() << " documents for BPE vocabulary training." << std::endl;
    std::cout << "Training BPE Vocabulary (target size = " << target_vocab_size << ")..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    tok::Tokenizer tokenizer;
    tokenizer.train(bpe_corpus, target_vocab_size, {"<|endoftext|>"});

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "BPE Training Completed in " << duration << " seconds!" << std::endl;
    std::cout << "Trained Vocabulary Size: " << tokenizer.vocab_size() << std::endl;

    try {
        tokenizer.save(vocab_out_path);
        std::cout << "Saved Vocabulary Dictionary to: " << vocab_out_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n[Error] Failed to save vocabulary file to " << vocab_out_path << ": " << e.what() << "\n";
        return 1;
    }

    // Step 2: Use loaded documents for Subsampling & Binary Compilation
    std::cout << "[Step 2/3] Using Ingested Corpus (" << bpe_corpus.size() << " documents) for Binary Compilation...\n";
    std::vector<std::string> full_docs = bpe_corpus;


    std::cout << "\n[Step 3/3] Compiling Binary Dataset (Subsampling target = 250,000,000 tokens)...\n";
    std::string train_bin = "train.bin";
    std::string val_bin = "val.bin";

    uint64_t token_target = 250000000ULL; // 250M Target Tokens

    tok::BinaryCompileResult comp_result;
    try {
        comp_result = tok::compile_dataset_to_binary(
            full_docs, tokenizer, train_bin, val_bin, 0.05, token_target, "<|endoftext|>");
    } catch (const std::exception& e) {
        std::cerr << "\n[Error] Binary dataset compilation failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n===========================================================\n";
    std::cout << " Compilation Successful!\n";
    std::cout << "===========================================================\n";
    std::cout << " Train Tokens Exported: " << comp_result.train_tokens << " -> " << comp_result.train_bin_path << "\n";
    std::cout << " Val Tokens Exported:   " << comp_result.val_tokens << " -> " << comp_result.val_bin_path << "\n";
    std::cout << " Total Subsampled:      " << (comp_result.train_tokens + comp_result.val_tokens) << " tokens\n\n";

    // Verify DataLoader
    std::cout << "Verifying Memory-Mapped DataLoader on " << train_bin << "...\n";
    tok::MMapDataLoader loader;
    try {
        loader.open(train_bin);
        std::cout << "Header Magic Verified: 0x" << std::hex << loader.header().magic << std::dec << "\n";
        std::cout << "Header Total Tokens: " << loader.total_tokens() << "\n";
        auto batch = loader.get_batch(0, 10);
        std::cout << "First 10 Token IDs from train.bin: ";
        for (size_t i = 0; i < batch.size(); ++i) {
            std::cout << batch[i] << (i + 1 < batch.size() ? ", " : "\n");
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[Error] DataLoader verification failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nAll handover tasks completed successfully!\n";
    return 0;
}
