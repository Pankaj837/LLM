cmake_minimum_required(VERSION 3.14)
project(Tokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(tokenizer_lib STATIC
    src/utils.cpp
    src/vocabulary.cpp
    src/encoder.cpp
    src/decoder.cpp
    src/trainer.cpp
    src/tokenizer.cpp
    src/dataset_prep.cpp
    src/mmap_dataloader.cpp
)

target_include_directories(tokenizer_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Example: train.cpp
# ---------------------------------------------------------------------------
add_executable(train_example examples/train.cpp)
target_link_libraries(train_example PRIVATE tokenizer_lib)

# ---------------------------------------------------------------------------
# Executable: prepare_dataset
# ---------------------------------------------------------------------------
add_executable(prepare_dataset examples/prepare_dataset.cpp)
target_link_libraries(prepare_dataset PRIVATE tokenizer_lib)
add_custom_command(TARGET prepare_dataset POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "--------------------------------------------------------------------------------"
    COMMAND ${CMAKE_COMMAND} -E echo "[prepare_dataset] Usage: ./build/prepare_dataset [bpe_sample_path] [vocab_out_path] [target_vocab_size]"
    COMMAND ${CMAKE_COMMAND} -E echo "[prepare_dataset] Example: ./build/prepare_dataset dataset.txt vocab_math.tok 4000"
    COMMAND ${CMAKE_COMMAND} -E echo "--------------------------------------------------------------------------------"
)

# ---------------------------------------------------------------------------
# Executable: dump_vocab
# ---------------------------------------------------------------------------
add_executable(dump_vocab examples/dump_vocab.cpp)
target_link_libraries(dump_vocab PRIVATE tokenizer_lib)

# ---------------------------------------------------------------------------
# Executable: encode_text
# ---------------------------------------------------------------------------
add_executable(encode_text examples/encode_text.cpp)
target_link_libraries(encode_text PRIVATE tokenizer_lib)

# ---------------------------------------------------------------------------
# Custom target: clean_data (removes generated dataset and vocab files without rm)
# ---------------------------------------------------------------------------
add_custom_target(clean_data
    COMMAND ${CMAKE_COMMAND} -E remove -f train.bin val.bin vocab.tok vocab_math.tok vocab_earth.tok vocab_qbio.tok
    COMMAND ${CMAKE_COMMAND} -E echo "Successfully removed generated dataset binaries and vocabulary files."
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE tokenizer_lib)
add_test(NAME TokenizerTests COMMAND test_tokenizer)

# ---------------------------------------------------------------------------
# Executable: encode_cli (batch harness for C++/Python parity testing --
# reads one line of text per stdin line, prints that line's space-separated
# token ids; see ../../tests/test_cpp_parity.py, which drives this exe via
# TOK_CLI + TOK_VOCAB)
# ---------------------------------------------------------------------------
add_executable(encode_cli tests/encode_cli.cpp)
target_link_libraries(encode_cli PRIVATE tokenizer_lib)
