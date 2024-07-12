#pragma once

#include <iostream>
#include <stdexcept>

namespace troy::utils::compression {

    namespace zstd {
        size_t compressed_size_upperbound(size_t input_size);
        size_t compress(const void* input, size_t input_size, std::ostream& output);
        size_t decompress(const void* input, size_t input_size, std::ostream& output);
        bool available();
    }

    enum class CompressionMode: uint8_t {
        Nil = 0,
        Zstd = 1,
    };

    inline size_t compressed_size_upperbound(size_t input_size, CompressionMode mode) {
        switch (mode) {
            case CompressionMode::Nil: return input_size;
            case CompressionMode::Zstd: return zstd::compressed_size_upperbound(input_size);
        }
        throw std::runtime_error("Invalid compression mode");
    }
    inline size_t compress(const void* input, size_t input_size, std::ostream& output, CompressionMode mode) {
        switch (mode) {
            case CompressionMode::Nil: {
                throw std::invalid_argument("[utils::compression::compress] Should not call this function with CompressionMode::Nil");
            }
            case CompressionMode::Zstd: return zstd::compress(input, input_size, output);
        }
        throw std::runtime_error("Invalid compression mode");
    
    }
    inline size_t decompress(const void* input, size_t input_size, std::ostream& output, CompressionMode mode) {
        switch (mode) {
            case CompressionMode::Nil: {
                throw std::invalid_argument("[utils::compression::decompress] Should not call this function with CompressionMode::Nil");
            }
            case CompressionMode::Zstd: return zstd::decompress(input, input_size, output);
        }
        throw std::runtime_error("Invalid compression mode");
    }
    inline bool available(CompressionMode mode) {
        switch (mode) {
            case CompressionMode::Nil: return true;
            case CompressionMode::Zstd: return zstd::available();
        }
        throw std::runtime_error("Invalid compression mode");
    }

}

namespace troy {
    using utils::compression::CompressionMode;
}