#pragma once
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <array>
#include <stdexcept>

// #include "common.h"
#include "blake2/blake2.h"

namespace troy {namespace utils {

    class HashFunction {
    public:
        HashFunction() = delete;

        static constexpr std::size_t hash_block_uint64_count = 4;

        static constexpr std::size_t hash_block_byte_count = hash_block_uint64_count * 8;

        using HashBlock = std::array<std::uint64_t, hash_block_uint64_count>;

        static constexpr HashBlock hash_zero_block{ { 0, 0, 0, 0 } };

        inline static void hash(const std::uint64_t *input, std::size_t uint64_count, HashBlock &destination)
        {
            if (blake2b(&destination, hash_block_byte_count, input, uint64_count * 8, nullptr, 0) !=
                0)
            {
                throw std::runtime_error("blake2b failed");
            }
        }

    };
    
    inline std::ostream& operator<<(std::ostream& os, const HashFunction::HashBlock& hash)
    {
        os << "[";
        for (size_t i = 0; i < HashFunction::hash_block_uint64_count; ++i) {
            os << std::hex << std::setfill('0') << std::setw(16) << hash[i];
            if (i < HashFunction::hash_block_uint64_count - 1) {
                os << ", ";
            }
        }
        os << "]";
        // reset default formatting
        os << std::dec << std::setfill(' ') << std::setw(0);
        return os;
    }

}}