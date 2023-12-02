#pragma once
#include "plaintext.cuh"
#include "he_context.cuh"

namespace troy {

    class BatchEncoder {
    
    private:
        HeContextPointer context_;
        size_t slots_;
        utils::Array<size_t> matrix_reps_index_map;

    public:

        BatchEncoder(HeContextPointer context);

        inline bool on_device() const noexcept {
            return matrix_reps_index_map.on_device();
        }

        inline void to_device_inplace() {
            matrix_reps_index_map.to_device_inplace();
        }

        inline size_t slot_count() const noexcept {
            return slots_;
        }

        inline constexpr size_t row_count() const noexcept {
            return 2;
        }

        inline size_t column_count() const noexcept {
            return slots_ / 2;
        }

        inline HeContextPointer context() const noexcept {
            return context_;
        }

        inline bool simd_encoding_supported() const {
            return matrix_reps_index_map.size() > 0;
        }

        /// Permutes a vector with index bitwise reversed.
        /// The length of the vector must be a power of 2.
        static void reverse_bits(utils::Slice<uint64_t> input);

        void encode(const std::vector<uint64_t>& values, Plaintext& destination) const;
        inline Plaintext encode_new(const std::vector<uint64_t>& values) const {
            Plaintext destination;
            encode(values, destination);
            return destination;
        }

        void encode_polynomial(const std::vector<uint64_t>& values, Plaintext& destination) const;
        inline Plaintext encode_polynomial_new(const std::vector<uint64_t>& values) const {
            Plaintext destination;
            encode_polynomial(values, destination);
            return destination;
        }

        void decode(const Plaintext& plaintext, std::vector<uint64_t>& destination) const;
        inline std::vector<uint64_t> decode_new(const Plaintext& plaintext) const {
            std::vector<uint64_t> destination(slot_count());
            decode(plaintext, destination);
            return destination;
        }

        void decode_polynomial(const Plaintext& plaintext, std::vector<uint64_t>& destination) const;
        inline std::vector<uint64_t> decode_polynomial_new(const Plaintext& plaintext) const {
            std::vector<uint64_t> destination(slot_count());
            decode_polynomial(plaintext, destination);
            return destination;
        }

    };

}