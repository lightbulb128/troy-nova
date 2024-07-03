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

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            matrix_reps_index_map.to_device_inplace(pool);
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

        void encode(const std::vector<uint64_t>& values, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext encode_new(const std::vector<uint64_t>& values, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            encode(values, destination, pool);
            return destination;
        }

        void encode_polynomial(const std::vector<uint64_t>& values, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext encode_polynomial_new(const std::vector<uint64_t>& values, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            encode_polynomial(values, destination, pool);
            return destination;
        }

        void decode(const Plaintext& plaintext, std::vector<uint64_t>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline std::vector<uint64_t> decode_new(const Plaintext& plaintext, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<uint64_t> destination(slot_count());
            decode(plaintext, destination, pool);
            return destination;
        }

        void decode_polynomial(const Plaintext& plaintext, std::vector<uint64_t>& destination) const;
        inline std::vector<uint64_t> decode_polynomial_new(const Plaintext& plaintext) const {
            std::vector<uint64_t> destination(slot_count());
            decode_polynomial(plaintext, destination);
            return destination;
        }

        Plaintext scale_up_new(const Plaintext& plain, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void scale_up(const Plaintext& plain, Plaintext& destination, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = scale_up_new(plain, parms_id, pool);
        }
        inline void scale_up_inplace(Plaintext& plain, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            plain = scale_up_new(plain, parms_id, pool);
        }

        Plaintext scale_down_new(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void scale_down(const Plaintext& plain, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = scale_down_new(plain, pool);
        }
        inline void scale_down_inplace(Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            plain = scale_down_new(plain, pool);
        }

        Plaintext centralize_new(const Plaintext& plain, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void centralize(const Plaintext& plain, Plaintext& destination, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = centralize_new(plain, parms_id, pool);
        }
        inline void centralize_inplace(Plaintext& plain, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            plain = centralize_new(plain, parms_id, pool);
        }


    };

}