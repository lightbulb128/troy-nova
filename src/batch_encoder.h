#pragma once
#include "batch_utils.h"
#include "plaintext.h"
#include "he_context.h"
#include "utils/box.h"

namespace troy {

    class BatchEncoder {
    
    private:
        HeContextPointer context_;
        size_t slots_;
        utils::Array<size_t> matrix_reps_index_map;

        inline bool pool_compatible(MemoryPoolHandle pool) const {
            if (this->on_device()) {
                return pool != nullptr && pool->get_device() == this->device_index();
            } else {
                return true;
            }
        }

    public:

        BatchEncoder(HeContextPointer context);

        inline bool on_device() const noexcept {
            return matrix_reps_index_map.on_device();
        }
        inline size_t device_index() const {
            return matrix_reps_index_map.device_index();
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

        void encode_slice(utils::ConstSlice<uint64_t> values, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext encode_slice_new(utils::ConstSlice<uint64_t> values, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            encode_slice(values, destination, pool);
            return destination;
        }

        void encode_slice_batched(const utils::ConstSliceVec<uint64_t>& values, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline std::vector<Plaintext> encode_slice_new_batched(const utils::ConstSliceVec<uint64_t>& values, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Plaintext> destination(values.size());
            encode_slice_batched(values, batch_utils::collect_pointer(destination), pool);
            return destination;
        }
        
        inline void encode(const std::vector<uint64_t>& values, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            encode_slice(utils::ConstSlice(values.data(), values.size(), false, nullptr), destination, pool);
        }
        inline Plaintext encode_new(const std::vector<uint64_t>& values, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            encode(values, destination, pool);
            return destination;
        }

        void encode_polynomial_slice(utils::ConstSlice<uint64_t> values, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext encode_polynomial_slice_new(utils::ConstSlice<uint64_t> values, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            encode_polynomial_slice(values, destination, pool);
            return destination;
        }
        void encode_polynomial(const std::vector<uint64_t>& values, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            encode_polynomial_slice(utils::ConstSlice(values.data(), values.size(), false, nullptr), destination, pool);
        }
        inline Plaintext encode_polynomial_new(const std::vector<uint64_t>& values, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            encode_polynomial(values, destination, pool);
            return destination;
        }

        void decode_slice(const Plaintext& plaintext, utils::Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline utils::Array<uint64_t> decode_slice_new(const Plaintext& plaintext, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            utils::Array<uint64_t> destination = utils::Array<uint64_t>::create_uninitialized(slot_count(), on_device(), pool);
            decode_slice(plaintext, destination.reference(), pool);
            return destination;
        }

        void decode_slice_batched(const std::vector<const Plaintext*>& plaintexts, const std::vector<utils::Slice<uint64_t>>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline std::vector<utils::Array<uint64_t>> decode_slice_new_batched(const std::vector<const Plaintext*>& plaintexts, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<utils::Array<uint64_t>> destination; destination.reserve(plaintexts.size());
            for (size_t i = 0; i < plaintexts.size(); i++) 
                destination.push_back(utils::Array<uint64_t>::create_uninitialized(slot_count(), on_device(), pool));
            decode_slice_batched(plaintexts, batch_utils::rcollect_reference(destination), pool);
            return destination;
        }



        inline void decode(const Plaintext& plaintext, std::vector<uint64_t>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination.resize(slot_count());
            decode_slice(plaintext, utils::Slice(destination.data(), destination.size(), false, nullptr), pool);
        }
        inline std::vector<uint64_t> decode_new(const Plaintext& plaintext, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<uint64_t> destination(slot_count());
            decode(plaintext, destination, pool);
            return destination;
        }

        void decode_polynomial_slice(const Plaintext& plaintext, utils::Slice<uint64_t> destination) const;
        inline utils::Array<uint64_t> decode_polynomial_slice_new(const Plaintext& plaintext, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            utils::Array<uint64_t> destination = utils::Array<uint64_t>::create_uninitialized(plaintext.data().size(), on_device(), pool);
            decode_polynomial_slice(plaintext, destination.reference());
            return destination;
        }
        inline void decode_polynomial(const Plaintext& plaintext, std::vector<uint64_t>& destination) const {
            destination.resize(plaintext.data().size());
            decode_polynomial_slice(plaintext, utils::Slice(destination.data(), destination.size(), false, nullptr));
        }
        inline std::vector<uint64_t> decode_polynomial_new(const Plaintext& plaintext) const {
            std::vector<uint64_t> destination(plaintext.data().size());
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

        Plaintext decentralize_new(const Plaintext& plain, uint64_t correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void decentralize(const Plaintext& plain, Plaintext& destination, uint64_t correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = decentralize_new(plain, correction_factor, pool);
        }
        inline void decentralize_inplace(Plaintext& plain, uint64_t correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            plain = decentralize_new(plain, correction_factor, pool);
        }


    };

}