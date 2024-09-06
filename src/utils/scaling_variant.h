#pragma once
#include "../plaintext.h"
#include "../context_data.h"

namespace troy { namespace scaling_variant {

    // add_to_destination can be a nullptr, which means no addition.
    void scale_up(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count, utils::ConstSlice<uint64_t> add_to_destination, bool subtract);
    void scale_up_batched(
        const std::vector<const Plaintext*>& plain, ContextDataPointer context_data,
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count,
        const utils::ConstSliceVec<uint64_t>& add_to_destination, bool subtract,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );
    void centralize(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    void centralize_batched(
        const std::vector<const Plaintext*> plain, ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count, 
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );

    void scale_down(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    void decentralize(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, uint64_t correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    void multiply_add_plain_inplace(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count);
    void multiply_sub_plain_inplace(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count);
    
    void multiply_add_plain_inplace_batched(
        const std::vector<const Plaintext*>& plain, ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );
    void multiply_sub_plain_inplace_batched(
        const std::vector<const Plaintext*>& plain, ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );
    
    
    void multiply_add_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count);
    void multiply_sub_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count);

    // this interface is exposed because evaluator's transform_plain_to_ntt_no_fast_plain_lift use it.
    void multiply_plain_normal_no_fast_plain_lift(
        size_t plain_coeff_count, size_t coeff_modulus_size,
        utils::ConstSlice<uint64_t> plain, 
        utils::Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        utils::ConstSlice<uint64_t> plain_upper_half_increment
    );

}}