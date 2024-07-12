#pragma once
#include "../plaintext.h"
#include "../context_data.h"

namespace troy { namespace scaling_variant {

    void add_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);
    void sub_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);

    void scale_up(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, bool add_to_destination, bool subtract);
    void centralize(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    void multiply_add_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);
    void multiply_sub_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);

    // this interface is exposed because evaluator's transform_plain_to_ntt_no_fast_plain_lift use it.
    void multiply_plain_normal_no_fast_plain_lift(
        size_t plain_coeff_count, size_t coeff_modulus_size,
        utils::ConstSlice<uint64_t> plain, 
        utils::Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        utils::ConstSlice<uint64_t> plain_upper_half_increment
    );

}}