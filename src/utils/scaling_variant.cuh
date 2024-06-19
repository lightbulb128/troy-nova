#pragma once
#include "../plaintext.cuh"
#include "../context_data.cuh"

namespace troy { namespace scaling_variant {

    void add_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);
    void sub_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);

    void scale_up(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, bool add_to_destination, bool subtract);

    void multiply_add_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);
    void multiply_sub_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination);

}}