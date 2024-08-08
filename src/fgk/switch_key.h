#pragma once

#include "../utils/basics.h"
#include "../evaluator.h"
#include "../utils/polynomial_buffer.h"
#include <cassert>

namespace troy::utils::fgk::switch_key {

    void set_accumulate(
        size_t decomp_modulus_size, size_t coeff_count, ConstSlice<uint64_t> target_intt, Buffer<uint64_t>& temp_ntt, ConstSlice<Modulus> key_modulus
    );

    void accumulate_products(
        size_t decomp_modulus_size, size_t key_component_count, size_t coeff_count, 
        ConstSlice<uint64_t> temp_ntt,
        ConstSlice<Modulus> key_moduli,
        ConstSlice<const uint64_t*> key_vector,
        Slice<uint64_t> poly_prod
    );

}