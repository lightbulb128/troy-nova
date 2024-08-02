#pragma once

#include "../utils/poly_small_mod.h"

namespace troy::utils::fgk::dyadic_convolute {

    void dyadic_convolute(
        ConstSlice<uint64_t> op1,
        ConstSlice<uint64_t> op2,
        size_t op1_pcount,
        size_t op2_pcount,
        ConstSlice<Modulus> moduli,
        size_t degree,
        Slice<uint64_t> result,
        MemoryPoolHandle pool
    );
     
    void dyadic_square(
        ConstSlice<uint64_t> op,
        ConstSlice<Modulus> moduli,
        size_t degree,
        Slice<uint64_t> result
    );

}