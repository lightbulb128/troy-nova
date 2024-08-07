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
    
    void dyadic_broadcast_product_ps(
        ConstSlice<uint64_t> op1,
        ConstSlice<uint64_t> op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli,
        Slice<uint64_t> result
    );
    
    inline void dyadic_broadcast_product_inplace_ps(
        Slice<uint64_t> op1,
        ConstSlice<uint64_t> op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli
    ) {
        dyadic_broadcast_product_ps(op1.as_const(), op2, op1_pcount, degree, moduli, op1);
    }

}