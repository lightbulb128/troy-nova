#pragma once

#include "../utils/poly_small_mod.h"
#include "../batch_utils.h"

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
    
    void dyadic_broadcast_product_accumulate_ps(
        ConstSlice<uint64_t> op1,
        ConstSlice<uint64_t> op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli,
        Slice<uint64_t> result
    );

    void dyadic_broadcast_product_bps(
        const ConstSliceVec<uint64_t>& op1,
        const ConstSliceVec<uint64_t>& op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli,
        const SliceVec<uint64_t>& result,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
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

    inline void dyadic_broadcast_product_inplace_bps(
        const SliceVec<uint64_t>& op1,
        const ConstSliceVec<uint64_t>& op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        dyadic_broadcast_product_bps(batch_utils::rcollect_as_const(op1), op2, op1_pcount, degree, moduli, op1, pool);
    }

    void dyadic_broadcast_product_accumulate_bps(
        const ConstSliceVec<uint64_t>& op1,
        const ConstSliceVec<uint64_t>& op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli,
        const SliceVec<uint64_t>& result,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ); 

}