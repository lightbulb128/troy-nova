#include "uint_small_mod.h"

namespace troy {namespace utils {
    
    __host__
    void divide_uint_mod_inplace(Slice<uint64_t> numerator, const Modulus& modulus, Slice<uint64_t> quotient, MemoryPoolHandle pool) {
        size_t uint64_count = quotient.size();
        if (uint64_count == 2) {
            utils::divide_uint128_uint64_inplace(numerator, modulus.value(), quotient); return;
        } else if (uint64_count == 1) {
            numerator[0] = barrett_reduce_uint64(numerator[0], modulus);
            quotient[0] = numerator[0] / modulus.value(); return;
        } else {
            // If uint64_count > 2.
            // x = numerator = x1 * 2^128 + x2.
            // 2^128 = A*value + B.
            Array<uint64_t> x1(uint64_count - 2, on_device(), pool);
            Array<uint64_t> x2(2, on_device(), pool);
            Array<uint64_t> quot(uint64_count, on_device(), pool);
            Array<uint64_t> rem(uint64_count, on_device(), pool);
            utils::set_uint(numerator.const_slice(2, numerator.size()), uint64_count - 2, x1.reference());
            utils::set_uint(numerator.const_slice(0, 2), 2, x2.reference()); // x2 = (num) % 2^128.

            utils::multiply_uint(x1.const_reference(), modulus.const_ratio().const_slice(0, 2), quot.reference());
            utils::multiply_uint_uint64(x1.const_reference(), modulus.const_ratio()[2], rem.reference());
            utils::add_uint_inplace(rem.reference(), x2.const_reference());

            size_t remainder_uint64_count = utils::get_significant_uint64_count_uint(rem.const_reference());
            divide_uint_mod_inplace(rem.reference(), modulus, quotient.slice(0, remainder_uint64_count), pool);
            utils::add_uint_inplace(quotient, quot.const_reference());
            numerator[0] = rem[0];
            return;
        }
    }

}}