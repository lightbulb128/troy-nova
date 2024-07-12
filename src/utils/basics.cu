#include "basics.h"

namespace troy {namespace utils {


    __host__
    void divide_uint_inplace(Slice<uint64_t> numerator, ConstSlice<uint64_t> denominator, Slice<uint64_t> quotient, MemoryPoolHandle pool) {
        // assert!(numerator.len() == denominator.len());
        // assert!(numerator.len() == quotient.len());
        size_t uint64_count = quotient.size();
        if (uint64_count == 0) {return;}
        set_zero_uint(quotient);
        // Determine significant bits in numerator and denominator.
        size_t numerator_bits = get_significant_bit_count_uint(numerator.as_const());
        size_t denominator_bits = get_significant_bit_count_uint(denominator);
        // If numerator has fewer bits than denominator, then done.
        if (numerator_bits < denominator_bits) {return;}
        // Only perform computation up to last non-zero uint64s.
        uint64_count = divide_round_up(numerator_bits, 64);
        // Handle fast case.
        if (uint64_count == 1) {
            quotient[0] = numerator[0] / denominator[0];
            numerator[0] -= quotient[0] * denominator[0];
            return;
        }
        // Create temporary space to store mutable copy of denominator.
        Array<uint64_t> shifted_denominator(uint64_count, on_device(), pool);
        // Shift denominator to bring MSB in alignment with MSB of numerator.
        size_t denominator_shift = numerator_bits - denominator_bits;
        left_shift_uint(denominator, denominator_shift, uint64_count, shifted_denominator.reference());
        Array<uint64_t> difference(uint64_count, on_device(), pool);
        denominator_bits += denominator_shift;
        // Perform bit-wise division algorithm.
        size_t remaining_shifts = denominator_shift;
        while (numerator_bits == denominator_bits) {
            // NOTE: MSBs of numerator and denominator are aligned.
            // Even though MSB of numerator and denominator are aligned,
            // still possible numerator < shifted_denominator.
            if (sub_uint(numerator.as_const(), shifted_denominator.const_reference(), difference.reference()) != 0) {
                if (remaining_shifts == 0) {break;}
                add_uint_inplace(difference.reference(), numerator.as_const());
                left_shift_uint_inplace(quotient, 1, uint64_count);
                remaining_shifts -= 1;
            }
            quotient[0] |= 1;
            numerator_bits = get_significant_bit_count_uint(difference.const_reference());
            size_t numerator_shift = denominator_bits - numerator_bits;
            if (numerator_shift > remaining_shifts) {
                numerator_shift = remaining_shifts;
            }
            if (numerator_bits > 0) {
                left_shift_uint(difference.const_reference(), numerator_shift, uint64_count, numerator);
                numerator_bits += numerator_shift;
            } else {
                set_zero_uint(numerator);
            }
            left_shift_uint_inplace(quotient, numerator_shift, uint64_count);
            remaining_shifts -= numerator_shift;
        }
        if (numerator_bits > 0) {
            right_shift_uint_inplace(numerator, denominator_shift, uint64_count);
        }
    }
    
    __host__
    void divide_uint192_uint64_inplace(Slice<uint64_t> numerator, uint64_t denominator, Slice<uint64_t> quotient, MemoryPoolHandle pool) {
        quotient[0] = 0; quotient[1] = 0; quotient[2] = 0;
        // Determine significant bits in numerator and denominator.
        size_t numerator_bits = get_significant_bit_count_uint(numerator.as_const());
        size_t denominator_bits = get_significant_bit_count(denominator);
        // If numerator has fewer bits than denominator, then done.
        if (numerator_bits < denominator_bits) {return;}
        // Only perform computation up to last non-zero uint64s.
        size_t uint64_count = divide_round_up(numerator_bits, 64);
        // Handle fast case.
        if (uint64_count == 1) {
            quotient[0] = numerator[0] / denominator;
            numerator[0] -= quotient[0] * denominator;
            return;
        }
        // Create temporary space to store mutable copy of denominator.
        Array<uint64_t> shifted_denominator(uint64_count, on_device(), pool);
        shifted_denominator[0] = denominator;
        // Shift denominator to bring MSB in alignment with MSB of numerator.
        size_t denominator_shift = numerator_bits - denominator_bits;
        left_shift_uint192_inplace(shifted_denominator.reference(), denominator_shift);
        Array<uint64_t> difference(uint64_count, on_device(), pool);
        denominator_bits += denominator_shift;
        // Perform bit-wise division algorithm.
        size_t remaining_shifts = denominator_shift;
        while (numerator_bits == denominator_bits) {
            // NOTE: MSBs of numerator and denominator are aligned.
            // Even though MSB of numerator and denominator are aligned,
            // still possible numerator < shifted_denominator.
            if (sub_uint(numerator.as_const(), shifted_denominator.const_reference(), difference.reference()) != 0) {
                if (remaining_shifts == 0) {break;}
                add_uint_inplace(difference.reference(), numerator.as_const());
                left_shift_uint192_inplace(quotient, 1);
                remaining_shifts -= 1;
            }
            quotient[0] |= 1;
            numerator_bits = get_significant_bit_count_uint(difference.const_reference());
            size_t numerator_shift = denominator_bits - numerator_bits;
            if (numerator_shift > remaining_shifts) {
                numerator_shift = remaining_shifts;
            }
            if (numerator_bits > 0) {
                left_shift_uint192(difference.const_reference(), numerator_shift, numerator);
                numerator_bits += numerator_shift;
            } else {
                set_zero_uint(numerator);
            }
            left_shift_uint192_inplace(quotient, numerator_shift);
            remaining_shifts -= numerator_shift;
        }
        if (numerator_bits > 0) {
            right_shift_uint192_inplace(numerator, denominator_shift);
        }
    }

}}