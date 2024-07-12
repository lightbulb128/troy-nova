#pragma once
#include "constants.h"
#include <cstdint>
#include <cmath>
#include "box.h"

namespace troy { namespace utils {

    const double EPSILON = 1e-8;
    const size_t KERNEL_THREAD_COUNT = 256;

    inline bool same(bool a, bool b) {
        return a == b;
    }
    inline bool same(bool a, bool b, bool c) {
        return a == b && b == c;
    }
    inline bool same(bool a, bool b, bool c, bool d) {
        return a == b && b == c && c == d;
    }

    template <typename T0, typename T1>
    inline bool device_compatible(const T0& t0, const T1& t1) {
        if (t0.on_device() != t1.on_device()) return false;
        if (t0.on_device()) {
            return t0.device_index() == t1.device_index();
        } else {
            return true;
        }
    }

    template <typename T0, typename T1, typename T2>
    inline bool device_compatible(const T0& t0, const T1& t1, const T2& t2) {
        return device_compatible(t0, t1) && device_compatible(t0, t2);
    }

    template <typename T0, typename T1, typename T2, typename T3>
    inline bool device_compatible(const T0& t0, const T1& t1, const T2& t2, const T3& t3) {
        return device_compatible(t0, t1) && device_compatible(t0, t2) && device_compatible(t0, t3);
    }

    template <typename T>
    inline T ceil_div(T a, T b) {
        return (a + b - 1) / b;
    }

    __host__ __device__
    inline constexpr bool on_device() {
        #ifdef __CUDA_ARCH__
            return true;
        #else
            return false;
        #endif
    }

    template <typename T>
    __host__ __device__
    T max(T a, T b) {
        return a > b ? a : b;
    }

    template <typename T>
    __host__ __device__
    T min(T a, T b) {
        return a < b ? a : b;
    }

    template <typename T>
    __host__ __device__
    void swap(T& a, T& b) {
        T temp = a;
        a = b;
        b = temp;
    }

    __host__ __device__
    inline size_t get_significant_bit_count(uint64_t value) {
        if (value == 0) return 0;
        unsigned long result = 0;
        #ifdef __CUDA_ARCH__
            result = 63 - __clzll(value);
        #else
            result = 63 - __builtin_clzll(value);
        #endif
        return result + 1;
    }

    __host__ __device__
    inline size_t get_significant_bit_count_uint(ConstSlice<uint64_t> value) {
        size_t c = value.size() - 1;
        while (c > 0 && value[c] == 0) { c-= 1; }
        return get_significant_bit_count(value[c]) + c * 64;
    }

    __host__ __device__
    inline size_t get_significant_uint64_count_uint(ConstSlice<uint64_t> value) {
        size_t c = value.size();
        while (c > 0 && value[c-1] == 0) { c-=1; }
        return c;
    }

    __host__ __device__
    inline size_t get_nonzero_uint64_count_uint(ConstSlice<uint64_t> value) {
        size_t c = 0;
        for (size_t i = 0; i < value.size(); i++) {
            if (value[i] != 0) { c += 1; }
        }
        return c;
    }

    __host__ __device__
    inline int get_power_of_two(uint64_t value) {
        if (value == 0 || (value & (value - 1)) != 0) {
            return -1;
        } else {
            return get_significant_bit_count(value) - 1;
        }
    }

    __host__ __device__
    inline uint32_t reverse_bits_uint32(uint32_t operand) {
        operand = (((operand & uint32_t(0xaaaaaaaa)) >> 1) | ((operand & uint32_t(0x55555555)) << 1));
        operand = (((operand & uint32_t(0xcccccccc)) >> 2) | ((operand & uint32_t(0x33333333)) << 2));
        operand = (((operand & uint32_t(0xf0f0f0f0)) >> 4) | ((operand & uint32_t(0x0f0f0f0f)) << 4));
        operand = (((operand & uint32_t(0xff00ff00)) >> 8) | ((operand & uint32_t(0x00ff00ff)) << 8));
        return static_cast<uint32_t>(operand >> 16) | static_cast<uint32_t>(operand << 16);
    }

    __host__ __device__
    inline uint64_t reverse_bits_uint64(uint64_t operand) {
        return static_cast<uint64_t>(reverse_bits_uint32(static_cast<std::uint32_t>(operand >> 32))) |
                (static_cast<uint64_t>(reverse_bits_uint32(static_cast<std::uint32_t>(operand & uint64_t(0xFFFFFFFF)))) << 32);
    }

    __host__ __device__
    inline uint32_t reverse_bits_uint32(uint32_t operand, size_t bit_count) {
        return (bit_count == 0) 
            ? uint32_t(0)
            : reverse_bits_uint32(operand) >> (32 - bit_count);
    }

    __host__ __device__
    inline uint64_t reverse_bits_uint64(uint64_t operand, size_t bit_count) {
        return (bit_count == 0) 
            ? uint64_t(0)
            : reverse_bits_uint64(operand) >> (64 - bit_count);
    }

    __host__ __device__
    inline bool are_close_double(double a, double b) {
        #ifdef __CUDA_ARCH__
            double scale_factor = max(a, b);
            scale_factor = max(scale_factor, 1.0);
            return fabs(a - b) < scale_factor * EPSILON;
        #else
            double scale_factor = std::max(a, b);
            scale_factor = std::max(scale_factor, 1.0);
            return std::fabs(a - b) < scale_factor * EPSILON;
        #endif
    }

    __host__ __device__
    inline size_t hamming_weight(uint8_t x) {
        size_t t = x;
        t -= (t >> 1) & 0x55;
        t = (t & 0x33) + ((t >> 2) & 0x33);
        return (t + (t >> 4)) & 0x0F;
    }

    __host__ __device__
    inline void set_zero_uint(Slice<uint64_t> value) {
        for (size_t i = 0; i < value.size(); i++) {
            value[i] = 0;
        }
    }

    __host__ __device__
    inline bool is_zero_uint(ConstSlice<uint64_t> value) {
        for (size_t i = 0; i < value.size(); i++) {
            if (value[i] != 0) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__
    inline void set_bit_uint(Slice<uint64_t> value, size_t bit_index) {
        size_t word_index = bit_index / 64;
        size_t bit_offset = bit_index % 64;
        value[word_index] |= (uint64_t(1) << bit_offset);
    }

    __host__ __device__
    inline void set_uint(ConstSlice<uint64_t> value, size_t len, Slice<uint64_t> result) {
        for (size_t i = 0; i < len; i++) {
            result[i] = value[i];
        }
    }

    __host__ __device__
    inline uint8_t add_uint64_carry(uint64_t operand1, uint64_t operand2, uint8_t carry, uint64_t& result) {
        uint64_t temp = operand1 + operand2;
        result = temp + static_cast<uint64_t>(carry);
        return static_cast<uint8_t>(temp < operand2 || (~temp < static_cast<uint64_t>(carry)));
    }

    __host__ __device__
    inline uint8_t add_uint64(uint64_t operand1, uint64_t operand2, uint64_t& result) {
        result = operand1 + operand2;
        return static_cast<uint8_t>(result < operand1);
    }

    __host__ __device__
    inline uint8_t add_uint128(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, Slice<uint64_t> result) {
        uint8_t carry = add_uint64(operand1[0], operand2[0], result[0]);
        carry = add_uint64_carry(operand1[1], operand2[1], carry, result[1]);
        return carry;
    }

    __host__ __device__
    inline uint8_t add_uint128_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        return add_uint128(operand1.as_const(), operand2, operand1);
    }

    __host__ __device__
    inline uint8_t add_uint_carry(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, uint8_t carry, Slice<uint64_t> result) {
        uint8_t c = carry;
        for (size_t i = 0; i < result.size(); i++) {
            uint64_t temp_result = 0;
            c = add_uint64_carry(
                i < operand1.size() ? operand1[i] : 0, 
                i < operand2.size() ? operand2[i] : 0, 
                c, 
                temp_result
            );
            result[i] = temp_result;
        }
        return c;
    }

    __host__ __device__
    inline uint8_t add_uint_carry_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2, uint8_t carry) {
        return add_uint_carry(operand1.as_const(), operand2, carry, operand1);
    }

    __host__ __device__
    inline uint8_t add_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, Slice<uint64_t> result) {
        uint8_t carry = add_uint_carry(operand1, operand2, 0, result);
        return carry;
    }

    __host__ __device__
    inline uint8_t add_uint_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        uint8_t carry = add_uint_carry_inplace(operand1, operand2, 0);
        return carry;
    }

    __host__ __device__
    inline uint8_t add_uint_uint64(ConstSlice<uint64_t> operand1, uint64_t operand2, Slice<uint64_t> result) {
        uint8_t carry = add_uint64(operand1[0], operand2, result[0]);
        for (size_t i = 1; i < result.size(); i++) {
            uint64_t temp_result = 0;
            carry = add_uint64_carry(operand1[i], 0, carry, temp_result);
            result[i] = temp_result;
        }
        return carry;
    }

    __host__ __device__
    inline uint8_t add_uint_uint64_inplace(Slice<uint64_t> operand1, uint64_t operand2) {
        return add_uint_uint64(operand1.as_const(), operand2, operand1);
    }

    __host__ __device__
    inline uint8_t sub_uint64_borrow(uint64_t operand1, uint64_t operand2, uint8_t borrow, uint64_t& result) {
        uint64_t temp = operand1 - operand2;
        result = temp - static_cast<uint64_t>(borrow != 0);
        return static_cast<uint8_t>(temp > operand1 || (temp < static_cast<uint64_t>(borrow)));
    }

    __host__ __device__
    inline uint8_t sub_uint64(uint64_t operand1, uint64_t operand2, uint64_t& result) {
        result = operand1 - operand2;
        return static_cast<uint8_t>(result > operand1);
    }

    __host__ __device__
    inline uint8_t sub_uint_borrow(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, uint8_t borrow, Slice<uint64_t> result) {
        uint8_t b = borrow;
        for (size_t i = 0; i < result.size(); i++) {
            uint64_t temp_result = 0;
            b = sub_uint64_borrow(
                i < operand1.size() ? operand1[i] : 0, 
                i < operand2.size() ? operand2[i] : 0, 
                b, 
                temp_result
            );
            result[i] = temp_result;
        }
        return b;
    }

    __host__ __device__
    inline uint8_t sub_uint_borrow_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2, uint8_t borrow) {
        return sub_uint_borrow(operand1.as_const(), operand2, borrow, operand1);
    }

    __host__ __device__
    inline uint8_t sub_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, Slice<uint64_t> result) {
        uint8_t borrow = sub_uint_borrow(operand1, operand2, 0, result);
        return borrow;
    }

    __host__ __device__
    inline uint8_t sub_uint_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        uint8_t borrow = sub_uint_borrow_inplace(operand1, operand2, 0);
        return borrow;
    }

    __host__ __device__
    inline uint8_t sub_uint_u64(ConstSlice<uint64_t> operand1, uint64_t operand2, Slice<uint64_t> result) {
        uint8_t borrow = sub_uint64(operand1[0], operand2, result[0]);
        for (size_t i = 1; i < result.size(); i++) {
            uint64_t temp_result = 0;
            borrow = sub_uint64_borrow(operand1[i], 0, borrow, temp_result);
            result[i] = temp_result;
        }
        return borrow;
    }

    __host__ __device__
    inline uint8_t sub_uint_u64_inplace(Slice<uint64_t> operand1, uint64_t operand2) {
        return sub_uint_u64(operand1.as_const(), operand2, operand1);
    }

    __host__ __device__
    inline uint8_t increment_uint(ConstSlice<uint64_t> operand, Slice<uint64_t> result) {
        return add_uint_uint64(operand, 1, result);
    }

    __host__ __device__
    inline uint8_t increment_uint_inplace(Slice<uint64_t> operand) {
        return add_uint_uint64_inplace(operand, 1);
    }

    __host__ __device__
    inline uint8_t decrement_uint(ConstSlice<uint64_t> operand, Slice<uint64_t> result) {
        return sub_uint_u64(operand, 1, result);
    }

    __host__ __device__
    inline uint8_t decrement_uint_inplace(Slice<uint64_t> operand) {
        return sub_uint_u64_inplace(operand, 1);
    }

    __host__ __device__
    inline void negate_uint(ConstSlice<uint64_t> operand, Slice<uint64_t> result) {
        // negation is equivalent to inverting bits and adding 1.
        uint8_t carry = add_uint64(~operand[0], 1, result[0]);
        for (size_t i = 0; i < result.size(); i++) {
            uint64_t temp_result = 0;
            carry = add_uint64_carry(~operand[i], 0, carry, temp_result);
            result[i] = temp_result;
        }
    }

    __host__ __device__
    inline void negate_uint_inplace(Slice<uint64_t> operand) {
        return negate_uint(operand.as_const(), operand);
    }

    __host__ __device__
    inline void left_shift_uint(ConstSlice<uint64_t> operand, size_t shift_amount, size_t uint64_count, Slice<uint64_t> result) {
        size_t u64_shift_amount = shift_amount / 64;
        for (size_t i = 0; i < uint64_count - u64_shift_amount; i++) {
            result[uint64_count - i - 1] = operand[uint64_count - i - 1 - u64_shift_amount];
        }
        for (size_t i = uint64_count - u64_shift_amount; i < uint64_count; i++) {
            result[uint64_count - i - 1] = 0;
        }
        size_t bit_shift_amount = (shift_amount - u64_shift_amount * 64);
        if (bit_shift_amount > 0) {
            size_t neg_bit_shift_amount = 64 - bit_shift_amount;
            for (size_t i = uint64_count - 1; i >= 1; i--) {
                result[i] = (result[i] << bit_shift_amount) | (result[i-1] >> neg_bit_shift_amount);
            }
            result[0] = result[0] << bit_shift_amount;
        }
    }

    __host__ __device__
    inline void left_shift_uint_inplace(Slice<uint64_t> operand, size_t shift_amount, size_t uint64_count) {
        left_shift_uint(operand.as_const(), shift_amount, uint64_count, operand);
    }

    __host__ __device__
    inline void right_shift_uint(ConstSlice<uint64_t> operand, size_t shift_amount, size_t uint64_count, Slice<uint64_t> result) {
        size_t u64_shift_amount = shift_amount / 64;
        for (size_t i = 0; i < uint64_count - u64_shift_amount; i++) {
            result[i] = operand[i + u64_shift_amount];
        }
        for (size_t i = uint64_count - u64_shift_amount; i < uint64_count; i++) {
            result[i] = 0;
        }
        size_t bit_shift_amount = (shift_amount - u64_shift_amount * 64);
        if (bit_shift_amount > 0) {
            size_t neg_bit_shift_amount = 64 - bit_shift_amount;
            for (size_t i = 0; i < uint64_count - 1; i++) {
                result[i] = (result[i] >> bit_shift_amount) | (result[i+1] << neg_bit_shift_amount);
            }
            result[uint64_count - 1] = result[uint64_count - 1] >> bit_shift_amount;
        }
    }

    __host__ __device__
    inline void right_shift_uint_inplace(Slice<uint64_t> operand, size_t shift_amount, size_t uint64_count) {
        right_shift_uint(operand.as_const(), shift_amount, uint64_count, operand);
    }

    __host__ __device__
    inline void left_shift_uint128(ConstSlice<uint64_t> operand, size_t shift_amount, Slice<uint64_t> result) {
        if ((shift_amount & 64) > 0) {
            result[1] = operand[0]; result[0] = 0;
        } else {
            result[1] = operand[1]; result[0] = operand[0];
        }
        size_t bit_shift_amount = shift_amount & 63;
        if (bit_shift_amount > 0) {
            size_t neg_bit_shift_amount = 64 - bit_shift_amount;
            // warning: if bit_shift_amount == 0 this is incorrect
            result[1] = (result[1] << bit_shift_amount) | (result[0] >> neg_bit_shift_amount);
            result[0] = result[0] << bit_shift_amount;
        }
    }

    __host__ __device__
    inline void left_shift_uint128_inplace(Slice<uint64_t> operand, size_t shift_amount) {
        left_shift_uint128(operand.as_const(), shift_amount, operand);
    }

    __host__ __device__
    inline void right_shift_uint128(ConstSlice<uint64_t> operand, size_t shift_amount, Slice<uint64_t> result) {
        if ((shift_amount & 64) > 0) {
            result[0] = operand[1]; result[1] = 0;
        } else {
            result[0] = operand[0]; result[1] = operand[1];
        }
        size_t bit_shift_amount = shift_amount & 63;
        if (bit_shift_amount > 0) {
            size_t neg_bit_shift_amount = 64 - bit_shift_amount;
            // warning: if bit_shift_amount == 0 this is incorrect
            result[0] = (result[0] >> bit_shift_amount) | (result[1] << neg_bit_shift_amount);
            result[1] = result[1] >> bit_shift_amount;
        }
    }

    __host__ __device__
    inline void right_shift_uint128_inplace(Slice<uint64_t> operand, size_t shift_amount) {
        right_shift_uint128(operand.as_const(), shift_amount, operand);
    }

    __host__ __device__
    inline void left_shift_uint192(ConstSlice<uint64_t> operand, size_t shift_amount, Slice<uint64_t> result) {
        if ((shift_amount & 128) > 0) {
            result[2] = operand[0]; result[1] = 0; result[0] = 0;
        } else if ((shift_amount & 64) > 0) {
            result[2] = operand[1]; result[1] = operand[0]; result[0] = 0;
        } else {
            result[2] = operand[2]; result[1] = operand[1]; result[0] = operand[0];
        }
        size_t bit_shift_amount = shift_amount & 63;
        if (bit_shift_amount > 0) {
            size_t neg_bit_shift_amount = 64 - bit_shift_amount;
            // warning: if bit_shift_amount == 0 this is incorrect
            result[2] = (result[2] << bit_shift_amount) | (result[1] >> neg_bit_shift_amount);
            result[1] = (result[1] << bit_shift_amount) | (result[0] >> neg_bit_shift_amount);
            result[0] = result[0] << bit_shift_amount;
        }
    }

    __host__ __device__
    inline void left_shift_uint192_inplace(Slice<uint64_t> operand, size_t shift_amount) {
        left_shift_uint192(operand.as_const(), shift_amount, operand);
    }

    __host__ __device__
    inline void right_shift_uint192(ConstSlice<uint64_t> operand, size_t shift_amount, Slice<uint64_t> result) {
        if ((shift_amount & 128) > 0) {
            result[0] = operand[2]; result[1] = 0; result[2] = 0;
        } else if ((shift_amount & 64) > 0) {
            result[0] = operand[1]; result[1] = operand[2]; result[2] = 0;
        } else {
            result[0] = operand[0]; result[1] = operand[1]; result[2] = operand[2];
        }
        size_t bit_shift_amount = shift_amount & 63;
        if (bit_shift_amount > 0) {
            size_t neg_bit_shift_amount = 64 - bit_shift_amount;
            // warning: if bit_shift_amount == 0 this is incorrect
            result[0] = (result[0] >> bit_shift_amount) | (result[1] << neg_bit_shift_amount);
            result[1] = (result[1] >> bit_shift_amount) | (result[2] << neg_bit_shift_amount);
            result[2] = result[2] >> bit_shift_amount;
        }
    }

    __host__ __device__
    inline void right_shift_uint192_inplace(Slice<uint64_t> operand, size_t shift_amount) {
        right_shift_uint192(operand.as_const(), shift_amount, operand);
    }

    __host__ __device__
    inline void half_round_up_uint(ConstSlice<uint64_t> operand, Slice<uint64_t> result) {
        if (result.size() == 0) return;
        bool low_bit_set = (operand[0] & 1) != 0;
        size_t uint64_count = result.size();
        for (size_t i = 0; i < uint64_count - 1; i++) {
            result[i] = (operand[i] >> 1) | (operand[i+1] << 63);
        }
        result[uint64_count - 1] = operand[uint64_count - 1] >> 1;
        if (low_bit_set) {
            increment_uint_inplace(result);
        }
    }

    __host__ __device__
    inline void half_round_up_uint_inplace(Slice<uint64_t> operand) {
        half_round_up_uint(operand.as_const(), operand);
    }

    __host__ __device__
    inline void not_uint(ConstSlice<uint64_t> operand, Slice<uint64_t> result) {
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = ~operand[i];
        }
    }

    __host__ __device__
    inline void not_uint_inplace(Slice<uint64_t> operand) {
        not_uint(operand.as_const(), operand);
    }

    __host__ __device__
    inline void and_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, Slice<uint64_t> result) {
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = operand1[i] & operand2[i];
        }
    }

    __host__ __device__
    inline void and_uint_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        and_uint(operand1.as_const(), operand2, operand1);
    }

    __host__ __device__
    inline void or_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, Slice<uint64_t> result) {
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = operand1[i] | operand2[i];
        }
    }

    __host__ __device__
    inline void or_uint_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        or_uint(operand1.as_const(), operand2, operand1);
    }

    __host__ __device__
    inline void xor_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, Slice<uint64_t> result) {
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = operand1[i] ^ operand2[i];
        }
    }

    __host__ __device__
    inline void xor_uint_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        xor_uint(operand1.as_const(), operand2, operand1);
    }

    __host__ __device__
    inline void multiply_uint64_high_word(uint64_t operand1, uint64_t operand2, uint64_t& result_hw) {
        __uint128_t temp_result = static_cast<__uint128_t>(operand1) * static_cast<__uint128_t>(operand2);
        result_hw = static_cast<uint64_t>(temp_result >> 64);
    }

    __host__ __device__
    inline void multiply_uint64_uint64(uint64_t operand1, uint64_t operand2, Slice<uint64_t> result) {
        __uint128_t temp_result = static_cast<__uint128_t>(operand1) * static_cast<__uint128_t>(operand2);
        result[0] = static_cast<uint64_t>(temp_result);
        result[1] = static_cast<uint64_t>(temp_result >> 64);
    }

    __host__ __device__
    inline void multiply_uint_uint64(ConstSlice<uint64_t> operand1, uint64_t operand2, Slice<uint64_t> result) {
        if (operand1.size() == 0 || operand2 == 0) {
            set_zero_uint(result);
            return;
        }
        if (result.size() == 1) {
            result[0] = operand1[0] * operand2;
            return;
        }
        set_zero_uint(result);
        uint64_t carry = 0;
        size_t max_index = (operand1.size() < result.size()) ? operand1.size() : result.size();
        for (size_t i = 0; i < max_index; i++) {
            uint64_t temp_result[2] = { 0, 0 };
            multiply_uint64_uint64(operand1[i], operand2, Slice<uint64_t>(temp_result, 2, on_device(), nullptr));
            uint64_t temp = 0;
            carry = temp_result[1] + static_cast<uint64_t>(add_uint64_carry(temp_result[0], static_cast<uint64_t>(carry), 0, temp));
            result[i] = temp;
        }
        if (max_index < result.size()) {
            result[max_index] = carry;
        }
    }

    __host__ __device__
    inline void multiply_uint_uint64_inplace(Slice<uint64_t> operand1, uint64_t operand2) {
        multiply_uint_uint64(operand1.as_const(), operand2, operand1);
    }

    __host__ __device__
    inline void multiply_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, Slice<uint64_t> result) {
        if (operand1.size() == 0 || operand2.size() == 0) {return set_zero_uint(result);}
        if (result.size() == 1) {result[0] = operand1[0] * operand1[0]; return;}
        size_t operand1_uint64_count = get_significant_uint64_count_uint(operand1);
        size_t operand2_uint64_count = get_significant_uint64_count_uint(operand2);
        if (operand1_uint64_count == 1) {
            return multiply_uint_uint64(operand2, operand1[0], result);
        }
        if (operand2_uint64_count == 1) {
            return multiply_uint_uint64(operand1, operand2[0], result);
        }
        set_zero_uint(result);
        size_t operand1_index_max = operand1.size() < result.size() ? operand1.size() : result.size();
        for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++) {
            size_t operand2_index_max = (operand2.size() < result.size() - operand1_index) ? operand2.size() : result.size() - operand1_index;
            size_t carry = 0;
            for (size_t operand2_index = 0; operand2_index < operand2_index_max; operand2_index++) {
            uint64_t temp_result[2] = { 0, 0 };
                multiply_uint64_uint64(operand1[operand1_index], operand2[operand2_index], Slice<uint64_t>(temp_result, 2, on_device(), nullptr));
                carry = temp_result[1] + static_cast<uint64_t>(add_uint64_carry(temp_result[0], carry, 0, temp_result[0]));  // Wrapping add?
                uint64_t temp = 0;
                carry += static_cast<uint64_t>(add_uint64_carry(result[operand1_index + operand2_index], temp_result[0], 0, temp));  // Wrapping add?
                result[operand1_index + operand2_index] = temp;
            }
            if (operand1_index + operand2_index_max < result.size()) {
                result[operand1_index + operand2_index_max] = carry;
            }
        }
    }

    __host__ __device__
    inline size_t divide_round_up(size_t value, size_t divisor) {
        return (value + divisor - 1) / divisor;
    }

    __host__
    void divide_uint_inplace(Slice<uint64_t> numerator, ConstSlice<uint64_t> denominator, Slice<uint64_t> quotient, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    __host__
    inline void divide_uint(ConstSlice<uint64_t> numerator, ConstSlice<uint64_t> denominator, Slice<uint64_t> quotient, Slice<uint64_t> remainder, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        set_uint(numerator, remainder.size(), remainder);
        divide_uint_inplace(remainder, denominator, quotient, pool);
    }

    __host__ __device__
    inline void divide_uint128_uint64_inplace(Slice<uint64_t> numerator, uint64_t denominator, Slice<uint64_t> quotient) {
        __uint128_t n = (static_cast<__uint128_t>(numerator[1]) << 64) | static_cast<__uint128_t>(numerator[0]);
        __uint128_t d = static_cast<__uint128_t>(denominator);
        __uint128_t q = n / d;
        n -= q * d;
        quotient[0] = static_cast<uint64_t>(q);
        quotient[1] = static_cast<uint64_t>(q >> 64);
        numerator[0] = static_cast<uint64_t>(n);
        numerator[1] = static_cast<uint64_t>(n >> 64);
    }

    __host__
    void divide_uint192_uint64_inplace(Slice<uint64_t> numerator, uint64_t denominator, Slice<uint64_t> quotient, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    __host__ __device__
    inline int compare_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        size_t n = max(operand1.size(), operand2.size());
        for (int i = static_cast<int>(n - 1); i >= 0; i--) {
            uint64_t a = static_cast<size_t>(i) < operand1.size() ? operand1[i] : 0;
            uint64_t b = static_cast<size_t>(i) < operand2.size() ? operand2[i] : 0;
            if (a > b) return 1;
            if (a < b) return -1;
        }
        return 0;
    }

    __host__ __device__
    inline bool is_greater_than_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        return compare_uint(operand1, operand2) > 0;
    }

    __host__ __device__
    inline bool is_greater_or_equal_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        return compare_uint(operand1, operand2) >= 0;
    }

    __host__ __device__
    inline bool is_less_than_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        return compare_uint(operand1, operand2) < 0;
    }

    __host__ __device__
    inline bool is_less_or_equal_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        return compare_uint(operand1, operand2) <= 0;
    }

    __host__ __device__
    inline bool is_equal_uint(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2) {
        return compare_uint(operand1, operand2) == 0;
    }

    __host__ __device__
    inline void add_uint_mod(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, ConstSlice<uint64_t> modulus, Slice<uint64_t> result) {
        uint8_t carry = add_uint(operand1, operand2, result);
        if (carry || is_greater_or_equal_uint(result.as_const(), modulus)) {
            sub_uint_inplace(result, modulus);
        }
    }

    __host__ __device__
    inline void add_uint_mod_inplace(Slice<uint64_t> operand1, ConstSlice<uint64_t> operand2, ConstSlice<uint64_t> modulus) {
        add_uint_mod(operand1.as_const(), operand2, modulus, operand1);
    }

    __host__ __device__
    inline void sub_uint_mod(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, ConstSlice<uint64_t> modulus, Slice<uint64_t> result) {
        if (sub_uint(operand1, operand2, result) != 0) {
            add_uint_inplace(result, modulus);
        }
    }

    __host__
    inline void multiply_many_uint64(ConstSlice<uint64_t> operands, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
        if (operands.size() == 0) {
            return;
        }
        set_zero_uint(result);
        result[0] = operands[0];
        Array<uint64_t> temp_mpi(operands.size(), on_device(), pool);
        for (size_t i = 1; i < operands.size(); i++) {
            multiply_uint_uint64(result.as_const(), operands[i], temp_mpi.reference());
            set_uint(temp_mpi.const_reference(), i + 1, result);
        }
    }

}}