#pragma once
#include <cstdint>
#include <vector>
#include "../modulus.h"
#include "uint_small_mod.h"

namespace troy { namespace utils {

    const size_t TRY_PRIMITIVE_ROOT_NUM_ROUNDS = 100;

    std::vector<int> naf(int value);

    __host__ __device__
    inline uint64_t gcd(uint64_t x, uint64_t y) {
        while (y != 0) {
            uint64_t r = x % y;
            x = y;
            y = r;
        }
        return x;
    }


    /** Extended GCD:
    Returns (gcd, x, y) where gcd is the greatest common divisor of a and b.
    The numbers x, y are such that gcd = ax + by.
    */
    __host__ __device__
    inline void xgcd(uint64_t a, uint64_t b, uint64_t &gcd, int64_t &x, int64_t &y) {
        int64_t x0 = 1, y0 = 0, x1 = 0, y1 = 1;
        while (b != 0) {
            int64_t q = a / b;
            int64_t r = a % b;
            a = b;
            b = r;
            int64_t x2 = x0 - q * x1;
            int64_t y2 = y0 - q * y1;
            x0 = x1;
            y0 = y1;
            x1 = x2;
            y1 = y2;
        }
        gcd = a;
        x = x0;
        y = y0;
    }

    inline bool are_coprime(uint64_t a, uint64_t b) {
        return gcd(a, b) == 1;
    }

    __host__ __device__
    inline bool try_invert_uint64_mod_uint64(uint64_t value, uint64_t modulus, uint64_t& result) {
        if (value == 0) {
            return false;
        }
        uint64_t cd; int64_t a, b;
        xgcd(value, modulus, cd, a, b);
        if (cd != 1) {
            return false;
        } else if (a < 0) {
            result = static_cast<uint64_t>((static_cast<int64_t>(modulus) + a));
            return true;
        } else {
            result = static_cast<uint64_t>(a);
            return true;
        }
    }
    
    __host__ __device__
    inline bool try_invert_uint64_mod(const uint64_t& value, ConstPointer<Modulus> modulus, uint64_t& result) {
        return try_invert_uint64_mod_uint64(value, modulus->value(), result);
    }

    inline bool try_invert_uint64_mod(const uint64_t& value, const Modulus& modulus, uint64_t& result) {
        return try_invert_uint64_mod_uint64(value, modulus.value(), result);
    }

    std::vector<Modulus> get_primes(uint64_t factor, size_t bit_size, size_t count);

    inline Modulus get_prime(uint64_t factor, size_t bit_size) {
        return get_primes(factor, bit_size, 1)[0];
    }

    inline bool is_primitive_root(uint64_t root, uint64_t degree, const Modulus& modulus) {
        if (root == 0) {
            return false;
        } else {
            return utils::exponentiate_uint64_mod(root, degree >> 1, modulus)
                == modulus.value() - 1;
        }
    }

    bool try_primitive_root(uint64_t degree, const Modulus& modulus, uint64_t& destination);

    bool try_minimal_primitive_root(uint64_t degree, const Modulus& modulus, uint64_t& destination);


}}