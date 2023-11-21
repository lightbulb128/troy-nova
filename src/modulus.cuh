#pragma once
#include <cstdint>
#include "utils/box.cuh"
#include "utils/basics.cuh"

namespace troy {

    class Modulus {

        uint64_t value_;
        uint64_t const_ratio_[3];
        size_t bit_count_;
        bool is_prime_;

        void set_value(uint64_t value);

    public:

        Modulus(uint64_t value = 0);

        __host__ __device__
        inline uint64_t reduce(uint64_t input) const {
            using utils::Slice;
            using utils::ConstSlice;
            // Reduces input using base 2^64 Barrett reduction
            // floor(2^64 / mod) == floor( floor(2^128 / mod) )
            uint64_t tmp[2];
            Slice<uint64_t> tmp_slice(tmp, 2, utils::on_device());
            ConstSlice<uint64_t> const_ratio = this->const_ratio();
            utils::multiply_uint64_high_word(input, const_ratio[1], tmp[1]);

            // Barrett subtraction
            tmp[0] = input - tmp[1] * this->value();

            // One more subtraction is enough
            if (tmp[0] >= this->value()) {
                return tmp[0] - this->value();
            } else {
                return tmp[0];
            }
        }

        __host__ __device__
        inline uint64_t reduce_uint128(utils::ConstSlice<uint64_t> input) const {
            using utils::Slice;
            using utils::ConstSlice;
            // Reduces input using base 2^64 Barrett reduction
            // input allocation size must be 128 bits
            uint64_t tmp1 = 0;
            uint64_t tmp2[2];
            Slice<uint64_t> tmp2_slice(tmp2, 2, utils::on_device());
            uint64_t tmp3 = 0;
            uint64_t carry = 0;
            ConstSlice<uint64_t> const_ratio = this->const_ratio();

            // Multiply input and const_ratio
            // Round 1
            utils::multiply_uint64_high_word(input[0], const_ratio[0], carry);
            utils::multiply_uint64_uint64(input[0], const_ratio[1], tmp2_slice);
            tmp3 = tmp2[1] + static_cast<uint64_t>(utils::add_uint64(tmp2[0], carry, tmp1));
            
            // Round 2
            utils::multiply_uint64_uint64(input[1], const_ratio[0], tmp2_slice);
            carry = tmp2[1] + static_cast<uint64_t>(utils::add_uint64(tmp1, tmp2[0], tmp1));

            // This is all we care about
            tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

            // Barrett subtraction
            tmp3 = input[0] - tmp1 * this->value();

            // One more subtraction is enough
            if (tmp3 >= this->value()) {
                return tmp3 - this->value();
            } else {
                return tmp3;
            }
        }

        __host__ __device__
        inline uint64_t reduce_uint128(__uint128_t value) const {
            utils::ConstSlice<uint64_t> value_slice(reinterpret_cast<uint64_t*>(&value), 2, utils::on_device());
            return reduce_uint128(value_slice);
        }

        __host__ __device__
        inline uint64_t reduce_mul_uint64(uint64_t operand1, uint64_t operand2) {
            uint64_t tmp[2];
            utils::Slice<uint64_t> tmp_slice(tmp, 2, utils::on_device());
            utils::multiply_uint64_uint64(operand1, operand2, tmp_slice);
            return reduce_uint128(tmp_slice.as_const());
        }

        __host__ __device__
        inline utils::ConstSlice<uint64_t> const_ratio() const {
            return utils::ConstSlice<uint64_t>(const_ratio_, 3, utils::on_device());
        }

        __host__ __device__
        inline uint64_t value() const {
            return value_;
        }

        __host__ __device__
        inline uint64_t uint64_count() const {
            return 1;
        }

        __host__ __device__
        inline size_t bit_count() const {
            return bit_count_;
        }

        __host__ __device__
        inline bool is_prime() const {
            return is_prime_;
        }

        __host__ __device__
        inline bool is_zero() const {
            return value_ == 0;
        }

    };

}