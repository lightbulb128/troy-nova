#pragma once
#include <cstdint>
#include "../modulus.h"

namespace troy { namespace utils {

    const size_t IS_PRIME_TEST_COUNT = 40;

    __host__ __device__
    inline uint64_t increment_uint64_mod(uint64_t operand, const Modulus& modulus) {
        operand += 1;
        if (operand >= modulus.value()) {
            return operand - modulus.value();
        } else {
            return operand;
        }
    }

    __host__ __device__
    inline uint64_t decrement_uint64_mod(uint64_t operand, const Modulus& modulus) {
        if (operand == 0) {
            return modulus.value() - 1;
        } else {
            return operand - 1;
        }
    }

    __host__ __device__
    inline uint64_t negate_uint64_mod(uint64_t operand, const Modulus& modulus) {
        if (operand == 0) {
            return 0;
        } else {
            return modulus.value() - operand;
        }
    }

    __host__ __device__
    inline uint64_t divide2_uint64_mod(uint64_t operand, const Modulus& modulus) {
        if ((operand & 1) > 0) {
            uint64_t temp = 0;
            uint8_t carry = utils::add_uint64(operand, modulus.value(), temp);
            operand = temp >> 1;
            if (carry > 0) {
                return operand | (static_cast<uint64_t>(1) << 63);
            } else {
                return operand;
            }
        } else {
            return operand >> 1;
        }
    }

    __host__ __device__
    inline uint64_t add_uint64_mod(uint64_t operand1, uint64_t operand2, const Modulus& modulus) {
        operand1 += operand2;
        if (operand1 >= modulus.value()) {
            return operand1 - modulus.value();
        } else {
            return operand1;
        }
    }

    __host__ __device__
    inline uint64_t sub_uint64_mod(uint64_t operand1, uint64_t operand2, const Modulus& modulus) {
        uint64_t temp = 0;
        uint8_t borrow = utils::sub_uint64(operand1, operand2, temp);
        if (borrow > 0) {
            return temp + modulus.value();
        } else {
            return temp;
        }
    }

    __host__ __device__
    inline uint64_t barrett_reduce_uint128(ConstSlice<uint64_t> input, const Modulus& modulus) {
        return modulus.reduce_uint128_limbs(input);
    }

    __host__ __device__
    inline uint64_t barrett_reduce_uint64(uint64_t input, const Modulus& modulus) {
        return modulus.reduce(input);
    }

    __host__ __device__
    inline uint64_t multiply_uint64_mod(uint64_t operand1, uint64_t operand2, const Modulus& modulus) {
        uint64_t temp[2];
        Slice<uint64_t> temp_slice(temp, 2, on_device(), nullptr);
        utils::multiply_uint64_uint64(operand1, operand2, temp_slice);
        return modulus.reduce_uint128_limbs(temp_slice.as_const());
    }

    struct MultiplyUint64Operand {

        uint64_t operand;
        uint64_t quotient;

        __host__ __device__
        inline void set_quotient(const Modulus& modulus) {
            uint64_t wide_quotient[2]{0, 0};
            uint64_t wide_coeff[2]{0, this->operand};
            divide_uint128_uint64_inplace(
                Slice<uint64_t>(wide_coeff, 2, on_device(), nullptr),
                modulus.value(),
                Slice<uint64_t>(wide_quotient, 2, on_device(), nullptr)
            );
            this->quotient = wide_quotient[0];
        }

        __host__ __device__
        inline MultiplyUint64Operand(uint64_t operand, const Modulus& modulus) {
            this->operand = operand;
            this->quotient = 0;
            this->set_quotient(modulus);
        }

        __host__ __device__
        inline MultiplyUint64Operand() {
            this->operand = 0;
            this->quotient = 0;
        }

    };
    
    inline std::ostream& operator<<(std::ostream& os, const MultiplyUint64Operand& operand) {
        os << "Mo(" << operand.operand << ", " << operand.quotient << ")";
        return os;
    }

    __host__ __device__
    inline uint64_t multiply_uint64operand_mod(uint64_t x, const MultiplyUint64Operand& y, const Modulus& modulus) {
        uint64_t tmp1 = 0;
        uint64_t p = modulus.value();
        utils::multiply_uint64_high_word(x, y.quotient, tmp1);
        uint64_t tmp2 = y.operand * x - tmp1 * p;
        if (tmp2 >= p) {
            tmp2 -= p;
        }
        return tmp2;
    }

    __host__ __device__
    inline uint64_t multiply_uint64operand_mod_lazy(uint64_t y, const MultiplyUint64Operand& x, const Modulus& modulus) {
        uint64_t tmp1 = 0;
        uint64_t p = modulus.value();
        utils::multiply_uint64_high_word(y, x.quotient, tmp1);
        uint64_t tmp2 = x.operand * y - tmp1 * p;
        return tmp2;
    }

    __host__ __device__
    inline void modulo_uint_inplace(Slice<uint64_t> value, const Modulus& modulus) {
        if (value.size() == 1) {
            if (value[0] < modulus.value()) {
                return;
            } else {
                value[0] = modulus.reduce(value[0]);
            }
        }
        for (int i = static_cast<int>(value.size()) - 2; i >= 0; i--) {
            value[i] = modulus.reduce_uint128_limbs(value.const_slice(i, i + 2));
            value[i + 1] = 0;
        }
    }

    __host__ __device__
    inline uint64_t modulo_uint(ConstSlice<uint64_t> value, const Modulus& modulus) {
        if (value.size() == 1) {
            if (value[0] < modulus.value()) {
                return value[0];
            } else {
                return modulus.reduce(value[0]);
            }
        } else {
            uint64_t temp[2]; temp[0] = 0; temp[1] = value[value.size() - 1];
            ConstSlice<uint64_t> temp_slice(temp, 2, on_device(), nullptr);
            for (int i = static_cast<int>(value.size()) - 2; i >= 0; i--) {
                temp[0] = value[i];
                temp[1] = modulus.reduce_uint128_limbs(temp_slice);
            }
            return temp[1];
        }
    }

    /**
    Returns (operand1 * operand2) + operand3 mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    __host__ __device__
    inline uint64_t multiply_add_uint64_mod(uint64_t operand1, uint64_t operand2, uint64_t operand3, const Modulus& modulus) {
        // lazy reduction
        uint64_t temp[2];
        Slice<uint64_t> temp_slice(temp, 2, on_device(), nullptr);
        utils::multiply_uint64_uint64(operand1, operand2, temp_slice);
        temp[1] += static_cast<uint64_t>(utils::add_uint64(temp[0], operand3, temp[0]));
        return modulus.reduce_uint128_limbs(temp_slice.as_const());
    }

    /**
    Returns (operand1 * operand2) + operand3 mod modulus.
    Correctness: Follows the condition of multiply_uint_mod.
    */
    __host__ __device__
    inline uint64_t multiply_uint64operand_add_uint64_mod(
        uint64_t operand1,
        const MultiplyUint64Operand& operand2,
        uint64_t operand3,
        const Modulus& modulus
    ) {
        return utils::add_uint64_mod(
            utils::multiply_uint64operand_mod(operand1, operand2, modulus),
            modulus.reduce(operand3),
            modulus
        );
    }

    /**
    Returns operand^exponent mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    __host__ __device__
    inline uint64_t exponentiate_uint64_mod(uint64_t operand, uint64_t exponent, const Modulus& modulus) {
        if (exponent == 0) {return 1;}
        if (exponent == 1) {return operand;}
        uint64_t power = operand; uint64_t product; uint64_t intermediate = 1;
        while (true) {
            if ((exponent & 1) > 0) {
                product = utils::multiply_uint64_mod(power, intermediate, modulus);
                utils::swap(product, intermediate);
            }
            exponent >>= 1;
            if (exponent == 0) {break;}
            product = utils::multiply_uint64_mod(power, power, modulus);
            utils::swap(product, power);
        }
        return intermediate;
    }

    
    /**
    Computes numerator = numerator mod modulus, quotient = numerator / modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    __host__
    void divide_uint_mod_inplace(Slice<uint64_t> numerator, const Modulus& modulus, Slice<uint64_t> quotient, MemoryPoolHandle pool);

    /**
    Computes <operand1, operand2> mod modulus.
    Correctness: Follows the condition of barrett_reduce_128.
    */
    __host__ __device__
    inline uint64_t dot_product_mod(ConstSlice<uint64_t> operand1, ConstSlice<uint64_t> operand2, const Modulus& modulus) {
        uint64_t accumulator[2]{0, 0};
        Slice<uint64_t> accumulator_slice(accumulator, 2, on_device(), nullptr);
        uint64_t qword[2]{0, 0};
        Slice<uint64_t> qword_slice(qword, 2, on_device(), nullptr);
        for (size_t i = 0; i < operand1.size(); i++) {
            utils::multiply_uint64_uint64(operand1[i], operand2[i], qword_slice);
            utils::add_uint128_inplace(accumulator_slice, qword_slice.as_const());
        }
        return modulus.reduce_uint128_limbs(accumulator_slice.as_const());
    }

    __host__
    inline bool is_prime(const Modulus& modulus) {
        uint64_t value = modulus.value();
        // First check the simplest cases.
        if (value < 2) {return false;}
        if (value == 2) {return true;}
        if (value % 2 == 0) {return false;}
        if (value == 3) {return true;}
        if (value % 3 == 0) {return false;}
        if (value == 5) {return true;}
        if (value % 5 == 0) {return false;}
        if (value == 7) {return true;}
        if (value % 7 == 0) {return false;}
        if (value == 11) {return true;}
        if (value % 11 == 0) {return false;}
        if (value == 13) {return true;}
        if (value % 13 == 0) {return false;}
        // Second, Miller-Rabin test.
        // Find r and odd d that satisfy value = 2^r * d + 1.
        uint64_t d = value - 1;
        uint64_t r = 0;
        while ((d & 1) == 0) {d >>= 1; r += 1;}
        if (r == 0) {return false;}
        // 1) Pick a = 2, check a^(value - 1).
        // 2) Pick a randomly from [3, value - 1], check a^(value - 1).
        // 3) Repeat 2) for another num_rounds - 2 times.
        for (size_t i = 0; i < IS_PRIME_TEST_COUNT; i++) {
            uint64_t a = i==0 ? 2 : (rand() % (value - 3) + 3);
            uint64_t x = utils::exponentiate_uint64_mod(a, d, modulus);
            if (x == 1 || x == value - 1) {continue;}
            uint64_t count = 0;
            while (true) {
                x = utils::multiply_uint64_mod(x, x, modulus);
                count += 1;
                if ((x == value - 1) || (count >= r - 1)) {break;}
            }
            if (x != value - 1) {return false;}
        };
        return true;
    }

}}