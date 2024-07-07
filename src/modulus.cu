#include "modulus.h"
#include <stdexcept>
#include "utils/uint_small_mod.h"

namespace troy {
    
    void Modulus::set_value(uint64_t value) {
        if (value == 0) {
            this->bit_count_ = 0;
            this->value_ = 0;
            for (size_t i = 0; i < 3; i++) {
                this->const_ratio_[i] = 0;
            }
            this->is_prime_ = false;
        } else if ((value >> utils::HE_MOD_BIT_COUNT_MAX != 0) || (value == 1)) {
            throw std::invalid_argument("[Modulus::set_value] Value can be at most 61-bit and cannot be 1.");
        } else {
            this->value_ = value;
            this->bit_count_ = utils::get_significant_bit_count(value);
            uint64_t numerator[3]{0, 0, 1};
            uint64_t quotient[3]{0, 0, 0};
            utils::divide_uint192_uint64_inplace(
                utils::Slice<uint64_t>(numerator, 3, false, nullptr),
                value,
                utils::Slice<uint64_t>(quotient, 3, false, nullptr)
            );
            this->const_ratio_[0] = quotient[0];
            this->const_ratio_[1] = quotient[1];
            this->const_ratio_[2] = numerator[0];
            this->is_prime_ = utils::is_prime(*this);
        }
    }

    Modulus::Modulus(uint64_t value):
        value_(0),
        bit_count_(0),
        is_prime_(false) 
    {
        for (size_t i = 0; i < 3; i++) {
            this->const_ratio_[i] = 0;
        }
        set_value(value);
    }

}