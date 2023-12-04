#pragma once
#include <cstdint>

namespace troy {namespace utils {

    const std::size_t HE_MOD_BIT_COUNT_MAX = 61;

    const std::size_t HE_COEFF_MOD_COUNT_MAX = 64;
    const std::size_t HE_COEFF_MOD_COUNT_MIN = 1;

    const std::size_t HE_POLY_MOD_DEGREE_MAX = 131072;
    const std::size_t HE_POLY_MOD_DEGREE_MIN = 2;

    const std::size_t HE_INTERNAL_MOD_BIT_COUNT = 61;

    const std::size_t HE_USER_MOD_BIT_COUNT_MAX = 60;
    const std::size_t HE_USER_MOD_BIT_COUNT_MIN = 2;

    const std::size_t HE_PLAIN_MOD_BIT_COUNT_MAX = HE_USER_MOD_BIT_COUNT_MAX;
    const std::size_t HE_PLAIN_MOD_BIT_COUNT_MIN = HE_USER_MOD_BIT_COUNT_MIN;

    const std::size_t HE_CIPHERTEXT_SIZE_MIN = 2;
    const std::size_t HE_CIPHERTEXT_SIZE_MAX = 16;

}}