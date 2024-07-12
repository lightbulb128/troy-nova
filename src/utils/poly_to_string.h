#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include <ostream>
#include "basics.h"

namespace troy { namespace utils {

    std::string uint_to_hex_string(const uint64_t *value, size_t uint64_count);

    std::string uint_to_dec_string(const uint64_t *value, size_t uint64_count);

    // Input must be host memory!
    inline std::string poly_to_hex_string(
        const std::uint64_t *value, std::size_t coeff_count, std::size_t coeff_uint64_count)
    {
        // First check if there is anything to print
        if (!coeff_count || !coeff_uint64_count)
        {
            return "0";
        }

        std::ostringstream result;
        bool empty = true;
        value += (coeff_count - 1) * coeff_uint64_count;
        while (coeff_count--)
        {
            if (is_zero_uint(ConstSlice(value, coeff_uint64_count, false, nullptr)))
            {
                value -= coeff_uint64_count;
                continue;
            }
            if (!empty)
            {
                result << " + ";
            }
            result << uint_to_hex_string(value, coeff_uint64_count);
            if (coeff_count)
            {
                result << "x^" << coeff_count;
            }
            empty = false;
            value -= coeff_uint64_count;
        }
        if (empty)
        {
            result << "0";
        }
        return result.str();
    }

    // Input must be host memory!
    inline std::string poly_to_dec_string(
        const std::uint64_t *value, std::size_t coeff_count, std::size_t coeff_uint64_count)
    {
        // First check if there is anything to print
        if (!coeff_count || !coeff_uint64_count)
        {
            return "0";
        }

        std::ostringstream result;
        bool empty = true;
        value += coeff_count - 1;
        while (coeff_count--)
        {
            if (is_zero_uint(ConstSlice(value, coeff_uint64_count, false, nullptr)))
            {
                value -= coeff_uint64_count;
                continue;
            }
            if (!empty)
            {
                result << " + ";
            }
            result << uint_to_dec_string(value, coeff_uint64_count);
            if (coeff_count)
            {
                result << "x^" << coeff_count;
            }
            empty = false;
            value -= coeff_uint64_count;
        }
        if (empty)
        {
            result << "0";
        }
        return result.str();
    }
}}