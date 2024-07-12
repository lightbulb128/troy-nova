#include "poly_to_string.h"
#include <algorithm>

namespace troy::utils {

    using namespace std;

    const size_t nibbles_per_uint64 = 16;

    inline char nibble_to_upper_hex(int nibble)
    {
        if (nibble < 10)
        {
            return static_cast<char>(nibble + static_cast<int>('0'));
        }
        return static_cast<char>(nibble + static_cast<int>('A') - 10);
    }

    string uint_to_hex_string(const uint64_t *value, size_t uint64_count)
    {
        // Start with a string with a zero for each nibble in the array.
        size_t num_nibbles = uint64_count * static_cast<size_t>(nibbles_per_uint64);
        string output(num_nibbles, '0');

        // Iterate through each uint64 in array and set string with correct nibbles in hex.
        size_t nibble_index = num_nibbles;
        size_t leftmost_non_zero_pos = num_nibbles;
        for (size_t i = 0; i < uint64_count; i++)
        {
            uint64_t part = *value++;

            // Iterate through each nibble in the current uint64.
            for (size_t j = 0; j < nibbles_per_uint64; j++)
            {
                size_t nibble = static_cast<size_t>(part & uint64_t(0x0F));
                size_t pos = --nibble_index;
                if (nibble != 0)
                {
                    // If nibble is not zero, then update string and save this pos to determine
                    // number of leading zeros.
                    output[pos] = nibble_to_upper_hex(static_cast<int>(nibble));
                    leftmost_non_zero_pos = pos;
                }
                part >>= 4;
            }
        }

        // Trim string to remove leading zeros.
        output.erase(0, leftmost_non_zero_pos);

        // Return 0 if nothing remains.
        if (output.empty())
        {
            return string("0");
        }

        return output;
    }

    string uint_to_dec_string(const uint64_t *value, size_t uint64_count)
    {
        if (!uint64_count)
        {
            return string("0");
        }
        Array<uint64_t> remainder(uint64_count, false, nullptr);
        Array<uint64_t> quotient(uint64_count, false, nullptr);
        Array<uint64_t> base(uint64_count, false, nullptr);
        base[0] = 10;
        remainder.copy_from_slice(ConstSlice<uint64_t>(value, uint64_count, false, nullptr));
        string output;
        while (!is_zero_uint(remainder.const_reference()))
        {
            divide_uint_inplace(remainder.reference(), base.const_reference(), quotient.reference(), nullptr);
            char digit = static_cast<char>(remainder[0] + static_cast<uint64_t>('0'));
            output += digit;
            Array<uint64_t> t = std::move(remainder);
            remainder = std::move(quotient);
            quotient = std::move(t);
        }
        std::reverse(output.begin(), output.end());

        // Return 0 if nothing remains.
        if (output.empty())
        {
            return string("0");
        }

        return output;
    }

}