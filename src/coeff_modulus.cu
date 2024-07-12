#include "coeff_modulus.h"
#include <map>

namespace troy {

    std::vector<uint64_t> CoeffModulus::bfv_default_vector(size_t poly_modulus_degree, SecurityLevel sec_level) {
        if (CoeffModulus::max_bit_count(poly_modulus_degree, sec_level) == 0) {
            throw std::invalid_argument("[CoeffModulus::bfv_default_vector] Invalid poly_modulus_degree or sec_level.");
        }
        switch (sec_level) {
            case SecurityLevel::Nil: 
                throw std::invalid_argument("[CoeffModulus::bfv_default_vector] No default for Nil security.");
            case SecurityLevel::Classical128: {
                switch (poly_modulus_degree) {
                    case 1024: return { 0x7e00001 };
                    case 2048: return { 0x3fffffff000001 };
                    case 4096: return { 0xffffee001, 0xffffc4001, 0x1ffffe0001 };
                    case 8192: return { 0x7fffffd8001, 0x7fffffc8001, 0xfffffffc001, 0xffffff6c001, 0xfffffebc001 };
                    case 16384: return { 
                        0xfffffffd8001, 0xfffffffa0001, 0xfffffff00001, 0x1fffffff68001, 0x1fffffff50001,
                        0x1ffffffee8001, 0x1ffffffea0001, 0x1ffffffe88001, 0x1ffffffe48001
                        };
                    case 32768: return { 
                        0x7fffffffe90001, 0x7fffffffbf0001, 0x7fffffffbd0001, 0x7fffffffba0001, 0x7fffffffaa0001,
                        0x7fffffffa50001, 0x7fffffff9f0001, 0x7fffffff7e0001, 0x7fffffff770001, 0x7fffffff380001,
                        0x7fffffff330001, 0x7fffffff2d0001, 0x7fffffff170001, 0x7fffffff150001, 0x7ffffffef00001,
                        0xfffffffff70001
                    };
                    default: throw std::invalid_argument("[CoeffModulus::bfv_default_vector] Invalid poly_modulus_degree.");
                }
            }
            case SecurityLevel::Classical192: {
                switch (poly_modulus_degree) {
                    case 1024: return { 0x7f001 };
                    case 2048: return { 0x1ffffc0001 };
                    case 4096: return { 0x1ffc001, 0x1fce001, 0x1fc0001 };
                    case 8192: return { 0x3ffffac001, 0x3ffff54001, 0x3ffff48001, 0x3ffff28001 };
                    case 16384: return { 
                        0x3ffffffdf0001, 0x3ffffffd48001, 0x3ffffffd20001, 0x3ffffffd18001, 0x3ffffffcd0001,
                        0x3ffffffc70001 };
                    case 32768: return { 
                        0x3fffffffd60001, 0x3fffffffca0001, 0x3fffffff6d0001, 0x3fffffff5d0001, 0x3fffffff550001,
                        0x7fffffffe90001, 0x7fffffffbf0001, 0x7fffffffbd0001, 0x7fffffffba0001, 0x7fffffffaa0001,
                        0x7fffffffa50001 };
                    default: throw std::invalid_argument("[CoeffModulus::bfv_default_vector] Invalid poly_modulus_degree.");
                }
            }
            case SecurityLevel::Classical256: {
                switch (poly_modulus_degree) {
                    case 1024: return { 0x3001 };
                    case 2048: return { 0x1ffc0001 };
                    case 4096: return { 0x3ffffffff040001 };
                    case 8192: return { 0x7ffffec001, 0x7ffffb0001, 0xfffffdc001 };
                    case 16384: return { 0x7ffffffc8001, 0x7ffffff00001, 0x7fffffe70001, 0xfffffffd8001, 0xfffffffa0001 };
                    case 32768: return { 
                        0xffffffff00001, 0x1fffffffe30001, 0x1fffffffd80001, 0x1fffffffd10001, 0x1fffffffc50001,
                        0x1fffffffbf0001, 0x1fffffffb90001, 0x1fffffffb60001, 0x1fffffffa50001 };
                    default: throw std::invalid_argument("[CoeffModulus::bfv_default_vector] Invalid poly_modulus_degree.");
                }
            }
        }
        throw std::invalid_argument("[CoeffModulus::bfv_default_vector] Unreachable.");
    }

    utils::Array<Modulus> CoeffModulus::create(size_t poly_modulus_degree, std::vector<size_t> bit_sizes) {
        if (poly_modulus_degree > utils::HE_POLY_MOD_DEGREE_MAX || poly_modulus_degree < utils::HE_POLY_MOD_DEGREE_MIN) {
            throw std::invalid_argument("[CoeffModulus::create] Invalid poly_modulus_degree.");
        }
        if (bit_sizes.size() > utils::HE_COEFF_MOD_COUNT_MAX || bit_sizes.size() < utils::HE_COEFF_MOD_COUNT_MIN) {
            throw std::invalid_argument("[CoeffModulus::create] Invalid bit_sizes length.");
        }
        size_t max_bit_size = bit_sizes[0];
        size_t min_bit_size = bit_sizes[0];
        for (size_t i = 1; i < bit_sizes.size(); i++) {
            if (bit_sizes[i] > max_bit_size) {
                max_bit_size = bit_sizes[i];
            }
            if (bit_sizes[i] < min_bit_size) {
                min_bit_size = bit_sizes[i];
            }
        }
        if (max_bit_size > utils::HE_USER_MOD_BIT_COUNT_MAX || max_bit_size < utils::HE_USER_MOD_BIT_COUNT_MIN) {
            throw std::invalid_argument("[CoeffModulus::create] Invalid max_bit_size.");
        }
        std::map<size_t, size_t> count_table;
        std::map<size_t, std::vector<Modulus>> prime_table;
        for (size_t i = 0; i < bit_sizes.size(); i++) {
            size_t size = bit_sizes[i];
            if (count_table.find(size) == count_table.end()) {
                count_table[size] = 1;
            } else {
                count_table[size] += 1;
            }
        }
        size_t factor = 2 * poly_modulus_degree;
        for (auto const& [key, value] : count_table) {
            prime_table.emplace(key, utils::get_primes(factor, key, value));
        }
        utils::Array<Modulus> result(bit_sizes.size(), false, nullptr);
        size_t i = 0;
        for (size_t size : bit_sizes) {
            // get the last one of the size and remove it from prime table
            result[i] = prime_table[size].back();
            prime_table[size].pop_back();
            i++;
        }
        return result;
    }

}