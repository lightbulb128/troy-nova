#pragma once
#include "modulus.h"
#include "encryption_parameters.h"
#include "utils/he_standard_params.h"
#include "utils/number_theory.h"

namespace troy {

    class CoeffModulus {

        static std::vector<uint64_t> bfv_default_vector(size_t poly_modulus_degree, SecurityLevel sec_level);

    public:

        inline static size_t max_bit_count(size_t poly_modulus_degree, SecurityLevel sec_level) {
            switch (sec_level) {
                case SecurityLevel::Nil: return 2147483648;
                case SecurityLevel::Classical128: return he_standard_params::params_classical_128(poly_modulus_degree);
                case SecurityLevel::Classical192: return he_standard_params::params_classical_192(poly_modulus_degree);
                case SecurityLevel::Classical256: return he_standard_params::params_classical_256(poly_modulus_degree);
            }
            throw std::invalid_argument("[CoeffModulus::max_bit_count] Unreachable.");
        }

        inline static utils::Array<Modulus> bfv_default(size_t poly_modulus_degree, SecurityLevel sec_level) {
            std::vector<uint64_t> vec = bfv_default_vector(poly_modulus_degree, sec_level);
            utils::Array<Modulus> result(vec.size(), false, nullptr);
            for (size_t i = 0; i < vec.size(); i++) {
                result[i] = Modulus(vec[i]);
            }
            return result;
        }

        static utils::Array<Modulus> create(size_t poly_modulus_degree, std::vector<size_t> bit_sizes);

    };

    class PlainModulus {

    public:
    
        inline static Modulus batching(size_t poly_modulus_degree, size_t bit_size) {
            Modulus ret = CoeffModulus::create(poly_modulus_degree, {bit_size})[0];
            return ret;
        }

        inline static utils::Array<Modulus> batching_multiple(size_t poly_modulus_degree, std::vector<size_t> bit_sizes) {
            return CoeffModulus::create(poly_modulus_degree, bit_sizes);
        }

    };

}