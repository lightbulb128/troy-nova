#pragma once
#include <memory>
#include "basics.h"
#include "cuda_runtime.h"
#include "troy/troy.cuh"
#include "utils.h"
#include "rust/cxx.h"

namespace troy_rust {

    class Modulus;
    typedef std::unique_ptr<Modulus> UpModulus;

    class Modulus {
    private:
        troy::Modulus p;
    public:
        explicit inline Modulus(troy::Modulus&& p) : p(std::move(p)) {}
        explicit inline Modulus(uint64_t value): p(value) {}
        Modulus(const Modulus& p) = default;
        Modulus(Modulus&& p) = default;
        Modulus& operator=(const Modulus& p) = default;
        Modulus& operator=(Modulus&& p) = default;

        inline bool is_prime() const {
            return p.is_prime();
        }
        inline bool is_zero() const {
            return p.is_zero();
        }
        inline uint64_t value() const {
            return p.value();
        }
        inline size_t bit_count() const {
            return p.bit_count();
        }
        inline rust::String to_string() const {
            return rust::String(utils::to_string(p));
        }
        inline uint64_t reduce(uint64_t value) const {
            return p.reduce(value);
        }
        inline uint64_t reduce_u128(uint64_t high, uint64_t low) const {
            uint128_t value = (static_cast<uint128_t>(high) << 64) | (static_cast<uint128_t>(low));
            return p.reduce(value);
        }
        inline uint64_t reduce_mul_u64(uint64_t operand1, uint64_t operand2) const {
            return p.reduce_mul_uint64(operand1, operand2);
        }
        
    };

    inline UpModulus modulus_constructor(uint64_t value) {
        return std::make_unique<Modulus>(value);
    }
    inline UpModulus modulus_constructor_copy(const Modulus& p) {
        return std::make_unique<Modulus>(p);
    }

    inline size_t coeff_modulus_max_bit_count(size_t poly_modulus_degree, SecurityLevel security_level) {
        return troy::CoeffModulus::max_bit_count(poly_modulus_degree, security_level);
    }
    inline void coeff_modulus_bfv_default(size_t poly_modulus_degree, SecurityLevel sec, std::unique_ptr<std::vector<Modulus>>& output) {
        troy::utils::Array<troy::Modulus> moduli = troy::CoeffModulus::bfv_default(poly_modulus_degree, sec);
        output->clear();
        for (size_t i = 0; i < moduli.size(); i++) {
            output->push_back(Modulus(std::move(moduli[i])));
        }
    }
    // inline std::vector<UpModulus> coeff_modulus_create(size_t poly_modulus_degree, rust::Slice<const size_t> bit_sizes) {
    //     std::vector<size_t> bit_sizes_cpp(bit_sizes.begin(), bit_sizes.end());
    //     troy::utils::Array<troy::Modulus> moduli = troy::CoeffModulus::create(poly_modulus_degree, bit_sizes_cpp);
    //     std::vector<UpModulus> result;
    //     for (size_t i = 0; i < moduli.size(); i++) {
    //         result.push_back(std::make_unique<Modulus>(std::move(moduli[i])));
    //     }
    //     return result;
    // }

    // inline UpModulus plain_modulus_batching(size_t poly_modulus_degree, size_t bit_size) {
    //     troy::Modulus modulus = troy::PlainModulus::batching(poly_modulus_degree, bit_size);
    //     return std::make_unique<Modulus>(std::move(modulus));
    // }
    // inline std::vector<UpModulus> plain_modulus_batching_multiple(size_t poly_modulus_degree, rust::Slice<const size_t> bit_sizes) {
    //     std::vector<size_t> bit_sizes_cpp(bit_sizes.begin(), bit_sizes.end());
    //     troy::utils::Array<troy::Modulus> moduli = troy::PlainModulus::batching_multiple(poly_modulus_degree, bit_sizes_cpp);
    //     std::vector<UpModulus> result;
    //     for (size_t i = 0; i < moduli.size(); i++) {
    //         result.push_back(std::make_unique<Modulus>(std::move(moduli[i])));
    //     }
    //     return result;
    // }
    

}