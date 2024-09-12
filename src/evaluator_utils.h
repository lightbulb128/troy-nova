#pragma once
#include "ciphertext.h"

namespace troy {

    template <typename C>
    inline void check_no_seed(const char* prompt, const C& c) {
        if (c.contains_seed()) {
            throw std::invalid_argument(std::string(prompt) + " Argument contains seed.");
        }
    }

    template <typename C>
    inline void check_no_seed_vec(const char* prompt, const std::vector<C*>& c) {
        for (const C* item : c) {
            check_no_seed(prompt, *item);
        }
    }

    inline void check_ciphertext(const char* prompt, const Ciphertext& ciphertext) {
        check_no_seed(prompt, ciphertext);
    }

    inline void check_ciphertext_vec(const char* prompt, const std::vector<Ciphertext*>& ciphertexts) {
        for (const Ciphertext* c : ciphertexts) {
            check_ciphertext(prompt, *c);
        }
    }

    inline void check_ciphertext_vec(const char* prompt, const std::vector<const Ciphertext*>& ciphertexts) {
        for (const Ciphertext* c : ciphertexts) {
            check_ciphertext(prompt, *c);
        }
    }
    

    template <typename C1, typename C2>
    inline void check_same_parms_id(const char* prompt, const C1& a, const C2& b) {
        if (a.parms_id() != b.parms_id()) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different parms ID.");
        }
    }

    template <typename C1, typename C2>
    inline void check_same_parms_id_vec(const char* prompt, const std::vector<C1*>& a, const C2& b) {
        for (const C1* c : a) {
            check_same_parms_id(prompt, *c, b);
        }
    }

    template <typename C>
    inline void check_same_parms_id_vec(const char* prompt, const std::vector<C*>& a) {
        if (a.size() == 0) return;
        const ParmsID& parms_id = a[0]->parms_id();
        for (size_t i = 1; i < a.size(); i++) {
            if (a[i]->parms_id() != parms_id) {
                throw std::invalid_argument(std::string(prompt) + " Arguments have different parms ID.");
            }
        }
    }

    template <typename C1, typename C2>
    inline void check_same_parms_id_vec(const char* prompt, const std::vector<C1*>& a, const std::vector<C2*>& b) {
        check_same_parms_id_vec(prompt, a);
        check_same_parms_id_vec(prompt, b);
        if (a.size() == 0 || b.size() == 0) return;
        if (a[0]->parms_id() != b[0]->parms_id()) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different parms ID.");
        }
    }

    template<typename C1, typename C2>
    inline void check_same_scale(const char* prompt, const C1& a, const C2& b) {
        if (!utils::are_close_double(a.scale(), b.scale())) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different scales.");
        }
    }

    template<typename C1>
    inline void check_same_scale_vec(const char* prompt, const std::vector<C1*>& a) {
        if (a.size() == 0) return;
        double scale = a[0]->scale();
        for (size_t i = 1; i < a.size(); i++) {
            if (!utils::are_close_double(a[i]->scale(), scale)) {
                throw std::invalid_argument(std::string(prompt) + " Arguments have different scales.");
            }
        }
    }

    template<typename C1, typename C2>
    inline void check_same_scale_vec(const char* prompt, const std::vector<C1*>& a, const C2& b) {
        for (const Ciphertext* c : a) {
            if (!utils::are_close_double(c->scale(), b.scale())) {
                throw std::invalid_argument(std::string(prompt) + " Arguments have different scales.");
            }
        }
    }

    template<typename C1, typename C2>
    inline void check_same_scale_vec(const char* prompt, const std::vector<C1*>& a, const std::vector<C2*>& b) {
        check_same_scale_vec(prompt, a);
        check_same_scale_vec(prompt, b);
        if (a.size() == 0 || b.size() == 0) return;
        if (!utils::are_close_double(a[0]->scale(), b[0]->scale())) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different scales.");
        }
    }

    template<typename C1, typename C2>
    inline void check_same_ntt_form(const char* prompt, const C1& a, const C2& b) {
        if (a.is_ntt_form() != b.is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different NTT form.");
        }
    }

    template<typename C1>
    inline void check_same_ntt_form_vec(const char* prompt, const std::vector<C1*>& a) {
        if (a.size() == 0) return;
        bool ntt_form = a[0]->is_ntt_form();
        for (size_t i = 1; i < a.size(); i++) {
            if (a[i]->is_ntt_form() != ntt_form) {
                throw std::invalid_argument(std::string(prompt) + " Arguments have different NTT form.");
            }
        }
    }

    template<typename C1, typename C2>
    inline void check_same_ntt_form_vec(const char* prompt, const std::vector<C1*>& a, const C2& b) {
        check_same_ntt_form_vec(prompt, a);
        if (a.size() == 0) return;
        if (a[0]->is_ntt_form() != b.is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different NTT form.");
        }
    }

    template<typename C1, typename C2>
    inline void check_same_ntt_form_vec(const char* prompt, const std::vector<C1*>& a, const std::vector<C2*>& b) {
        check_same_ntt_form_vec(prompt, a);
        check_same_ntt_form_vec(prompt, b);
        if (a.size() == 0 || b.size() == 0) return;
        if (a[0]->is_ntt_form() != b[0]->is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different NTT form.");
        }
    }

    template<typename C1>
    inline void check_is_ntt_form(const char* prompt, const C1& a) {
        if (!a.is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Argument is not in NTT form.");
        }
    }

    template<typename C1>
    inline void check_is_ntt_form_vec(const char* prompt, const std::vector<C1*>& a) {
        for (const C1* c : a) {
            check_is_ntt_form(prompt, *c);
        }
    }

    template<typename C1>
    inline void check_is_not_ntt_form(const char* prompt, const C1& a) {
        if (a.is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Argument is in NTT form.");
        }
    }

    template<typename C1>
    inline void check_is_not_ntt_form_vec(const char* prompt, const std::vector<C1*>& a) {
        for (C1* c : a) {
            check_is_not_ntt_form(prompt, *c);
        }
    }

    template <typename T>
    inline ParmsID get_vec_parms_id(const std::vector<T*>& vec) {
        check_same_parms_id_vec("[get_vec_parms_id]", vec);
        if (vec.size() == 0) {
            throw std::invalid_argument("[get_vec_parms_id] Empty vector.");
        }
        return vec[0]->parms_id();
    }

    template <typename T>
    inline bool get_is_ntt_form_vec(const std::vector<T*>& vec) {
        if (vec.size() == 0) {
            throw std::invalid_argument("[get_is_ntt_form] Empty vector.");
        }
        bool r = vec[0]->is_ntt_form();
        for (size_t i = 1; i < vec.size(); i++) {
            if (vec[i]->is_ntt_form() != r) {
                throw std::invalid_argument("[get_is_ntt_form] Arguments have different NTT forms.");
            }
        }
        return r;
    }

    template <typename T>
    inline size_t get_vec_coeff_count(const std::vector<T*>& vec) {
        if (vec.size() == 0) {
            throw std::invalid_argument("[get_vec_coeff_count] Empty vector.");
        }
        size_t r = vec[0]->coeff_count();
        for (size_t i = 1; i < vec.size(); i++) {
            if (vec[i]->coeff_count() != r) {
                throw std::invalid_argument("[get_vec_coeff_count] Arguments have different coefficient counts.");
            }
        }
        return r;
    }

    template <typename T>
    inline size_t get_vec_polynomial_count(const std::vector<T*>& vec) {
        if (vec.size() == 0) {
            throw std::invalid_argument("[get_vec_polynomial_count] Empty vector.");
        }
        size_t r = vec[0]->polynomial_count();
        for (size_t i = 1; i < vec.size(); i++) {
            if (vec[i]->polynomial_count() != r) {
                throw std::invalid_argument("[get_vec_polynomial_count] Arguments have different polynomial counts.");
            }
        }
        return r;
    }

    template <typename T>
    inline double get_vec_scale(const std::vector<T*>& vec) {
        if (vec.size() == 0) {
            throw std::invalid_argument("[get_vec_scale] Empty vector.");
        }
        double s = vec[0]->scale();
        for (size_t i = 1; i < vec.size(); i++) {
            if (!utils::are_close_double(vec[i]->scale(), s)) {
                throw std::invalid_argument("[get_vec_scale] Arguments have different scales.");
            }
        }
        return s;
    }

    template <typename T>
    inline uint64_t get_vec_correction_factor(const std::vector<T*>& vec) {
        if (vec.size() == 0) {
            throw std::invalid_argument("[get_vec_correction_factor] Empty vector.");
        }
        uint64_t r = vec[0]->correction_factor();
        for (size_t i = 1; i < vec.size(); i++) {
            if (vec[i]->correction_factor() != r) {
                throw std::invalid_argument("[get_vec_correction_factor] Arguments have different correction factors.");
            }
        }        
        return r;
    }


    inline void balance_correction_factors(
        uint64_t factor1, uint64_t factor2, const Modulus& plain_modulus,
        uint64_t& prod, uint64_t& e1, uint64_t& e2
    ) {
        uint64_t t = plain_modulus.value();
        uint64_t half_t = t >> 1;
        // dunno why GCC complains about an unused typedef here
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wunused-local-typedefs"
        auto sum_abs = [half_t, t](uint64_t x, uint64_t y) -> uint64_t {
            int64_t x_bal = x > half_t ? static_cast<int64_t>(x - t) : static_cast<int64_t>(x);
            int64_t y_bal = y > half_t ? static_cast<int64_t>(y - t) : static_cast<int64_t>(y);
            return static_cast<uint64_t>(std::abs(x_bal) + std::abs(y_bal));
        };
        #pragma GCC diagnostic pop
        uint64_t ratio = 1;
        if (!utils::try_invert_uint64_mod(factor1, plain_modulus, ratio)) {
            throw std::logic_error("[balance_correction_factors] Failed to invert factor1.");
        }
        ratio = utils::multiply_uint64_mod(ratio, factor2, plain_modulus);
        e1 = ratio;
        e2 = 1;
        uint64_t sum = sum_abs(factor1, factor2);
        
        // Extended Euclidean
        int64_t prev_a = static_cast<int64_t>(plain_modulus.value());
        int64_t prev_b = static_cast<int64_t>(0);
        int64_t a = static_cast<int64_t>(ratio);
        int64_t b = static_cast<int64_t>(1);
        while (a != 0) {
            int64_t q = prev_a / a;
            int64_t temp = prev_a % a;
            prev_a = a;
            a = temp;
            temp = prev_b - q * b;
            prev_b = b;
            b = temp;
            uint64_t a_mod = plain_modulus.reduce(static_cast<uint64_t>(std::abs(a)));
            if (a < 0) {a_mod = utils::negate_uint64_mod(a_mod, plain_modulus);}
            uint64_t b_mod = plain_modulus.reduce(static_cast<uint64_t>(std::abs(b)));
            if (b < 0) {b_mod = utils::negate_uint64_mod(b_mod, plain_modulus);}
            if ((a_mod != 0) && (utils::gcd(a_mod, t) == 1)) {
                uint64_t new_sum = sum_abs(a_mod, b_mod);
                if (new_sum < sum) {
                    e1 = a_mod;
                    e2 = b_mod;
                    sum = new_sum;
                }
            }
        }
        prod = utils::multiply_uint64_mod(e1, factor1, plain_modulus);
    }

    inline bool is_scale_within_bounds(double scale, ContextDataPointer context_data) {
        SchemeType scheme = context_data->parms().scheme();
        int scale_bit_count_bound = -1;
        switch (scheme) {
            case SchemeType::BFV: case SchemeType::BGV: {
                scale_bit_count_bound = static_cast<int>(context_data->parms().plain_modulus_host().bit_count());
                break;
            }
            case SchemeType::CKKS: {
                scale_bit_count_bound = static_cast<int>(context_data->total_coeff_modulus_bit_count());
                break;
            }
            default: break;
        }
        // std::cerr << static_cast<int>(std::log2(scale)) << " " << scale_bit_count_bound << std::endl;
        return !(scale <= 0.0 || static_cast<int>(std::log2(scale)) >= scale_bit_count_bound);
    }


}