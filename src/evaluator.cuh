#pragma once
#include "he_context.cuh"
#include "plaintext.cuh"
#include "ciphertext.cuh"
#include "kswitch_keys.cuh"
#include "utils/scaling_variant.cuh"
#include "lwe_ciphertext.cuh"
#include <string>

namespace troy {

    class Evaluator {
        HeContextPointer context_;

        ContextDataPointer get_context_data(const char* prompt, const ParmsID& encrypted) const;

        void translate_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, bool subtract) const;
        
        void bfv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const;
        void ckks_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const;
        void bgv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const;

        void bfv_square_inplace(Ciphertext& encrypted) const;
        void ckks_square_inplace(Ciphertext& encrypted) const;
        void bgv_square_inplace(Ciphertext& encrypted) const;

        /// Suppose kswitch_keys[kswitch_kes_index] is generated with s' on a KeyGenerator of secret key s.
        /// Then the semantic of this function is as follows: `target` is supposed to multiply with s' to contribute to the
        /// decrypted result, now we apply this function, to decompose (target * s') into (c0, c1) such that c0 + c1 * s = target * s.
        /// And then we add c0, c1 to the original c0, c1 in the `encrypted`.
        void switch_key_inplace_internal(Ciphertext& encrypted, utils::ConstSlice<uint64_t> target, const KSwitchKeys& kswitch_keys, size_t kswitch_keys_index) const;

        void relinearize_inplace_internal(Ciphertext& encrypted, const RelinKeys& relin_keys, size_t destination_size) const;

        void mod_switch_scale_to_next_internal(const Ciphertext& encrypted, Ciphertext& destination) const;
        void mod_switch_drop_to_next_internal(const Ciphertext& encrypted, Ciphertext& destination) const;
        void mod_switch_drop_to_next_plain_inplace_internal(Plaintext& plain) const;

        void translate_plain_inplace(Ciphertext& encrypted, const Plaintext& plain, bool subtract) const;

        void multiply_plain_normal_inplace(Ciphertext& encrypted, const Plaintext& plain) const;
        void multiply_plain_ntt_inplace(Ciphertext& encrypted, const Plaintext& plain) const;

        void rotate_inplace_internal(Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys) const;
        void conjugate_inplace_internal(Ciphertext& encrypted, const GaloisKeys& galois_keys) const;

    public:
        inline Evaluator(HeContextPointer context): context_(context) {}
        inline HeContextPointer context() const { return context_; }
        inline bool on_device() const {return this->context()->on_device();}

        void negate_inplace(Ciphertext& encrypted) const;
        inline void negate(const Ciphertext& encrypted, Ciphertext& destination) const {
            destination = encrypted;
            negate_inplace(destination);
        }
        inline Ciphertext negate_new(const Ciphertext& encrypted) const {
            Ciphertext destination = encrypted;
            negate_inplace(destination);
            return destination;
        }

        inline void add_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
            translate_inplace(encrypted1, encrypted2, false);
        }
        inline void add(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination) const {
            destination = encrypted1;
            add_inplace(destination, encrypted2);
        }
        inline Ciphertext add_new(const Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
            Ciphertext destination;
            add(encrypted1, encrypted2, destination);
            return destination;
        }

        inline void sub_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
            translate_inplace(encrypted1, encrypted2, true);
        }
        inline void sub(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination) const {
            destination = encrypted1;
            sub_inplace(destination, encrypted2);
        }
        inline Ciphertext sub_new(const Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
            Ciphertext destination;
            sub(encrypted1, encrypted2, destination);
            return destination;
        }

        void multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const;
        inline void multiply(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination) const {
            destination = encrypted1;
            multiply_inplace(destination, encrypted2);
        }
        inline Ciphertext multiply_new(const Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
            Ciphertext destination;
            multiply(encrypted1, encrypted2, destination);
            return destination;
        }

        void square_inplace(Ciphertext& encrypted) const;
        inline void square(const Ciphertext& encrypted, Ciphertext& destination) const {
            destination = encrypted;
            square_inplace(destination);
        }
        inline Ciphertext square_new(const Ciphertext& encrypted) const {
            Ciphertext destination;
            square(encrypted, destination);
            return destination;
        }

        void apply_keyswitching_inplace(Ciphertext& encrypted, const KSwitchKeys& kswitch_keys) const;
        inline void apply_keyswitching(const Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, Ciphertext& destination) const {
            destination = encrypted;
            apply_keyswitching_inplace(destination, kswitch_keys);
        }
        inline Ciphertext apply_keyswitching_new(const Ciphertext& encrypted, const KSwitchKeys& kswitch_keys) const {
            Ciphertext destination;
            apply_keyswitching(encrypted, kswitch_keys, destination);
            return destination;
        }

        inline void relinearize_inplace(Ciphertext& encrypted, const RelinKeys& relin_keys) const {
            relinearize_inplace_internal(encrypted, relin_keys, 2);
        }
        inline void relinearize(const Ciphertext& encrypted, const RelinKeys& relin_keys, Ciphertext& destination) const {
            destination = encrypted;
            relinearize_inplace(destination, relin_keys);
        }
        inline Ciphertext relinearize_new(const Ciphertext& encrypted, const RelinKeys& relin_keys) const {
            Ciphertext destination;
            relinearize(encrypted, relin_keys, destination);
            return destination;
        }

        void mod_switch_to_next(const Ciphertext& encrypted, Ciphertext& destination) const;
        inline void mod_switch_to_next_inplace(Ciphertext& encrypted) const {
            Ciphertext cloned = encrypted;
            mod_switch_to_next(cloned, encrypted);
        }
        inline Ciphertext mod_switch_to_next_new(const Ciphertext& encrypted) const {
            Ciphertext destination;
            mod_switch_to_next(encrypted, destination);
            return destination;
        }

        inline void mod_switch_plain_to_next_inplace(Plaintext& plain) const {
            this->mod_switch_drop_to_next_plain_inplace_internal(plain);
        }
        inline void mod_switch_plain_to_next(const Plaintext& plain, Plaintext& destination) const {
            destination = plain.clone();
            this->mod_switch_drop_to_next_plain_inplace_internal(destination);
        }
        inline Plaintext mod_switch_plain_to_next_new(const Plaintext& plain) const {
            Plaintext destination = plain.clone();
            this->mod_switch_drop_to_next_plain_inplace_internal(destination);
            return destination;
        }

        void mod_switch_to_inplace(Ciphertext& encrypted, const ParmsID& parms_id) const;
        inline void mod_switch_to(const Ciphertext& encrypted, const ParmsID& parms_id, Ciphertext& destination) const {
            destination = encrypted;
            mod_switch_to_inplace(destination, parms_id);
        }
        inline Ciphertext mod_switch_to_new(const Ciphertext& encrypted, const ParmsID& parms_id) const {
            Ciphertext destination;
            mod_switch_to(encrypted, parms_id, destination);
            return destination;
        }

        void mod_switch_plain_to_inplace(Plaintext& plain, const ParmsID& parms_id) const;
        inline void mod_switch_plain_to(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination) const {
            destination = plain.clone();
            mod_switch_plain_to_inplace(destination, parms_id);
        }
        inline Plaintext mod_switch_plain_to_new(const Plaintext& plain, const ParmsID& parms_id) const {
            Plaintext destination = plain.clone();
            mod_switch_plain_to_inplace(destination, parms_id);
            return destination;
        }

        void rescale_to_next(const Ciphertext& encrypted, Ciphertext& destination) const;
        inline void rescale_to_next_inplace(Ciphertext& encrypted) const {
            Ciphertext cloned = encrypted;
            rescale_to_next(cloned, encrypted);
        }
        inline Ciphertext rescale_to_next_new(const Ciphertext& encrypted) const {
            Ciphertext destination;
            rescale_to_next(encrypted, destination);
            return destination;
        }

        void rescale_to(const Ciphertext& encrypted, const ParmsID& parms_id, Ciphertext& destination) const;
        inline void rescale_to_inplace(Ciphertext& encrypted, const ParmsID& parms_id) const {
            Ciphertext cloned = encrypted;
            rescale_to(cloned, parms_id, encrypted);
        }
        inline Ciphertext rescale_to_new(const Ciphertext& encrypted, const ParmsID& parms_id) const {
            Ciphertext destination;
            rescale_to(encrypted, parms_id, destination);
            return destination;
        }

        inline void add_plain_inplace(Ciphertext& encrypted, const Plaintext& plain) const {
            translate_plain_inplace(encrypted, plain, false);
        }
        inline void add_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination) const {
            destination = encrypted;
            add_plain_inplace(destination, plain);
        }
        inline Ciphertext add_plain_new(const Ciphertext& encrypted, const Plaintext& plain) const {
            Ciphertext destination;
            add_plain(encrypted, plain, destination);
            return destination;
        }

        inline void sub_plain_inplace(Ciphertext& encrypted, const Plaintext& plain) const {
            translate_plain_inplace(encrypted, plain, true);
        }
        inline void sub_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination) const {
            destination = encrypted;
            sub_plain_inplace(destination, plain);
        }
        inline Ciphertext sub_plain_new(const Ciphertext& encrypted, const Plaintext& plain) const {
            Ciphertext destination;
            sub_plain(encrypted, plain, destination);
            return destination;
        }

        void multiply_plain_inplace(Ciphertext& encrypted, const Plaintext& plain) const;
        inline void multiply_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination) const {
            destination = encrypted;
            multiply_plain_inplace(destination, plain);
        }
        inline Ciphertext multiply_plain_new(const Ciphertext& encrypted, const Plaintext& plain) const {
            Ciphertext destination;
            multiply_plain(encrypted, plain, destination);
            return destination;
        }

        void transform_plain_to_ntt_inplace(Plaintext& plain, const ParmsID& parms_id) const;
        inline void transform_plain_to_ntt(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination) const {
            destination = plain.clone();
            transform_plain_to_ntt_inplace(destination, parms_id);
        }
        inline Plaintext transform_plain_to_ntt_new(const Plaintext& plain, const ParmsID& parms_id) const {
            Plaintext destination = plain.clone();
            transform_plain_to_ntt_inplace(destination, parms_id);
            return destination;
        }

        void transform_to_ntt_inplace(Ciphertext& encrypted) const;
        inline void transform_to_ntt(const Ciphertext& encrypted, Ciphertext& destination) const {
            destination = encrypted;
            transform_to_ntt_inplace(destination);
        }
        inline Ciphertext transform_to_ntt_new(const Ciphertext& encrypted) const {
            Ciphertext destination;
            transform_to_ntt(encrypted, destination);
            return destination;
        }

        void transform_from_ntt_inplace(Ciphertext& encrypted) const;
        inline void transform_from_ntt(const Ciphertext& encrypted, Ciphertext& destination) const {
            destination = encrypted;
            transform_from_ntt_inplace(destination);
        }
        inline Ciphertext transform_from_ntt_new(const Ciphertext& encrypted) const {
            Ciphertext destination;
            transform_from_ntt(encrypted, destination);
            return destination;
        }

        void apply_galois_inplace(Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys) const;
        inline void apply_galois(const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, Ciphertext& destination) const {
            destination = encrypted;
            apply_galois_inplace(destination, galois_element, galois_keys);
        }
        inline Ciphertext apply_galois_new(const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys) const {
            Ciphertext destination;
            apply_galois(encrypted, galois_element, galois_keys, destination);
            return destination;
        }

        void apply_galois_plain_inplace(Plaintext& plain, size_t galois_element) const;
        inline void apply_galois_plain(const Plaintext& plain, size_t galois_element, Plaintext& destination) const {
            destination = plain.clone();
            apply_galois_plain_inplace(destination, galois_element);
        }
        inline Plaintext apply_galois_plain_new(const Plaintext& plain, size_t galois_element) const {
            Plaintext destination = plain.clone();
            apply_galois_plain_inplace(destination, galois_element);
            return destination;
        }

        inline void rotate_rows_inplace(Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) {
                rotate_inplace_internal(encrypted, steps, galois_keys);
            } else {
                throw std::invalid_argument("[Evaluator::rotate_rows_inplace] Rotate rows only applies for BFV or BGV");
            }
        }
        inline void rotate_rows(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination) const {
            destination = encrypted;
            rotate_rows_inplace(destination, steps, galois_keys);
        }
        inline Ciphertext rotate_rows_new(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys) const {
            Ciphertext destination;
            rotate_rows(encrypted, steps, galois_keys, destination);
            return destination;
        }

        inline void rotate_columns_inplace(Ciphertext& encrypted, const GaloisKeys& galois_keys) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) {
                conjugate_inplace_internal(encrypted, galois_keys);
            } else {
                throw std::invalid_argument("[Evaluator::rotate_columns_inplace] Rotate columns only applies for BFV or BGV");
            }
        }
        inline void rotate_columns(const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination) const {
            destination = encrypted;
            rotate_columns_inplace(destination, galois_keys);
        }
        inline Ciphertext rotate_columns_new(const Ciphertext& encrypted, const GaloisKeys& galois_keys) const {
            Ciphertext destination;
            rotate_columns(encrypted, galois_keys, destination);
            return destination;
        }

        inline void rotate_vector_inplace(Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::CKKS) {
                rotate_inplace_internal(encrypted, steps, galois_keys);
            } else {
                throw std::invalid_argument("[Evaluator::rotate_vector_inplace] Rotate vector only applies for CKKS");
            }
        }
        inline void rotate_vector(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination) const {
            destination = encrypted;
            rotate_vector_inplace(destination, steps, galois_keys);
        }
        inline Ciphertext rotate_vector_new(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys) const {
            Ciphertext destination;
            rotate_vector(encrypted, steps, galois_keys, destination);
            return destination;
        }

        inline void complex_conjugate_inplace(Ciphertext& encrypted, const GaloisKeys& galois_keys) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::CKKS) {
                conjugate_inplace_internal(encrypted, galois_keys);
            } else {
                throw std::invalid_argument("[Evaluator::complex_conjugate_inplace] Complex conjugate only applies for CKKS");
            }
        }
        inline void complex_conjugate(const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination) const {
            destination = encrypted;
            complex_conjugate_inplace(destination, galois_keys);
        }
        inline Ciphertext complex_conjugate_new(const Ciphertext& encrypted, const GaloisKeys& galois_keys) const {
            Ciphertext destination;
            complex_conjugate(encrypted, galois_keys, destination);
            return destination;
        }

        // Pack LWE utilities

        LWECiphertext extract_lwe_new(const Ciphertext& encrypted, size_t term) const;
        inline Ciphertext assemble_lwe_new(const LWECiphertext& lwe_encrypted) const {
            return lwe_encrypted.assemble_lwe();
        }
        
        void field_trace_inplace(Ciphertext& encrypted, const GaloisKeys& automorphism_keys, size_t logn) const;
        
        void divide_by_poly_modulus_degree_inplace(Ciphertext& encrypted, uint64_t mul = 1) const;
        
        Ciphertext pack_lwe_ciphertexts_new(const std::vector<LWECiphertext>& lwe_encrypted, const GaloisKeys& automorphism_keys) const;

        void negacyclic_shift(const Ciphertext& encrypted, size_t shift, Ciphertext& destination) const;
        inline Ciphertext negacyclic_shift_new(const Ciphertext& encrypted, size_t shift) const {
            Ciphertext destination;
            negacyclic_shift(encrypted, shift, destination);
            return destination;
        }
        inline void negacyclic_shift_inplace(Ciphertext& encrypted, size_t shift) const {
            Ciphertext cloned = encrypted;
            negacyclic_shift(cloned, shift, encrypted);
        }
    };

}