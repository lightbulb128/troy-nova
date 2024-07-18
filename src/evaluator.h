#pragma once
#include "he_context.h"
#include "plaintext.h"
#include "ciphertext.h"
#include "kswitch_keys.h"
#include "utils/memory_pool.h"
#include "utils/scaling_variant.h"
#include "lwe_ciphertext.h"
#include <string>

namespace troy {

    class Evaluator {
        HeContextPointer context_;

        ContextDataPointer get_context_data(const char* prompt, const ParmsID& encrypted) const;

        void translate_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, bool subtract, MemoryPoolHandle pool) const;
        
        void bfv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool) const;
        void ckks_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool) const;
        void bgv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool) const;

        void bfv_square_inplace(Ciphertext& encrypted, MemoryPoolHandle pool) const;
        void ckks_square_inplace(Ciphertext& encrypted, MemoryPoolHandle pool) const;
        void bgv_square_inplace(Ciphertext& encrypted, MemoryPoolHandle pool) const;

        /// Suppose kswitch_keys[kswitch_kes_index] is generated with s' on a KeyGenerator of secret key s.
        /// Then the semantic of this function is as follows: `target` is supposed to multiply with s' to contribute to the
        /// decrypted result, now we apply this function, to decompose (target * s') into (c0, c1) such that c0 + c1 * s = target * s.
        /// And then we add c0, c1 to the original c0, c1 in the `encrypted`.
        void switch_key_inplace_internal(Ciphertext& encrypted, utils::ConstSlice<uint64_t> target, const KSwitchKeys& kswitch_keys, size_t kswitch_keys_index, MemoryPoolHandle pool) const;

        void relinearize_inplace_internal(Ciphertext& encrypted, const RelinKeys& relin_keys, size_t destination_size, MemoryPoolHandle pool) const;

        void mod_switch_scale_to_next_internal(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const;
        void mod_switch_drop_to_next_internal(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const;
        void mod_switch_drop_to_next_plain_inplace_internal(Plaintext& plain) const;

        void translate_plain_inplace(Ciphertext& encrypted, const Plaintext& plain, bool subtract, MemoryPoolHandle pool) const;

        void multiply_plain_normal_inplace(Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandle pool) const;
        void multiply_plain_ntt_inplace(Ciphertext& encrypted, const Plaintext& plain) const;

        void rotate_inplace_internal(Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandle pool) const;
        void conjugate_inplace_internal(Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandle pool) const;

    public:
        inline Evaluator(HeContextPointer context): context_(context) {}
        inline HeContextPointer context() const { return context_; }
        inline bool on_device() const {return this->context()->on_device();}

        void negate_inplace(Ciphertext& encrypted) const;
        inline void negate(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            negate_inplace(destination);
        }
        inline Ciphertext negate_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination = encrypted.clone(pool);
            negate_inplace(destination);
            return destination;
        }

        inline void add_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            translate_inplace(encrypted1, encrypted2, false, pool);
        }
        inline void add(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted1.clone(pool);
            add_inplace(destination, encrypted2, pool);
        }
        inline Ciphertext add_new(const Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            add(encrypted1, encrypted2, destination, pool);
            return destination;
        }

        inline void sub_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            translate_inplace(encrypted1, encrypted2, true, pool);
        }
        inline void sub(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted1.clone(pool);
            sub_inplace(destination, encrypted2);
        }
        inline Ciphertext sub_new(const Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            sub(encrypted1, encrypted2, destination, pool);
            return destination;
        }

        void multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void multiply(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted1.clone(pool);
            multiply_inplace(destination, encrypted2, pool);
        }
        inline Ciphertext multiply_new(const Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            multiply(encrypted1, encrypted2, destination, pool);
            return destination;
        }

        void square_inplace(Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void square(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            square_inplace(destination, pool);
        }
        inline Ciphertext square_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            square(encrypted, destination, pool);
            return destination;
        }

        void apply_keyswitching_inplace(Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void apply_keyswitching(const Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            apply_keyswitching_inplace(destination, kswitch_keys, pool);
        }
        inline Ciphertext apply_keyswitching_new(const Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            apply_keyswitching(encrypted, kswitch_keys, destination, pool);
            return destination;
        }

        inline void relinearize_inplace(Ciphertext& encrypted, const RelinKeys& relin_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            relinearize_inplace_internal(encrypted, relin_keys, 2, pool);
        }
        inline void relinearize(const Ciphertext& encrypted, const RelinKeys& relin_keys, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            relinearize_inplace(destination, relin_keys, pool);
        }
        inline Ciphertext relinearize_new(const Ciphertext& encrypted, const RelinKeys& relin_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            relinearize(encrypted, relin_keys, destination, pool);
            return destination;
        }

        void mod_switch_to_next(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void mod_switch_to_next_inplace(Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext cloned = encrypted.clone(pool);
            mod_switch_to_next(cloned, encrypted, pool);
        }
        inline Ciphertext mod_switch_to_next_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            mod_switch_to_next(encrypted, destination, pool);
            return destination;
        }

        inline void mod_switch_plain_to_next_inplace(Plaintext& plain) const {
            this->mod_switch_drop_to_next_plain_inplace_internal(plain);
        }
        inline void mod_switch_plain_to_next(const Plaintext& plain, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = plain.clone(pool);
            this->mod_switch_drop_to_next_plain_inplace_internal(destination);
        }
        inline Plaintext mod_switch_plain_to_next_new(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination = plain.clone(pool);
            this->mod_switch_drop_to_next_plain_inplace_internal(destination);
            return destination;
        }

        void mod_switch_to_inplace(Ciphertext& encrypted, const ParmsID& parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void mod_switch_to(const Ciphertext& encrypted, const ParmsID& parms_id, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            mod_switch_to_inplace(destination, parms_id, pool);
        }
        inline Ciphertext mod_switch_to_new(const Ciphertext& encrypted, const ParmsID& parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            mod_switch_to(encrypted, parms_id, destination, pool);
            return destination;
        }

        void mod_switch_plain_to_inplace(Plaintext& plain, const ParmsID& parms_id) const;
        inline void mod_switch_plain_to(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = plain.clone(pool);
            mod_switch_plain_to_inplace(destination, parms_id);
        }
        inline Plaintext mod_switch_plain_to_new(const Plaintext& plain, const ParmsID& parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination = plain.clone(pool);
            mod_switch_plain_to_inplace(destination, parms_id);
            return destination;
        }

        void rescale_to_next(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void rescale_to_next_inplace(Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext cloned = encrypted.clone(pool);
            rescale_to_next(cloned, encrypted, pool);
        }
        inline Ciphertext rescale_to_next_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            rescale_to_next(encrypted, destination, pool);
            return destination;
        }

        void rescale_to(const Ciphertext& encrypted, const ParmsID& parms_id, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void rescale_to_inplace(Ciphertext& encrypted, const ParmsID& parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext cloned = encrypted.clone(pool);
            rescale_to(cloned, parms_id, encrypted, pool);
        }
        inline Ciphertext rescale_to_new(const Ciphertext& encrypted, const ParmsID& parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            rescale_to(encrypted, parms_id, destination, pool);
            return destination;
        }

        inline void add_plain_inplace(Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            translate_plain_inplace(encrypted, plain, false, pool);
        }
        inline void add_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            add_plain_inplace(destination, plain, pool);
        }
        inline Ciphertext add_plain_new(const Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            add_plain(encrypted, plain, destination, pool);
            return destination;
        }

        inline void sub_plain_inplace(Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            translate_plain_inplace(encrypted, plain, true, pool);
        }
        inline void sub_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            sub_plain_inplace(destination, plain, pool);
        }
        inline Ciphertext sub_plain_new(const Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            sub_plain(encrypted, plain, destination, pool);
            return destination;
        }

        void multiply_plain_inplace(Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void multiply_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            multiply_plain_inplace(destination, plain, pool);
        }
        inline Ciphertext multiply_plain_new(const Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            multiply_plain(encrypted, plain, destination, pool);
            return destination;
        }

        void transform_plain_to_ntt_inplace(Plaintext& plain, const ParmsID& parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void transform_plain_to_ntt(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = plain.clone(pool);
            transform_plain_to_ntt_inplace(destination, parms_id, pool);
        }
        inline Plaintext transform_plain_to_ntt_new(const Plaintext& plain, const ParmsID& parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination = plain.clone();
            transform_plain_to_ntt_inplace(destination, parms_id, pool);
            return destination;
        }

        void transform_plain_from_ntt_inplace(Plaintext& plain) const;
        inline void transform_plain_from_ntt(const Plaintext& plain, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = plain.clone(pool);
            transform_plain_from_ntt_inplace(destination);
        }
        inline Plaintext transform_plain_from_ntt_new(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination = plain.clone(pool);
            transform_plain_from_ntt_inplace(destination);
            return destination;
        }

        void transform_to_ntt_inplace(Ciphertext& encrypted) const;
        inline void transform_to_ntt(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            transform_to_ntt_inplace(destination);
        }
        inline Ciphertext transform_to_ntt_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            transform_to_ntt(encrypted, destination, pool);
            return destination;
        }

        void transform_from_ntt_inplace(Ciphertext& encrypted) const;
        inline void transform_from_ntt(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            transform_from_ntt_inplace(destination);
        }
        inline Ciphertext transform_from_ntt_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            transform_from_ntt(encrypted, destination, pool);
            return destination;
        }

        void apply_galois_inplace(Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void apply_galois(const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            apply_galois_inplace(destination, galois_element, galois_keys, pool);
        }
        inline Ciphertext apply_galois_new(const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            apply_galois(encrypted, galois_element, galois_keys, destination, pool);
            return destination;
        }

        void apply_galois_plain_inplace(Plaintext& plain, size_t galois_element, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline void apply_galois_plain(const Plaintext& plain, size_t galois_element, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = plain.clone(pool);
            apply_galois_plain_inplace(destination, galois_element, pool);
        }
        inline Plaintext apply_galois_plain_new(const Plaintext& plain, size_t galois_element, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination = plain.clone();
            apply_galois_plain_inplace(destination, galois_element, pool);
            return destination;
        }

        inline void rotate_rows_inplace(Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) {
                rotate_inplace_internal(encrypted, steps, galois_keys, pool);
            } else {
                throw std::invalid_argument("[Evaluator::rotate_rows_inplace] Rotate rows only applies for BFV or BGV");
            }
        }
        inline void rotate_rows(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            rotate_rows_inplace(destination, steps, galois_keys, pool);
        }
        inline Ciphertext rotate_rows_new(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            rotate_rows(encrypted, steps, galois_keys, destination, pool);
            return destination;
        }

        inline void rotate_columns_inplace(Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) {
                conjugate_inplace_internal(encrypted, galois_keys, pool);
            } else {
                throw std::invalid_argument("[Evaluator::rotate_columns_inplace] Rotate columns only applies for BFV or BGV");
            }
        }
        inline void rotate_columns(const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            rotate_columns_inplace(destination, galois_keys, pool);
        }
        inline Ciphertext rotate_columns_new(const Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            rotate_columns(encrypted, galois_keys, destination, pool);
            return destination;
        }

        inline void rotate_vector_inplace(Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::CKKS) {
                rotate_inplace_internal(encrypted, steps, galois_keys, pool);
            } else {
                throw std::invalid_argument("[Evaluator::rotate_vector_inplace] Rotate vector only applies for CKKS");
            }
        }
        inline void rotate_vector(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            rotate_vector_inplace(destination, steps, galois_keys, pool);
        }
        inline Ciphertext rotate_vector_new(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            rotate_vector(encrypted, steps, galois_keys, destination, pool);
            return destination;
        }

        inline void complex_conjugate_inplace(Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
            if (scheme == SchemeType::CKKS) {
                conjugate_inplace_internal(encrypted, galois_keys, pool);
            } else {
                throw std::invalid_argument("[Evaluator::complex_conjugate_inplace] Complex conjugate only applies for CKKS");
            }
        }
        inline void complex_conjugate(const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            destination = encrypted.clone(pool);
            complex_conjugate_inplace(destination, galois_keys, pool);
        }
        inline Ciphertext complex_conjugate_new(const Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            complex_conjugate(encrypted, galois_keys, destination, pool);
            return destination;
        }

        // Pack LWE utilities

        LWECiphertext extract_lwe_new(const Ciphertext& encrypted, size_t term, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Ciphertext assemble_lwe_new(const LWECiphertext& lwe_encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            return lwe_encrypted.assemble_lwe(pool);
        }
        
        void field_trace_inplace(Ciphertext& encrypted, const GaloisKeys& automorphism_keys, size_t logn, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        
        void divide_by_poly_modulus_degree_inplace(Ciphertext& encrypted, uint64_t mul = 1) const;
        
        Ciphertext pack_lwe_ciphertexts_new(const std::vector<LWECiphertext>& lwe_encrypted, const GaloisKeys& automorphism_keys, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void negacyclic_shift(const Ciphertext& encrypted, size_t shift, Ciphertext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Ciphertext negacyclic_shift_new(const Ciphertext& encrypted, size_t shift, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            negacyclic_shift(encrypted, shift, destination, pool);
            return destination;
        }
        inline void negacyclic_shift_inplace(Ciphertext& encrypted, size_t shift, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext cloned = encrypted.clone(pool);
            negacyclic_shift(cloned, shift, encrypted, pool);
        }
    };

}