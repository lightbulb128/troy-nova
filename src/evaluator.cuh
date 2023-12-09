#pragma once
#include "he_context.cuh"
#include "plaintext.cuh"
#include "ciphertext.cuh"
#include "kswitch_keys.cuh"
#include "utils/scaling_variant.cuh"
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


    };

}