#pragma once
#include "he_context.cuh"
#include "plaintext.cuh"
#include "ciphertext.cuh"
#include "kswitch_keys.cuh"
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

    };

}