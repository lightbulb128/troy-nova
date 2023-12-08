#pragma once
#include "he_context.cuh"
#include "plaintext.cuh"
#include "ciphertext.cuh"
#include <string>

namespace troy {

    class Evaluator {
        HeContextPointer context_;

        ContextDataPointer get_context_data(const char* prompt, const ParmsID& encrypted) const;

        void translate_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, bool subtract) const;
        void bfv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const;
        void ckks_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const;
        void bgv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const;

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

    };

}