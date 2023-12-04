#pragma once
#include "plaintext.cuh"
#include "ciphertext.cuh"

namespace troy {

    class SecretKey {

    private:
        Plaintext sk;

    public:

        inline SecretKey() {}

        inline SecretKey(Plaintext sk): sk(sk.clone()) {}

        inline bool on_device() const {
            return sk.on_device();
        }

        inline SecretKey clone() const {
            return SecretKey(sk.clone());
        }

        inline void to_device_inplace() {
            sk.to_device_inplace();
        }

        inline void to_host_inplace() {
            sk.to_host_inplace();
        }

        inline SecretKey to_device() const {
            SecretKey cloned = this->clone();
            cloned.to_device_inplace();
            return cloned;
        }

        inline SecretKey to_host() const {
            SecretKey cloned = this->clone();
            cloned.to_host_inplace();
            return cloned;
        }

        inline const ParmsID& parms_id() const {
            return sk.parms_id();
        }

        inline ParmsID& parms_id() {
            return sk.parms_id();
        }

        inline const Plaintext& as_plaintext() const {
            return sk;
        }

        inline Plaintext& as_plaintext() {
            return sk;
        }

        inline const utils::DynamicArray<uint64_t>& data() const {
            return sk.data();
        }

        inline utils::DynamicArray<uint64_t>& data() {
            return sk.data();
        }

    };

    class PublicKey {

    private:
        Ciphertext pk;

    public:
        inline PublicKey() {}
        inline PublicKey(Ciphertext pk): pk(pk.clone()) {}

        inline bool on_device() const {
            return pk.on_device();
        }

        inline PublicKey clone() const {
            return PublicKey(pk.clone());
        }

        inline void to_device_inplace() {
            pk.to_device_inplace();
        }

        inline void to_host_inplace() {
            pk.to_host_inplace();
        }

        inline PublicKey to_device() const {
            PublicKey cloned = this->clone();
            cloned.to_device_inplace();
            return cloned;
        }

        inline PublicKey to_host() const {
            PublicKey cloned = this->clone();
            cloned.to_host_inplace();
            return cloned;
        }

        inline const ParmsID& parms_id() const {
            return pk.parms_id();
        }

        inline ParmsID& parms_id() {
            return pk.parms_id();
        }

        inline const Ciphertext& as_ciphertext() const {
            return pk;
        }

        inline Ciphertext& as_ciphertext() {
            return pk;
        }

        inline const utils::DynamicArray<uint64_t>& data() const {
            return pk.data();
        }

        inline utils::DynamicArray<uint64_t>& data() {
            return pk.data();
        }

    };

}