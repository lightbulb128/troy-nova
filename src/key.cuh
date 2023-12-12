#pragma once
#include "plaintext.cuh"
#include "ciphertext.cuh"

namespace troy {

    class SecretKey {

    private:
        Plaintext sk;

    public:

        inline SecretKey() {}

        inline SecretKey(const Plaintext& sk): sk(sk.clone()) {}

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

        inline void save(std::ostream& stream) const {
            sk.save(stream);
        }
        inline void load(std::istream& stream) {
            sk.load(stream);
        }
        inline static SecretKey load_new(std::istream& stream) {
            SecretKey sk;
            sk.load(stream);
            return sk;
        }
        inline size_t serialized_size() const {
            return sk.serialized_size();
        }

    };

    class PublicKey {

    private:
        Ciphertext pk;

    public:
        inline PublicKey() {}
        inline PublicKey(const Ciphertext& pk): pk(pk.clone()) {}

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

        inline bool contains_seed() const {
            return pk.contains_seed();
        }
        inline void expand_seed(HeContextPointer context) {
            pk.expand_seed(context);
        }

        inline void save(std::ostream& stream, HeContextPointer context) const {
            pk.save(stream, context);
        }
        inline void load(std::istream& stream, HeContextPointer context) {
            pk.load(stream, context);
        }
        inline static PublicKey load_new(std::istream& stream, HeContextPointer context) {
            Ciphertext result;
            result.load(stream, context);
            return result;
        }
        inline size_t serialized_size(HeContextPointer context) const {
            return pk.serialized_size(context);
        }
    };

}