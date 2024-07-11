#pragma once
#include "plaintext.h"
#include "ciphertext.h"

namespace troy {

    class SecretKey {

    private:
        Plaintext sk;

    public:

        inline MemoryPoolHandle pool() const { return sk.pool(); }
        inline size_t device_index() const { return sk.device_index(); }

        inline SecretKey() {}
        inline SecretKey(const Plaintext& sk): sk(sk.clone(sk.pool())) {}
        inline SecretKey(Plaintext&& sk): sk(std::move(sk)) {}

        inline bool on_device() const {
            return sk.on_device();
        }

        inline SecretKey clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            return SecretKey(sk.clone(pool));
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            sk.to_device_inplace(pool);
        }

        inline void to_host_inplace() {
            sk.to_host_inplace();
        }

        inline SecretKey to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SecretKey cloned = this->clone(pool);
            cloned.to_device_inplace(pool);
            return cloned;
        }

        inline SecretKey to_host() const {
            SecretKey cloned = this->clone(pool());
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

        inline size_t save(std::ostream& stream, CompressionMode mode = CompressionMode::Nil) const {
            return sk.save(stream, mode);
        }
        inline void load(std::istream& stream, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            sk.load(stream, pool);
        }
        inline static SecretKey load_new(std::istream& stream, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            SecretKey sk;
            sk.load(stream, pool);
            return sk;
        }
        inline size_t serialized_size_upperbound(CompressionMode mode = CompressionMode::Nil) const {
            return sk.serialized_size_upperbound(mode);
        }

    };

    class PublicKey {

    private:
        Ciphertext pk;

    public:

        inline MemoryPoolHandle pool() const { return pk.pool(); }
        inline size_t device_index() const { return pk.device_index(); }

        inline PublicKey() {}
        inline PublicKey(const Ciphertext& pk): pk(pk.clone(pk.pool())) {}
        inline PublicKey(Ciphertext&& pk): pk(std::move(pk)) {}

        inline bool on_device() const {
            return pk.on_device();
        }

        inline PublicKey clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            return PublicKey(pk.clone(pool));
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            pk.to_device_inplace(pool);
        }

        inline void to_host_inplace() {
            pk.to_host_inplace();
        }

        inline PublicKey to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            PublicKey cloned = this->clone(pool);
            cloned.to_device_inplace(pool);
            return cloned;
        }

        inline PublicKey to_host() const {
            PublicKey cloned = this->clone(pool());
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

        inline size_t save(std::ostream& stream, HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const {
            return pk.save(stream, context, mode);
        }
        inline void load(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            pk.load(stream, context, pool);
        }
        inline static PublicKey load_new(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            Ciphertext result;
            result.load(stream, context, pool);
            return result;
        }
        inline size_t serialized_size_upperbound(HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const {
            return pk.serialized_size_upperbound(context, mode);
        }
    };

}