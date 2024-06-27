#pragma once
#include <mutex>
#include <shared_mutex>
#include "he_context.cuh"
#include "key.cuh"
#include "key_generator.cuh"

namespace troy {

    class Decryptor {

    private:

        HeContextPointer context_;
        mutable utils::DynamicArray<uint64_t> secret_key_array_;
        mutable std::shared_mutex secret_key_array_mutex;

        void dot_product_ct_sk_array(const Ciphertext& encrypted, utils::Slice<uint64_t> destination, MemoryPoolHandle pool) const;
        void bfv_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const;
        void ckks_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const;
        void bgv_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const;
    
    public:
        
        Decryptor(HeContextPointer context, const SecretKey& secret_key, MemoryPoolHandle pool = MemoryPool::GlobalPool());

        inline bool on_device() const {return secret_key_array_.on_device();}
        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {secret_key_array_.to_device_inplace(pool);}

        inline utils::DynamicArray<uint64_t>& debug_secret_key_array() const {return secret_key_array_;}

        HeContextPointer context() const {return context_;}

        void decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext decrypt_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            decrypt(encrypted, destination);
            return destination;
        }

        void bfv_decrypt_without_scaling_down(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext bfv_decrypt_without_scaling_down_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            bfv_decrypt_without_scaling_down(encrypted, destination);
            return destination;
        }

        size_t invariant_noise_budget(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

    };

}