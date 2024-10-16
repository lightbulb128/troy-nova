#pragma once
#include <mutex>
#include <shared_mutex>
#include "batch_utils.h"
#include "he_context.h"
#include "key.h"
#include "key_generator.h"

namespace troy {

    class Decryptor {

    private:

        HeContextPointer context_;
        mutable utils::DynamicArray<uint64_t> secret_key_array_;
        mutable std::shared_mutex secret_key_array_mutex;

        void dot_product_ct_sk_array(const Ciphertext& encrypted, utils::Slice<uint64_t> destination, MemoryPoolHandle pool) const;
        void dot_product_ct_sk_array_batched(const std::vector<const Ciphertext*>& encrypted, const utils::SliceVec<uint64_t>& destination, MemoryPoolHandle pool) const;

        void bfv_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const;
        void bfv_decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const;

        void ckks_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const;
        void ckks_decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const;

        void bgv_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const;
        void bgv_decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const;
    
    public:
        
        Decryptor(HeContextPointer context, const SecretKey& secret_key, MemoryPoolHandle pool = MemoryPool::GlobalPool());
        Decryptor(const Decryptor& copy) = delete;
        inline Decryptor(Decryptor&& source) {
            std::unique_lock<std::shared_mutex> lock(secret_key_array_mutex);
            context_ = source.context_;
            secret_key_array_ = std::move(source.secret_key_array_);
            lock.unlock();
        }

        inline bool on_device() const {return secret_key_array_.on_device();}
        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {secret_key_array_.to_device_inplace(pool);}

        inline utils::DynamicArray<uint64_t>& debug_secret_key_array() const {return secret_key_array_;}

        HeContextPointer context() const {return context_;}

        void decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        void decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext decrypt_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            decrypt(encrypted, destination, pool);
            return destination;
        }
        inline std::vector<Plaintext> decrypt_batched_new(const std::vector<const Ciphertext*>& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Plaintext> destination(encrypted.size());
            decrypt_batched(encrypted, batch_utils::collect_pointer(destination), pool);
            return destination;
        }

        void bfv_decrypt_without_scaling_down(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        void bfv_decrypt_without_scaling_down_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        inline Plaintext bfv_decrypt_without_scaling_down_new(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext destination;
            bfv_decrypt_without_scaling_down(encrypted, destination, pool);
            return destination;
        }
        inline std::vector<Plaintext> bfv_decrypt_without_scaling_down_batched_new(const std::vector<const Ciphertext*>& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Plaintext> destination(encrypted.size());
            bfv_decrypt_without_scaling_down_batched(encrypted, batch_utils::collect_pointer(destination), pool);
            return destination;
        }

        size_t invariant_noise_budget(const Ciphertext& encrypted, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

    };

}