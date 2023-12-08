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

        void dot_product_ct_sk_array(const Ciphertext& encrypted, utils::Slice<uint64_t> destination) const;
        void bfv_decrypt(const Ciphertext& encrypted, Plaintext& destination) const;
        void ckks_decrypt(const Ciphertext& encrypted, Plaintext& destination) const;
        void bgv_decrypt(const Ciphertext& encrypted, Plaintext& destination) const;
    
    public:
        
        Decryptor(HeContextPointer context, const SecretKey& secret_key);

        inline bool on_device() const {return secret_key_array_.on_device();}
        inline void to_device_inplace() {secret_key_array_.to_device_inplace();}

        inline utils::DynamicArray<uint64_t>& debug_secret_key_array() const {return secret_key_array_;}

        HeContextPointer context() const {return context_;}

        void decrypt(const Ciphertext& encrypted, Plaintext& destination) const;
        inline Plaintext decrypt_new(const Ciphertext& encrypted) const {
            Plaintext destination;
            decrypt(encrypted, destination);
            return destination;
        }

    };

}