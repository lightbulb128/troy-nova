#pragma once
#include "batch_utils.h"
#include "encryption_parameters.h"
#include "key.h"
#include "he_context.h"
#include "utils/memory_pool.h"
#include "utils/rlwe.h"
#include "utils/scaling_variant.h"

namespace troy {

    class Encryptor {

    private:
        HeContextPointer context_;
        std::optional<PublicKey> public_key_;
        std::optional<SecretKey> secret_key_;

        void encrypt_zero_internal(
            const ParmsID& parms_id, 
            bool is_ntt_form,
            bool is_asymmetric, bool save_seed, 
            utils::RandomGenerator* u_prng, 
            Ciphertext& destination,
            MemoryPoolHandle pool) const;

        void encrypt_zero_internal_batched(
            const ParmsID& parms_id, 
            bool is_ntt_form,
            bool is_asymmetric, bool save_seed, 
            utils::RandomGenerator* u_prng, 
            const std::vector<Ciphertext*>& destination,
            MemoryPoolHandle pool) const;

        void encrypt_internal(
            const Plaintext& plain,
            bool is_asymmetric, bool save_seed,
            utils::RandomGenerator* u_prng,
            Ciphertext& destination,
            MemoryPoolHandle pool) const;
            
        void encrypt_internal_batched(
            const std::vector<const Plaintext*>& plain,
            bool is_asymmetric, bool save_seed,
            utils::RandomGenerator* u_prng,
            const std::vector<Ciphertext*>& destination,
            MemoryPoolHandle pool) const;

    public:

        inline Encryptor(HeContextPointer context) : context_(context) {}

        inline HeContextPointer context() const {
            return context_;
        }

        inline bool on_device() const {
            // if only one has value, return that one's on_device
            if (public_key_.has_value() && !secret_key_.has_value()) {
                return public_key_.value().on_device();
            }
            if (!public_key_.has_value() && secret_key_.has_value()) {
                return secret_key_.value().on_device();
            }
            // if neither has value, raise error
            if (!public_key_.has_value() && !secret_key_.has_value()) {
                throw std::runtime_error("[Encryptor::on_device] Encryptor has no keys");
            }
            // if both has value, check same
            if (public_key_.value().on_device() != secret_key_.value().on_device()) {
                throw std::runtime_error("[Encryptor::on_device] public key and secret key are not on the same device");
            }
            return public_key_.value().on_device();
        }
        
        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            if (public_key_.has_value()) {
                public_key_.value().to_device_inplace(pool);
            }
            if (secret_key_.has_value()) {
                secret_key_.value().to_device_inplace(pool);
            }
        }

        inline const PublicKey& public_key() const {
            if (!public_key_.has_value()) {
                throw std::runtime_error("[Encryptor::public_key] Encryptor has no public key");
            }
            return public_key_.value();
        }

        inline const SecretKey& secret_key() const {
            if (!secret_key_.has_value()) {
                throw std::runtime_error("[Encryptor::secret_key] Encryptor has no secret key");
            }
            return secret_key_.value();
        }

        inline void set_public_key(const PublicKey& public_key, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            PublicKey cloned = public_key.clone(pool);
            public_key_ = std::move(cloned);
            if (this->secret_key_.has_value()) {
                if (this->secret_key_.value().on_device() != public_key.on_device()) {
                    throw std::runtime_error("[Encryptor::set_public_key] public key and secret key are not on the same device");
                }
            }
        }

        inline void set_secret_key(const SecretKey& secret_key, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            secret_key_ = secret_key.clone(pool);
            if (this->public_key_.has_value()) {
                if (this->public_key_.value().on_device() != secret_key.on_device()) {
                    throw std::runtime_error("[Encryptor::set_secret_key] public key and secret key are not on the same device");
                }
            }
        }

        // ------------------------
        //    asymmetric single
        // ------------------------

        inline void encrypt_asymmetric(const Plaintext& plain, Ciphertext& destination, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            encrypt_internal(plain, true, false, u_prng, destination, pool);
        }

        inline Ciphertext encrypt_asymmetric_new(const Plaintext& plain, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            encrypt_asymmetric(plain, destination, u_prng, pool);
            return destination;
        }

        inline void encrypt_zero_asymmetric(Ciphertext& destination, std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = context_->first_context_data_pointer()->parms().scheme();
            bool is_ntt_form = scheme == SchemeType::CKKS || scheme == SchemeType::BGV;
            encrypt_zero_internal(parms_id.value_or(context_->first_parms_id()), is_ntt_form, true, false, u_prng, destination, pool);
        }

        inline Ciphertext encrypt_zero_asymmetric_new(std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            encrypt_zero_asymmetric(destination, parms_id, u_prng, pool);
            return destination;
        }


        // ------------------------
        //    asymmetric batched
        // ------------------------

        
        inline void encrypt_asymmetric_batched(const std::vector<const Plaintext*>& plain, const std::vector<Ciphertext*>& destination, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            encrypt_internal_batched(plain, true, false, u_prng, destination, pool);
        }

        inline std::vector<Ciphertext> encrypt_asymmetric_new_batched(const std::vector<const Plaintext*>& plain, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Ciphertext> destination(plain.size());
            encrypt_asymmetric_batched(plain, batch_utils::collect_pointer(destination), u_prng, pool);
            return destination;
        }

        inline void encrypt_zero_asymmetric_batched(const std::vector<Ciphertext*>& destination, std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = context_->first_context_data_pointer()->parms().scheme();
            bool is_ntt_form = scheme == SchemeType::CKKS || scheme == SchemeType::BGV;
            encrypt_zero_internal_batched(parms_id.value_or(context_->first_parms_id()), is_ntt_form, true, false, u_prng, destination, pool);
        }

        inline std::vector<Ciphertext> encrypt_zero_asymmetric_new_batched(size_t count, std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Ciphertext> destination(count);
            encrypt_zero_asymmetric_batched(batch_utils::collect_pointer(destination), parms_id, u_prng, pool);
            return destination;
        }




        // ------------------------
        //    symmetric single
        // ------------------------

        inline void encrypt_symmetric(const Plaintext& plain, bool save_seed, Ciphertext& destination, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            encrypt_internal(plain, false, save_seed, u_prng, destination, pool);
        }

        inline Ciphertext encrypt_symmetric_new(const Plaintext& plain, bool save_seed, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            encrypt_symmetric(plain, save_seed, destination, u_prng, pool);
            return destination;
        }

        inline void encrypt_zero_symmetric(bool save_seed, Ciphertext& destination, std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = context_->first_context_data_pointer()->parms().scheme();
            bool is_ntt_form = scheme == SchemeType::CKKS || scheme == SchemeType::BGV;
            encrypt_zero_internal(parms_id.value_or(context_->first_parms_id()), is_ntt_form, false, save_seed, u_prng, destination, pool);
        }

        inline Ciphertext encrypt_zero_symmetric_new(bool save_seed, std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext destination;
            encrypt_zero_symmetric(save_seed, destination, parms_id, u_prng, pool);
            return destination;
        }

        
        // ------------------------
        //    symmetric batched
        // ------------------------

        inline void encrypt_symmetric_batched(const std::vector<const Plaintext*>& plain, bool save_seed, const std::vector<Ciphertext*>& destination, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            encrypt_internal_batched(plain, false, save_seed, u_prng, destination, pool);
        }

        inline std::vector<Ciphertext> encrypt_symmetric_new_batched(const std::vector<const Plaintext*>& plain, bool save_seed, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Ciphertext> destination(plain.size());
            encrypt_symmetric_batched(plain, save_seed, batch_utils::collect_pointer(destination), u_prng, pool);
            return destination;
        }

        inline void encrypt_zero_symmetric_batched(bool save_seed, const std::vector<Ciphertext*>& destination, std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            SchemeType scheme = context_->first_context_data_pointer()->parms().scheme();
            bool is_ntt_form = scheme == SchemeType::CKKS || scheme == SchemeType::BGV;
            encrypt_zero_internal_batched(parms_id.value_or(context_->first_parms_id()), is_ntt_form, false, save_seed, u_prng, destination, pool);
        }

        inline std::vector<Ciphertext> encrypt_zero_symmetric_new_batched(size_t count, bool save_seed, std::optional<ParmsID> parms_id = std::nullopt, utils::RandomGenerator* u_prng = nullptr, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Ciphertext> destination(count);
            encrypt_zero_symmetric_batched(save_seed, batch_utils::collect_pointer(destination), parms_id, u_prng, pool);
            return destination;
        }

    };

}