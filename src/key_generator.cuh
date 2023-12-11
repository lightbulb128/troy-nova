#pragma once
#include <shared_mutex>
#include <mutex>
#include "key.cuh"
#include "kswitch_keys.cuh"
#include "he_context.cuh"
#include "utils/rlwe.cuh"

namespace troy {

    class KeyGenerator {

    private:
        HeContextPointer context_;
        SecretKey secret_key_;
        
        mutable utils::DynamicArray<uint64_t> secret_key_array_;
        mutable std::shared_mutex secret_key_array_mutex;

        void create_secret_key_array();
        PublicKey generate_pk(bool save_seed, utils::RandomGenerator* u_prng) const;
        
        void compute_secret_key_array(size_t max_power) const;
        void generate_one_kswitch_key(utils::ConstSlice<uint64_t> new_key, std::vector<PublicKey>& destination, bool save_seed) const;
        void generate_kswitch_keys(utils::ConstSlice<uint64_t> new_keys, size_t num_keys, KSwitchKeys& destination, bool save_seed) const;
        RelinKeys generate_rlk(size_t count, bool save_seed) const;
        GaloisKeys generate_galois_keys(const std::vector<size_t>& galois_elements, bool save_seed) const;

    public:

        inline bool on_device() const {
            return secret_key_array_.on_device();
        }

        inline void to_device_inplace() {
            std::unique_lock<std::shared_mutex> lock(secret_key_array_mutex);
            secret_key_array_.to_device_inplace();
            lock.unlock();
            secret_key_.to_device_inplace();
        }

        inline HeContextPointer context() const {
            return context_;
        }

        inline const SecretKey& secret_key() const {
            return secret_key_;
        }
        
        KeyGenerator(HeContextPointer context);
        KeyGenerator(HeContextPointer context, const SecretKey& secret_key);

        inline PublicKey create_public_key(bool save_seed) const {
            return generate_pk(save_seed, nullptr);
        }

        inline PublicKey create_public_key_with_u_prng(bool save_seed, utils::RandomGenerator& u_prng) const {
            return generate_pk(save_seed, &u_prng);
        }

        static void compute_secret_key_powers(HeContextPointer context, size_t max_power, utils::DynamicArray<uint64_t>& secret_key_array);
    
        KSwitchKeys create_keyswitching_key(const SecretKey& new_key, bool save_seed) const;
        inline RelinKeys create_relin_keys(bool save_seed, size_t max_power = 2) const {
            if (max_power < 2) {
                throw std::invalid_argument("[KeyGenerator::create_relin_keys] max_power must be at least 2");
            }
            return this->generate_rlk(max_power - 1, save_seed);
        }

        inline GaloisKeys create_galois_keys_from_elements(const std::vector<size_t>& galois_elements, bool save_seed) const {
            return generate_galois_keys(galois_elements, save_seed);
        }

        inline GaloisKeys create_galois_keys_from_steps(const std::vector<int>& steps, bool save_seed) const {
            if (!this->context()->key_context_data().value()->qualifiers().using_batching) {
                throw std::invalid_argument("[KeyGenerator::create_galois_keys_from_steps] batching must be enabled");
            }
            const utils::GaloisTool& galois_tool = this->context()->key_context_data().value()->galois_tool();
            std::vector<size_t> galois_elements = galois_tool.get_elements_from_steps(steps);
            return generate_galois_keys(galois_elements, save_seed);
        }

        inline GaloisKeys create_galois_keys(bool save_seed) const {
            if (!this->context()->key_context_data().value()->qualifiers().using_batching) {
                throw std::invalid_argument("[KeyGenerator::create_galois_keys] batching must be enabled");
            }
            const utils::GaloisTool& galois_tool = this->context()->key_context_data().value()->galois_tool();
            std::vector<size_t> galois_elements = galois_tool.get_elements_all();
            return generate_galois_keys(galois_elements, save_seed);
        }

        inline GaloisKeys create_automorphism_keys(bool save_seed) const {
            size_t poly_degree = this->context()->key_context_data().value()->parms().poly_modulus_degree();
            std::vector<size_t> galois_elements;
            while (poly_degree >= 2) {
                galois_elements.push_back(poly_degree + 1);
                poly_degree >>= 1;
            }
            return this->create_galois_keys_from_elements(galois_elements, save_seed);
        }
        
    };

}