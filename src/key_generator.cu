#include "key_generator.cuh"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::ConstPointer;
    using utils::Array;
    using utils::NTTTables;

    void KeyGenerator::create_secret_key_array() {
        // lock
        std::unique_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
        // create secret key array
        ContextDataPointer context_data = this->context_->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        bool device = this->secret_key_.on_device();
        if (device) this->secret_key_array_.to_device_inplace();
        else this->secret_key_array_.to_host_inplace();
        this->secret_key_array_.resize(coeff_count * coeff_modulus_size);
        this->secret_key_array_.copy_from_slice(this->secret_key_.data().const_reference());
        // unlock
        lock.unlock();
    }

    KeyGenerator::KeyGenerator(HeContextPointer context):
        context_(context)
    {
        // sample secret key
        bool device = context->on_device();
        Plaintext& sk = this->secret_key_.as_plaintext();
        utils::RandomGenerator& rng = context->random_generator();
        ContextDataPointer context_data = context->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        
        if (device) sk.to_device_inplace();
        else sk.to_host_inplace();

        sk.resize(coeff_count * coeff_modulus_size);
        rng.sample_poly_ternary(sk.poly(), coeff_count, coeff_modulus);

        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::ntt_negacyclic_harvey_p(
            sk.poly(), coeff_count, ntt_tables
        );

        sk.parms_id() = context_data->parms_id();
        this->create_secret_key_array();
    }

    KeyGenerator::KeyGenerator(HeContextPointer context, const SecretKey& secret_key):
        context_(context), secret_key_(secret_key.clone())
    {
        this->create_secret_key_array();
    }

    PublicKey KeyGenerator::generate_pk(bool save_seed, utils::RandomGenerator* u_prng) const {
        ContextDataPointer context_data = this->context()->key_context_data().value();
        PublicKey public_key;
        if (u_prng == nullptr) {
            rlwe::symmetric(
                this->secret_key(), this->context(), context_data->parms_id(), 
                true, save_seed, public_key.as_ciphertext()
            );
        } else {
            rlwe::symmetric_with_c1_prng(
                this->secret_key(), this->context(), context_data->parms_id(), 
                true, *u_prng, save_seed, public_key.as_ciphertext()
            );
        }
        public_key.parms_id() = context_data->parms_id();
        return public_key;
    }

    void KeyGenerator::compute_secret_key_powers(HeContextPointer context, size_t max_power, utils::DynamicArray<uint64_t>& secret_key_array) {
        ContextDataPointer context_data = context->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        
        // sanity check: array length must be a multiple of (coeff_count * coeff_modulus_size)
        size_t poly_size = coeff_count * coeff_modulus_size;
        if (secret_key_array.size() % poly_size != 0 || secret_key_array.size() == 0) {
            throw std::invalid_argument("[static KeyGenerator::compute_secret_key_powers] secret_key_array size must be a positive multiple of (coeff_count * coeff_modulus_size)");
        }

        size_t old_size = secret_key_array.size() / poly_size;
        size_t new_size = std::max(old_size, max_power);
        if (old_size == new_size) return;

        // Need to extend the array
        // Compute powers of secret key until max_power
        secret_key_array.resize(new_size * poly_size);
        for (size_t i = 0; i < new_size - old_size; i++) {
            ConstSlice<uint64_t> last = secret_key_array.const_slice((old_size + i - 1) * poly_size, (old_size + i) * poly_size);
            Slice<uint64_t> next = secret_key_array.slice((old_size + i) * poly_size, (old_size + i + 1) * poly_size);
            ConstSlice<uint64_t> first = secret_key_array.const_slice(0, poly_size);
            utils::dyadic_product_p(last, first, coeff_count, coeff_modulus, next);
        }
    }

}