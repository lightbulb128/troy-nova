#include "key_generator.h"
#include <thread>

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::ConstPointer;
    using utils::Array;
    using utils::NTTTables;
    using utils::GaloisTool;

    void KeyGenerator::create_secret_key_array(MemoryPoolHandle pool) {
        // lock
        std::unique_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
        // create secret key array
        ContextDataPointer context_data = this->context_->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        bool device = this->secret_key_.on_device();
        if (device) this->secret_key_array_.to_device_inplace(pool);
        else this->secret_key_array_.to_host_inplace();
        this->secret_key_array_.resize(coeff_count * coeff_modulus_size, true);
        this->secret_key_array_.copy_from_slice(this->secret_key_.data().const_reference());
        // unlock
        lock.unlock();
    }

    KeyGenerator::KeyGenerator(HeContextPointer context, MemoryPoolHandle pool):
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
        
        if (device) sk.to_device_inplace(pool);
        else sk.to_host_inplace();

        sk.resize(coeff_count * coeff_modulus_size);
        rng.sample_poly_ternary(sk.poly(), coeff_count, coeff_modulus);

        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::ntt_inplace_p(
            sk.poly(), coeff_count, ntt_tables
        );

        sk.resize_rns(*context, context_data->parms_id());
        sk.is_ntt_form() = true;
        this->create_secret_key_array(pool);
    }

    KeyGenerator::KeyGenerator(HeContextPointer context, const SecretKey& secret_key, MemoryPoolHandle pool):
        context_(context), secret_key_(secret_key.clone(pool))
    {
        this->create_secret_key_array(pool);
    }

    PublicKey KeyGenerator::generate_pk(bool save_seed, utils::RandomGenerator* u_prng, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->context()->key_context_data().value();
        PublicKey public_key;
        if (u_prng == nullptr) {
            rlwe::symmetric(
                this->secret_key(), this->context(), context_data->parms_id(), 
                true, save_seed, public_key.as_ciphertext(), pool
            );
        } else {
            rlwe::symmetric_with_c1_prng(
                this->secret_key(), this->context(), context_data->parms_id(), 
                true, *u_prng, save_seed, public_key.as_ciphertext(), pool
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
        secret_key_array.resize(new_size * poly_size, true);
        for (size_t i = 0; i < new_size - old_size; i++) {
            ConstSlice<uint64_t> last = secret_key_array.const_slice((old_size + i - 1) * poly_size, (old_size + i) * poly_size);
            Slice<uint64_t> next = secret_key_array.slice((old_size + i) * poly_size, (old_size + i + 1) * poly_size);
            ConstSlice<uint64_t> first = secret_key_array.const_slice(0, poly_size);
            utils::dyadic_product_p(last, first, coeff_count, coeff_modulus, next);
        }
    }
    
    void KeyGenerator::compute_secret_key_array(size_t max_power) const {
        ContextDataPointer context_data = this->context()->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // acquire read lock
        std::shared_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
        // check if we need to compute more powers
        size_t old_size = this->secret_key_array_.size() / (coeff_count * coeff_modulus_size);
        // release read lock
        lock.unlock();
        size_t new_size = std::max(old_size, max_power);
        if (old_size == new_size) return;

        // Need to extend the array
        // acquire write lock
        std::unique_lock<std::shared_mutex> lock2(this->secret_key_array_mutex);
        KeyGenerator::compute_secret_key_powers(this->context(), max_power, this->secret_key_array_);
        // release write lock
        lock2.unlock();
    }
    
    void KeyGenerator::generate_one_kswitch_key(utils::ConstSlice<uint64_t> new_key, std::vector<PublicKey>& destination, bool save_seed, MemoryPoolHandle pool) const {
        if (!this->context()->using_keyswitching()) {
            throw std::logic_error("[KeyGenerator::generate_one_kswitch_key] Keyswitching is not enabled.");
        }
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const EncryptionParameters& key_parms = key_context_data->parms();
        size_t coeff_count = key_parms.poly_modulus_degree();
        ConstSlice<Modulus> key_modulus = key_parms.coeff_modulus();
        Array<Modulus> key_modulus_host = Array<Modulus>::create_and_copy_from_slice(key_modulus, false, nullptr);
        size_t decomp_mod_count = this->context()->first_context_data().value()->parms().coeff_modulus().size();
        ParmsID key_parms_id = key_context_data->parms_id();

        Array<uint64_t> temp(coeff_count, this->on_device(), pool);
        destination.resize(decomp_mod_count);
        for (size_t i = 0; i < decomp_mod_count; i++) {
            rlwe::symmetric(this->secret_key(), this->context(), key_parms_id, true, save_seed, destination[i].as_ciphertext(), pool);
            uint64_t factor = utils::barrett_reduce_uint64(key_modulus_host[key_modulus_host.size() - 1].value(), key_modulus_host[i]);
            utils::multiply_scalar(new_key.const_slice(i * coeff_count, (i + 1) * coeff_count), factor, key_modulus.at(i), temp.reference());
            Slice<uint64_t> destination_component = destination[i].as_ciphertext().poly_component(0, i);
            utils::add_inplace(destination_component, temp.const_reference(), key_modulus.at(i));
        }
    }
    
    KSwitchKeys KeyGenerator::create_keyswitching_key(const SecretKey& new_key, bool save_seed, MemoryPoolHandle pool) const {
        KSwitchKeys ret;
        ret.data().resize(1);
        this->generate_one_kswitch_key(
            new_key.as_plaintext().poly(),
            ret.data()[0],
            save_seed, pool
        );
        ret.parms_id() = this->context()->key_parms_id();
        ret.build_key_data_ptrs(pool);
        return ret;
    }
    
    void KeyGenerator::generate_kswitch_keys(utils::ConstSlice<uint64_t> new_keys, size_t num_keys, KSwitchKeys& destination, bool save_seed, MemoryPoolHandle pool) const {
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const EncryptionParameters& key_parms = key_context_data->parms();
        size_t coeff_count = key_parms.poly_modulus_degree();
        ConstSlice<Modulus> key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();
        destination.data().resize(num_keys);
        size_t d = coeff_count * key_modulus_size;
        for (size_t i = 0; i < num_keys; i++) {
            this->generate_one_kswitch_key(
                new_keys.const_slice(i * d, (i + 1) * d),
                destination.data()[i],
                save_seed,
                pool
            );
        }
        destination.build_key_data_ptrs(pool);
    }

    RelinKeys KeyGenerator::generate_rlk(size_t count, bool save_seed, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->context()->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t modulus_size = coeff_modulus.size();

        // Make sure we have enough secret keys computed
        this->compute_secret_key_array(count + 1);
        
        // Create the RelinKeys object to return
        RelinKeys relin_keys;
        
        // Assume the secret key is already transformed into NTT form.
        size_t d = coeff_count * modulus_size;

        // Acquire read lock
        std::shared_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
        this->generate_kswitch_keys(
            this->secret_key_array_.const_slice(d, (count + 1) * d),
            count,
            relin_keys.as_kswitch_keys(),
            save_seed, pool
        );

        // Release read lock
        lock.unlock();

        // Set the parms_id
        relin_keys.parms_id() = context_data->parms_id();
        return relin_keys;
    }

    GaloisKeys KeyGenerator::generate_galois_keys(const std::vector<size_t>& galois_elements, bool save_seed, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->context()->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        const GaloisTool& galois_tool = context_data->galois_tool();

        // Create the GaloisKeys object to return
        GaloisKeys galois_keys;
        galois_keys.as_kswitch_keys().data().resize(coeff_count);

        Array<uint64_t> rotated_secret_key(coeff_count * coeff_modulus_size, this->on_device(), pool);
        for (size_t galois_element: galois_elements) {
            if (galois_element % 2 == 0 || galois_element >= (coeff_count << 1)) {
                throw std::invalid_argument("[KeyGenerator::generate_galois_keys] Galois element is not valid.");
            }
            if (galois_keys.has_key(galois_element)) {
                continue;
            }
            galois_tool.apply_ntt_p(
                this->secret_key().data().const_reference(),
                coeff_modulus_size, galois_element,
                rotated_secret_key.reference(), pool
            );
            size_t index = GaloisKeys::get_index(galois_element);
            this->generate_one_kswitch_key(
                rotated_secret_key.const_reference(),
                galois_keys.as_kswitch_keys().data()[index],
                save_seed, pool
            );
        }
        galois_keys.as_kswitch_keys().build_key_data_ptrs(pool);

        // Set the parms_id
        galois_keys.parms_id() = context_data->parms_id();
        return galois_keys;
    }
}