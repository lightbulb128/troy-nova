#include "decryptor.cuh"

namespace troy {

    using utils::ConstSlice;
    using utils::NTTTables;
    using utils::Array;

    Decryptor::Decryptor(HeContextPointer context, const SecretKey& secret_key) :
        context_(context) 
    {
        ContextDataPointer key_context_data = context->key_context_data().value();
        const EncryptionParameters& parms = key_context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        if (secret_key.data().size() != coeff_count * coeff_modulus_size)
            throw std::invalid_argument("[Decryptor::Decryptor] secret_key is not valid for encryption parameters");
        this->secret_key_array_ = secret_key.data().clone();
    }

    void Decryptor::dot_product_ct_sk_array(const Ciphertext& encrypted, utils::Slice<uint64_t> destination) const {
        if (!utils::same(this->on_device(), encrypted.on_device(), destination.on_device())) {
            throw std::invalid_argument("[Decryptor::dot_product_ct_sk_array] Arguments are not on the same device.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t encrypted_size = encrypted.polynomial_count();
        size_t key_coeff_modulus_size = this->context()->key_context_data().value()->parms().coeff_modulus().size();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        bool is_ntt_form = encrypted.is_ntt_form();

        // Make sure we have enough secret key powers computed
        if (this->secret_key_array_.size() < (encrypted_size - 1) * coeff_count * key_coeff_modulus_size) {
            std::unique_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
            KeyGenerator::compute_secret_key_powers(this->context(), encrypted_size - 1, this->secret_key_array_);
            lock.unlock();
        }

        // acquire read lock
        std::shared_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
        if (encrypted_size == 2) {
            ConstSlice<uint64_t> c0 = encrypted.poly(0);
            ConstSlice<uint64_t> c1 = encrypted.poly(1);
            ConstSlice<uint64_t> s = this->secret_key_array_.const_slice(0, c0.size());
            if (is_ntt_form) {
                // put < c_1 * s > mod q in destination
                utils::dyadic_product_p(c1, s, coeff_count, coeff_modulus, destination);
                // add c_0 to the result; note that destination should be in the same (NTT) form as encrypted
                utils::add_inplace_p(destination, c0, coeff_count, coeff_modulus);
            } else {
                destination.copy_from_slice(c1);
                utils::ntt_negacyclic_harvey_p(destination, coeff_count, ntt_tables);
                utils::dyadic_product_inplace_p(destination, s, coeff_count, coeff_modulus);
                utils::inverse_ntt_negacyclic_harvey_p(destination, coeff_count, ntt_tables);
                utils::add_inplace_p(destination, c0, coeff_count, coeff_modulus);
            }
        } else {
            size_t poly_coeff_count = coeff_count * coeff_modulus_size;
            size_t key_poly_coeff_count = coeff_count * key_coeff_modulus_size;
            Array<uint64_t> encrypted_copy = Array<uint64_t>::create_and_copy_from_slice(encrypted.data().const_slice(poly_coeff_count, encrypted_size * poly_coeff_count));
            if (!is_ntt_form) {
                utils::ntt_negacyclic_harvey_ps(encrypted_copy.reference(), encrypted_size - 1, coeff_count, ntt_tables);
            }
            for (size_t i = 0; i < encrypted_size - 1; i++) {
                utils::dyadic_product_inplace_p(
                    encrypted_copy.slice(i*poly_coeff_count, (i+1)*poly_coeff_count), 
                    this->secret_key_array_.const_slice(i*key_poly_coeff_count, i*key_poly_coeff_count+poly_coeff_count), 
                    coeff_count, coeff_modulus);
            }
            destination.set_zero();
            for (size_t i = 0; i < encrypted_size - 1; i++) {
                utils::add_inplace_p(
                    destination, 
                    encrypted_copy.const_slice(i*poly_coeff_count, (i+1)*poly_coeff_count),
                    coeff_count, coeff_modulus);
            }
            if (!is_ntt_form) {
                utils::inverse_ntt_negacyclic_harvey_p(destination, coeff_count, ntt_tables);
            }
            utils::add_inplace_p(destination, encrypted.poly(0), coeff_count, coeff_modulus);
        }

        // release read lock
        lock.unlock();
    }

    void Decryptor::decrypt(const Ciphertext& encrypted, Plaintext& destination) const {
        // sanity check
        if (encrypted.contains_seed()) {
            throw std::invalid_argument("[Decryptor::decrypt] Seed should be expanded first.");
        }
        if (encrypted.polynomial_count() < utils::HE_CIPHERTEXT_SIZE_MIN) {
            throw std::invalid_argument("[Decryptor::decrypt] Ciphertext is empty.");
        }
        if (encrypted.on_device()) destination.to_device_inplace();
        else destination.to_host_inplace();
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: this->bfv_decrypt(encrypted, destination); break;
            case SchemeType::CKKS: this->ckks_decrypt(encrypted, destination); break;
            case SchemeType::BGV: this->bgv_decrypt(encrypted, destination); break;
            default: throw std::invalid_argument("[Decryptor::decrypt] Unsupported scheme.");
        }
    }

    void Decryptor::bfv_decrypt(const Ciphertext& encrypted, Plaintext& destination) const {
        if (encrypted.is_ntt_form()) {
            throw std::invalid_argument("[Decryptor::bfv_decrypt] Ciphertext is in NTT form.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        
        // Firstly find c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
        // This is equal to Delta m + v where ||v|| < Delta/2.
        // Add Delta / 2 and now we have something which is Delta * (m + epsilon) where epsilon < 1
        // Therefore, we can (integer) divide by Delta and the answer will round down to m.

        // Make a temp destination for all the arithmetic mod qi before calling FastBConverse
        bool device = encrypted.on_device();
        if (device) destination.to_device_inplace();
        else destination.to_host_inplace();
        Array<uint64_t> tmp_dest_modq(coeff_count * coeff_modulus_size, device);

        // put < (c_1 , c_2, ... , c_{count-1}) , (s,s^2,...,s^{count-1}) > mod q in destination
        // Now do the dot product of encrypted_copy and the secret key array using NTT.
        // The secret key powers are already NTT transformed.
        this->dot_product_ct_sk_array(encrypted, tmp_dest_modq.reference());

        // Allocate a full size destination to write to
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count);

        // Divide scaling variant using BEHZ FullRNS techniques
        context_data->rns_tool().decrypt_scale_and_round(
            tmp_dest_modq.const_reference(), destination.poly()
        );
    }

    void Decryptor::ckks_decrypt(const Ciphertext& encrypted, Plaintext& destination) const {

        if (!encrypted.is_ntt_form()) {
            throw std::invalid_argument("[Decryptor::ckks_decrypt] Ciphertext is not in NTT form.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t rns_poly_uint64_count = coeff_count * coeff_modulus_size;
        
        bool device = encrypted.on_device();
        if (device) destination.to_device_inplace();
        else destination.to_host_inplace();

        // Decryption consists in finding
        // c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q_1 * q_2 * q_3
        // as long as ||m + v|| < q_1 * q_2 * q_3.
        // This is equal to m + v where ||v|| is small enough.
        
        // Since we overwrite destination, we zeroize destination parameters
        // This is necessary, otherwise resize will throw an exception.
        destination.parms_id() = parms_id_zero;
        // Resize destination to appropriate size
        destination.resize(rns_poly_uint64_count);
        
        // Do the dot product of encrypted and the secret key array using NTT.
        this->dot_product_ct_sk_array(encrypted, destination.poly());

        // Set destination parameters as in encrypted
        destination.parms_id() = encrypted.parms_id();
        destination.scale() = encrypted.scale();
        destination.is_ntt_form() = true;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
    }

    void Decryptor::bgv_decrypt(const Ciphertext& encrypted, Plaintext& destination) const {
        if (!encrypted.is_ntt_form()) {
            throw std::invalid_argument("[Decryptor::bgv_decrypt] Ciphertext is not in NTT form.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        
        // Make a temp destination for all the arithmetic mod qi before calling FastBConverse
        bool device = encrypted.on_device();
        if (device) destination.to_device_inplace();
        else destination.to_host_inplace();
        Array<uint64_t> tmp_dest_modq(coeff_count * coeff_modulus_size, device);

        this->dot_product_ct_sk_array(encrypted, tmp_dest_modq.reference());

        // Allocate a full size destination to write to
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count);

        utils::inverse_ntt_negacyclic_harvey_p(tmp_dest_modq.reference(), coeff_count, context_data->small_ntt_tables());

        // Divide scaling variant using BEHZ FullRNS techniques
        context_data->rns_tool().decrypt_mod_t(
            tmp_dest_modq.const_reference(), destination.poly()
        );
        
        if (encrypted.correction_factor() != 1) {
            uint64_t fix = 1;
            if (!utils::try_invert_uint64_mod(encrypted.correction_factor(), parms.plain_modulus_host(), fix)) {
                throw std::logic_error("[Decryptor::bgv_decrypt] Correction factor is not invertible.");
            }
            utils::multiply_scalar_inplace(destination.poly(), fix, parms.plain_modulus());
        }

        destination.is_ntt_form() = false;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
    }

}