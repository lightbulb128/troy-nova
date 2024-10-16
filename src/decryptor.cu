#include "batch_utils.h"
#include "decryptor.h"
#include "encryption_parameters.h"
#include "plaintext.h"
#include "utils/constants.h"
#include "utils/scaling_variant.h"

namespace troy {

    using utils::ConstSlice;
    using utils::NTTTables;
    using utils::Array;

    Decryptor::Decryptor(HeContextPointer context, const SecretKey& secret_key, MemoryPoolHandle pool) :
        context_(context) 
    {
        ContextDataPointer key_context_data = context->key_context_data().value();
        const EncryptionParameters& parms = key_context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        if (secret_key.data().size() != coeff_count * coeff_modulus_size)
            throw std::invalid_argument("[Decryptor::Decryptor] secret_key is not valid for encryption parameters");
        this->secret_key_array_ = secret_key.data().clone(pool);
    }

    void Decryptor::dot_product_ct_sk_array(const Ciphertext& encrypted, utils::Slice<uint64_t> destination, MemoryPoolHandle pool) const {
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
                utils::ntt_inplace_p(destination, coeff_count, ntt_tables);
                utils::dyadic_product_inplace_p(destination, s, coeff_count, coeff_modulus);
                utils::intt_inplace_p(destination, coeff_count, ntt_tables);
                utils::add_inplace_p(destination, c0, coeff_count, coeff_modulus);
            }
        } else {
            size_t poly_coeff_count = coeff_count * coeff_modulus_size;
            size_t key_poly_coeff_count = coeff_count * key_coeff_modulus_size;
            Array<uint64_t> encrypted_copy = Array<uint64_t>::create_uninitialized((encrypted_size - 1) * poly_coeff_count, encrypted.on_device(), pool);
            bool use_encrypted_copy = false;
            if (!is_ntt_form) {
                utils::ntt_ps(encrypted.data().const_slice(poly_coeff_count, encrypted_size * poly_coeff_count), encrypted_size - 1, coeff_count, ntt_tables, encrypted_copy.reference());
                use_encrypted_copy = true;
            }
            for (size_t i = 0; i < encrypted_size - 1; i++) {
                if (use_encrypted_copy) {
                    utils::dyadic_product_inplace_p(
                        encrypted_copy.slice(i*poly_coeff_count, (i+1)*poly_coeff_count), 
                        this->secret_key_array_.const_slice(i*key_poly_coeff_count, i*key_poly_coeff_count+poly_coeff_count), 
                        coeff_count, coeff_modulus);
                } else {
                    utils::dyadic_product_p(
                        encrypted.data().const_slice((i+1)*poly_coeff_count, (i+2)*poly_coeff_count), 
                        this->secret_key_array_.const_slice(i*key_poly_coeff_count, i*key_poly_coeff_count+poly_coeff_count), 
                        coeff_count, coeff_modulus, 
                        encrypted_copy.slice(i*poly_coeff_count, (i+1)*poly_coeff_count));
                }
            }
            destination.set_zero();
            for (size_t i = 0; i < encrypted_size - 1; i++) {
                utils::add_inplace_p(
                    destination, 
                    encrypted_copy.const_slice(i*poly_coeff_count, (i+1)*poly_coeff_count),
                    coeff_count, coeff_modulus);
            }
            if (!is_ntt_form) {
                utils::intt_inplace_p(destination, coeff_count, ntt_tables);
            }
            utils::add_inplace_p(destination, encrypted.poly(0), coeff_count, coeff_modulus);
        }

        // release read lock
        lock.unlock();
    }

    
    void Decryptor::dot_product_ct_sk_array_batched(const std::vector<const Ciphertext*>& encrypted, const utils::SliceVec<uint64_t>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Decryptor::dot_product_ct_sk_array_batched] encrypted and destination are not the same size.");
        }
        if (encrypted.size() == 0) return;
        if (!this->on_device() || encrypted.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                this->dot_product_ct_sk_array(*encrypted[i], destination[i], pool);
            }
            return;
        }

        ParmsID parms_id = encrypted[0]->parms_id();
        size_t encrypted_size = encrypted[0]->polynomial_count();
        bool is_ntt_form = encrypted[0]->is_ntt_form();
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (encrypted[i]->parms_id() != parms_id) {
                throw std::invalid_argument("[Decryptor::dot_product_ct_sk_array_batched] Ciphertexts are not in the same context.");
            }
            if (encrypted[i]->polynomial_count() != encrypted_size) {
                throw std::invalid_argument("[Decryptor::dot_product_ct_sk_array_batched] Ciphertexts are not in the same size.");
            }
            if (encrypted[i]->is_ntt_form() != is_ntt_form) {
                throw std::invalid_argument("[Decryptor::dot_product_ct_sk_array_batched] Ciphertexts are not in the same form.");
            }
        }
        ContextDataPointer context_data = this->context()->get_context_data(parms_id).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t key_coeff_modulus_size = this->context()->key_context_data().value()->parms().coeff_modulus().size();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        // Make sure we have enough secret key powers computed
        if (this->secret_key_array_.size() < (encrypted_size - 1) * coeff_count * key_coeff_modulus_size) {
            std::unique_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
            KeyGenerator::compute_secret_key_powers(this->context(), encrypted_size - 1, this->secret_key_array_);
            lock.unlock();
        }

        // acquire read lock
        std::shared_lock<std::shared_mutex> lock(this->secret_key_array_mutex);
        if (encrypted_size == 2) {
            auto c0 = batch_utils::pcollect_const_poly(encrypted, 0);
            auto c1 = batch_utils::pcollect_const_poly(encrypted, 1);
            ConstSlice<uint64_t> s = this->secret_key_array_.const_slice(0, c0.size());
            auto s_repeated = std::vector<ConstSlice<uint64_t>>(encrypted.size(), s);
            if (is_ntt_form) {
                // put < c_1 * s > mod q in destination
                utils::dyadic_product_bp(c1, s_repeated, coeff_count, coeff_modulus, destination, pool);
                // add c_0 to the result; note that destination should be in the same (NTT) form as encrypted
                utils::add_inplace_bp(destination, c0, coeff_count, coeff_modulus, pool);
            } else {
                utils::copy_slice_b(c1, destination, pool);
                utils::ntt_inplace_bp(destination, coeff_count, ntt_tables, pool);
                utils::dyadic_product_inplace_bp(destination, s_repeated, coeff_count, coeff_modulus, pool);
                utils::intt_inplace_bp(destination, coeff_count, ntt_tables, pool);
                utils::add_inplace_bp(destination, c0, coeff_count, coeff_modulus, pool);
            }
        } else {
            size_t poly_coeff_count = coeff_count * coeff_modulus_size;
            size_t key_poly_coeff_count = coeff_count * key_coeff_modulus_size;
            size_t n = encrypted.size();
            std::vector<Array<uint64_t>> encrypted_copy; encrypted_copy.reserve(n);
            for (size_t i = 0; i < n; i++) {
                encrypted_copy.push_back(Array<uint64_t>::create_uninitialized((encrypted_size - 1) * poly_coeff_count, true, pool));
            }
            bool use_encrypted_copy = false;
            if (!is_ntt_form) {
                utils::ConstSliceVec<uint64_t> encrypted_from; encrypted_from.reserve(n);
                for (size_t i = 0; i < n; i++) {
                    encrypted_from.push_back(encrypted[i]->data().const_slice(poly_coeff_count, encrypted_size * poly_coeff_count));
                }
                utils::ntt_bps(encrypted_from, encrypted_size - 1, coeff_count, ntt_tables, batch_utils::rcollect_reference(encrypted_copy), pool);
                use_encrypted_copy = true;
            }
            for (size_t i = 0; i < encrypted_size - 1; i++) {
                utils::SliceVec<uint64_t> target; target.reserve(n);
                for (size_t j = 0; j < n; j++) {
                    target.push_back(encrypted_copy[j].slice(i*poly_coeff_count, (i+1)*poly_coeff_count));
                }
                if (use_encrypted_copy) {
                    utils::dyadic_product_inplace_bp(
                        target, 
                        utils::ConstSliceVec<uint64_t>(n, this->secret_key_array_.const_slice(i*key_poly_coeff_count, i*key_poly_coeff_count+poly_coeff_count)), 
                        coeff_count, coeff_modulus, pool);
                } else {
                    utils::ConstSliceVec<uint64_t> encrypted_from; encrypted_from.reserve(n);
                    for (size_t j = 0; j < n; j++) {
                        encrypted_from.push_back(encrypted[j]->data().const_slice((i+1)*poly_coeff_count, (i+2)*poly_coeff_count));
                    }
                    utils::dyadic_product_bp(
                        encrypted_from, 
                        utils::ConstSliceVec<uint64_t>(n, this->secret_key_array_.const_slice(i*key_poly_coeff_count, i*key_poly_coeff_count+poly_coeff_count)),  
                        coeff_count, coeff_modulus, 
                        target, pool);
                }
            }
            utils::set_slice_b(0, destination, pool);
            for (size_t i = 0; i < encrypted_size - 1; i++) {
                utils::ConstSliceVec<uint64_t> target; target.reserve(n);
                for (size_t j = 0; j < n; j++) {
                    target.push_back(encrypted_copy[j].const_slice(i*poly_coeff_count, (i+1)*poly_coeff_count));
                }
                utils::add_inplace_bp(
                    destination, 
                    target,
                    coeff_count, coeff_modulus, pool);
            }
            if (!is_ntt_form) {
                utils::intt_inplace_bp(destination, coeff_count, ntt_tables, pool);
            }
            utils::add_inplace_bp(destination, batch_utils::pcollect_const_poly(encrypted, 0), coeff_count, coeff_modulus, pool);
        }

        // release read lock
        lock.unlock();
    }

    void Decryptor::decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const {
        // sanity check
        if (encrypted.contains_seed()) {
            throw std::invalid_argument("[Decryptor::decrypt] Seed should be expanded first.");
        }
        if (encrypted.polynomial_count() < utils::HE_CIPHERTEXT_SIZE_MIN) {
            throw std::invalid_argument("[Decryptor::decrypt] Ciphertext is empty.");
        }
        if (encrypted.on_device()) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: this->bfv_decrypt(encrypted, destination, pool); break;
            case SchemeType::CKKS: this->ckks_decrypt(encrypted, destination, pool); break;
            case SchemeType::BGV: this->bgv_decrypt(encrypted, destination, pool); break;
            default: throw std::invalid_argument("[Decryptor::decrypt] Unsupported scheme.");
        }
    }

    void Decryptor::decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        // sanity check
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Decryptor::decrypt_batched] encrypted and destination are not the same size.");
        }
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (encrypted[i]->contains_seed()) {
                throw std::invalid_argument("[Decryptor::decrypt] Seed should be expanded first.");
            }
            if (encrypted[i]->polynomial_count() < utils::HE_CIPHERTEXT_SIZE_MIN) {
                throw std::invalid_argument("[Decryptor::decrypt] Ciphertext is empty.");
            }
        }
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: this->bfv_decrypt_batched(encrypted, destination, pool); break;
            case SchemeType::CKKS: this->ckks_decrypt_batched(encrypted, destination, pool); break;
            case SchemeType::BGV: this->bgv_decrypt_batched(encrypted, destination, pool); break;
            default: throw std::invalid_argument("[Decryptor::decrypt] Unsupported scheme.");
        }
    }

    void Decryptor::bfv_decrypt_without_scaling_down(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const {
        if (encrypted.is_ntt_form()) {
            throw std::invalid_argument("[Decryptor::bfv_decrypt] Ciphertext is in NTT form.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        
        // Firstly find c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
        // This is equal to Delta m + v where ||v|| < Delta/2.

        // Make a temp destination for all the arithmetic mod qi before calling FastBConverse
        bool device = encrypted.on_device();
        destination = Plaintext();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        
        destination.resize_rns(*this->context_, encrypted.parms_id(), false, false);

        // put < (c_1 , c_2, ... , c_{count-1}) , (s,s^2,...,s^{count-1}) > mod q in destination
        // Now do the dot product of encrypted_copy and the secret key array using NTT.
        // The secret key powers are already NTT transformed.
        this->dot_product_ct_sk_array(encrypted, destination.reference(), pool);
    }

    void Decryptor::bfv_decrypt_without_scaling_down_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Decryptor::bfv_decrypt_without_scaling_down_batched] encrypted and destination are not the same size.");
        }
        if (!this->on_device() || encrypted.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                this->bfv_decrypt_without_scaling_down(*encrypted[i], *destination[i], pool);
            }
            return;
        }
        // all should be non_ntt_form
        ParmsID parms_id = encrypted[0]->parms_id();
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (encrypted[i]->is_ntt_form()) {
                throw std::invalid_argument("[Decryptor::bfv_decrypt_without_scaling_down_batched] Ciphertext is in NTT form.");
            }
            if (encrypted[i]->parms_id() != parms_id) {
                throw std::invalid_argument("[Decryptor::bfv_decrypt_without_scaling_down_batched] Ciphertexts are not in the same context.");
            }
        }
        ContextDataPointer context_data = this->context()->get_context_data(parms_id).value();
        
        // Firstly find c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
        // This is equal to Delta m + v where ||v|| < Delta/2.

        // Make a temp destination for all the arithmetic mod qi before calling FastBConverse
        bool device = this->on_device();
        for (size_t i = 0; i < destination.size(); i++) {
            *destination[i] = Plaintext();
            if (device) destination[i]->to_device_inplace(pool);
            else destination[i]->to_host_inplace();
            destination[i]->resize_rns(*this->context_, parms_id, false, false);
        }
        
        // put < (c_1 , c_2, ... , c_{count-1}) , (s,s^2,...,s^{count-1}) > mod q in destination
        // Now do the dot product of encrypted_copy and the secret key array using NTT.
        // The secret key powers are already NTT transformed.
        this->dot_product_ct_sk_array_batched(encrypted, batch_utils::pcollect_reference(destination), pool);
    }

    void Decryptor::bfv_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const {
        if (encrypted.is_ntt_form()) {
            throw std::invalid_argument("[Decryptor::bfv_decrypt] Ciphertext is in NTT form.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        Plaintext temp;
        this->bfv_decrypt_without_scaling_down(encrypted, temp, pool);
        
        // Add Delta / 2 and now we have something which is Delta * (m + epsilon) where epsilon < 1
        // Therefore, we can (integer) divide by Delta and the answer will round down to m.

        bool device = encrypted.on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();

        // Allocate a full size destination to write to
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count);

        // Divide scaling variant using BEHZ FullRNS techniques
        context_data->rns_tool().decrypt_scale_and_round(
            temp.const_reference(), coeff_count, destination.poly(), pool
        );
        destination.is_ntt_form() = false;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
    }

    void Decryptor::bfv_decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Decryptor::bfv_decrypt_batched] encrypted and destination are not the same size.");
        }
        if (!this->on_device() || encrypted.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                this->bfv_decrypt(*encrypted[i], *destination[i], pool);
            }
            return;
        }
        // all must be non ntt form
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (encrypted[i]->is_ntt_form()) {
                throw std::invalid_argument("[Decryptor::bfv_decrypt_batched] Ciphertext is in NTT form.");
            }
        }
        ParmsID parms_id = encrypted[0]->parms_id();
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (encrypted[i]->parms_id() != parms_id) {
                throw std::invalid_argument("[Decryptor::bfv_decrypt_batched] Ciphertexts are not in the same context.");
            }
        }
        ContextDataPointer context_data = this->context()->get_context_data(parms_id).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        size_t n = encrypted.size();
        std::vector<Plaintext> temp(n);
        this->bfv_decrypt_without_scaling_down_batched(encrypted, batch_utils::collect_pointer(temp), pool);
        
        // Add Delta / 2 and now we have something which is Delta * (m + epsilon) where epsilon < 1
        // Therefore, we can (integer) divide by Delta and the answer will round down to m.

        bool device = true;
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Plaintext();
            if (device) destination[i]->to_device_inplace(pool);
            else destination[i]->to_host_inplace();

            // Allocate a full size destination to write to
            destination[i]->parms_id() = parms_id_zero;
            destination[i]->resize(coeff_count);
            destination[i]->is_ntt_form() = false;
            destination[i]->coeff_modulus_size() = coeff_modulus_size;
            destination[i]->poly_modulus_degree() = coeff_count;
        }

        // Divide scaling variant using BEHZ FullRNS techniques
        context_data->rns_tool().decrypt_scale_and_round_batched(
            batch_utils::rcollect_const_reference(temp), coeff_count, batch_utils::pcollect_reference(destination), pool
        );
    }

    void Decryptor::ckks_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const {

        if (!encrypted.is_ntt_form()) {
            throw std::invalid_argument("[Decryptor::ckks_decrypt] Ciphertext is not in NTT form.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        
        bool device = encrypted.on_device();
        destination = Plaintext();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();

        // Decryption consists in finding
        // c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q_1 * q_2 * q_3
        // as long as ||m + v|| < q_1 * q_2 * q_3.
        // This is equal to m + v where ||v|| is small enough.
        
        // Since we overwrite destination, we zeroize destination parameters
        // This is necessary, otherwise resize will throw an exception.
        destination.resize_rns(*context(), encrypted.parms_id(), false, false);
        
        // Do the dot product of encrypted and the secret key array using NTT.
        this->dot_product_ct_sk_array(encrypted, destination.poly(), pool);

        // Set destination parameters as in encrypted
        destination.parms_id() = encrypted.parms_id();
        destination.scale() = encrypted.scale();
        destination.is_ntt_form() = true;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
    }


    void Decryptor::ckks_decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Decryptor::bfv_decrypt_batched] encrypted and destination are not the same size.");
        }
        if (!this->on_device() || encrypted.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                this->ckks_decrypt(*encrypted[i], *destination[i], pool);
            }
            return;
        }
        // all must be ntt form
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (!encrypted[i]->is_ntt_form()) {
                throw std::invalid_argument("[Decryptor::bfv_decrypt_batched] Ciphertext is in NTT form.");
            }
        }
        ParmsID parms_id = encrypted[0]->parms_id();
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (encrypted[i]->parms_id() != parms_id) {
                throw std::invalid_argument("[Decryptor::bfv_decrypt_batched] Ciphertexts are not in the same context.");
            }
        }
        ContextDataPointer context_data = this->context()->get_context_data(parms_id).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        size_t n = encrypted.size();
        
        // Add Delta / 2 and now we have something which is Delta * (m + epsilon) where epsilon < 1
        // Therefore, we can (integer) divide by Delta and the answer will round down to m.

        bool device = true;
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Plaintext();
            if (device) destination[i]->to_device_inplace(pool);
            else destination[i]->to_host_inplace();

            // Allocate a full size destination to write to
            destination[i]->parms_id() = parms_id_zero;
            destination[i]->resize_rns(*context(), parms_id, false, false);
            destination[i]->is_ntt_form() = true;
            destination[i]->coeff_modulus_size() = coeff_modulus_size;
            destination[i]->poly_modulus_degree() = coeff_count;
            destination[i]->scale() = encrypted[i]->scale();
        }

        // Divide scaling variant using BEHZ FullRNS techniques
        this->dot_product_ct_sk_array_batched(encrypted, batch_utils::pcollect_reference(destination), pool);
    }


    void Decryptor::bgv_decrypt(const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandle pool) const {
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
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        Plaintext tmp_dest_modq; if (encrypted.on_device()) tmp_dest_modq.to_device_inplace(pool);
        tmp_dest_modq.resize_rns(*this->context_, encrypted.parms_id());

        this->dot_product_ct_sk_array(encrypted, tmp_dest_modq.reference(), pool);

        // Allocate a full size destination to write to
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count);

        utils::intt_inplace_p(tmp_dest_modq.reference(), coeff_count, context_data->small_ntt_tables());

        scaling_variant::decentralize(tmp_dest_modq, context_data, destination.poly(), encrypted.correction_factor(), pool);

        destination.is_ntt_form() = false;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
    }


    void Decryptor::bgv_decrypt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Decryptor::bfv_decrypt_batched] encrypted and destination are not the same size.");
        }
        // TODO: implement batched decryption for BGV (decentralize required)
        for (size_t i = 0; i < encrypted.size(); i++) {
            this->bgv_decrypt(*encrypted[i], *destination[i], pool);
        }
    }

    static void poly_infty_norm(ConstSlice<uint64_t> poly, size_t coeff_uint64_count, ConstSlice<uint64_t> modulus, utils::Slice<uint64_t> result) {
        if (modulus.size() != coeff_uint64_count) {
            throw std::invalid_argument("[poly_infty_norm] Modulus is not valid.");
        }
        bool device = poly.on_device();
        if (device) {
            throw std::invalid_argument("[poly_infty_norm] Poly is on device.");
        }
        // Construct negative threshold: (modulus + 1) / 2
        Array<uint64_t> modulus_neg_threshold(modulus.size(), false, nullptr);
        utils::half_round_up_uint(modulus, modulus_neg_threshold.reference());
        // Mod out the poly coefficients and choose a symmetric representative from [-modulus,modulus)
        result.set_zero();
        Array<uint64_t> coeff_abs_value(coeff_uint64_count, false, nullptr);
        coeff_abs_value.set_zero();
        size_t coeff_count = poly.size() / coeff_uint64_count;
        for (size_t i = 0; i < coeff_count; i++) {
            ConstSlice<uint64_t> poly_i = poly.const_slice(i * coeff_uint64_count, (i + 1) * coeff_uint64_count);
            if (utils::is_greater_or_equal_uint(poly_i, modulus_neg_threshold.const_reference())) {
                utils::sub_uint(modulus, poly_i, coeff_abs_value.reference());
            } else {
                coeff_abs_value.copy_from_slice(poly_i);
            }
            if (utils::is_greater_than_uint(coeff_abs_value.const_reference(), result.as_const())) {
                result.copy_from_slice(coeff_abs_value.const_reference());
            }
        }
    }

    size_t Decryptor::invariant_noise_budget(const Ciphertext& encrypted, MemoryPoolHandle pool) const {
        if (encrypted.polynomial_count() < utils::HE_CIPHERTEXT_SIZE_MIN) {
            throw std::invalid_argument("[Decryptor::invariant_noise_budget] Ciphertext is invalid.");
        }
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        if (scheme != SchemeType::BFV && scheme != SchemeType::BGV) {
            throw std::invalid_argument("[Decryptor::invariant_noise_budget] Unsupported scheme.");
        }
        ContextDataPointer context_data = this->context()->get_context_data(encrypted.parms_id()).value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        const Modulus& plain_modulus = parms.plain_modulus_host();

        // Now need to compute c(s) - Delta*m (mod q)
        // Firstly find c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
        // This is equal to Delta m + v where ||v|| < Delta/2.
        // put < (c_1 , c_2, ... , c_{count-1}) , (s,s^2,...,s^{count-1}) > mod q
        // in destination_poly.
        // Now do the dot product of encrypted_copy and the secret key array using NTT.
        // The secret key powers are already NTT transformed.
        Array<uint64_t> noise_poly(coeff_count * coeff_modulus_size, encrypted.on_device(), pool);
        this->dot_product_ct_sk_array(encrypted, noise_poly.reference(), pool);

        if (encrypted.is_ntt_form()) {
            // In the case of NTT form, we need to transform the noise to normal form
            utils::intt_inplace_p(noise_poly.reference(), coeff_count, context_data->small_ntt_tables());
        }

        // Multiply by plain_modulus and reduce mod coeff_modulus to get
        // coeffModulus()*noise.
        if (scheme == SchemeType::BFV) {
            utils::multiply_scalar_inplace_p(
                noise_poly.reference(), plain_modulus.value(), coeff_count, coeff_modulus
            );
        }

        // CRT-compose the noise
        context_data->rns_tool().base_q().compose_array(noise_poly.reference(), pool);

        // Next we compute the infinity norm mod parms.coeffModulus()
        Array<uint64_t> norm(coeff_modulus_size, false, nullptr);
        noise_poly.to_host_inplace();
        Array<uint64_t> total_coeff_modulus = Array<uint64_t>::create_and_copy_from_slice(context_data->total_coeff_modulus(), false, nullptr);
        poly_infty_norm(noise_poly.const_reference(), coeff_modulus_size, total_coeff_modulus.const_reference(), norm.reference());

        // The -1 accounts for scaling the invariant noise by 2;
        // note that we already took plain_modulus into account in compose
        // so no need to subtract log(plain_modulus) from this
        int64_t bit_count_diff = 
            static_cast<int64_t>(context_data->total_coeff_modulus_bit_count()) 
            - static_cast<int64_t>(utils::get_significant_bit_count_uint(norm.const_reference())) 
            - 1;
        if (bit_count_diff < 0) {
            return 0;
        } else {
            return static_cast<size_t>(bit_count_diff);
        }
    }

}