#include "batch_utils.h"
#include "evaluator.h"
#include "evaluator_utils.h"
#include "utils/polynomial_buffer.h"
#include "fgk/dyadic_convolute.h"

namespace troy {

    using utils::ConstSlice;
    using utils::NTTTables;
    using utils::Buffer;

    void Evaluator::multiply_plain_normal(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::multiply_plain_normal]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::multiply_plain_normal]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        size_t encrypted_size = encrypted.polynomial_count();
        bool device = encrypted.on_device();
        Buffer<uint64_t> temp(coeff_modulus_size, coeff_count, device, pool);

        bool skip_ntt_temp = false;

        if (plain.parms_id() == parms_id_zero) {

            // Note: the original implementation has an optimization
            // for plaintexts with only one term.
            // But we are reluctant to detect the number of non-zero terms
            // in the plaintext, so we just use the general implementation.
            
            // Generic case: any plaintext polynomial
            // Allocate temporary space for an entire RNS polynomial
            scaling_variant::centralize(plain, context_data, temp.reference(), coeff_count, pool);
        
        } else {

            // The plaintext is already conveyed to modulus Q.
            // Directly copy.
            if (plain.coeff_count() == coeff_count) {
                utils::ntt_p(plain.reference(), coeff_count, ntt_tables, temp.reference());
                skip_ntt_temp = true;
            } else {
                utils::scatter_partial_p(plain.const_poly(), plain.coeff_count(), coeff_count, coeff_modulus_size, temp.reference());
            }
            
        }

        destination = Ciphertext::like(encrypted, false, pool);

        // Need to multiply each component in encrypted with temp; first step is to transform to NTT form
        // RNSIter temp_iter(temp.get(), coeff_count);
        if (!skip_ntt_temp) utils::ntt_inplace_p(temp.reference(), coeff_count, ntt_tables);
        utils::ntt_ps(encrypted.const_reference(), encrypted_size, coeff_count, ntt_tables, destination.reference());
        utils::fgk::dyadic_convolute::dyadic_broadcast_product_inplace_ps(destination.reference(), temp.const_reference(), encrypted_size, coeff_count, coeff_modulus);
        utils::intt_inplace_ps(destination.reference(), encrypted_size, coeff_count, ntt_tables);

        if (parms.scheme() == SchemeType::CKKS) {
            destination.scale() = encrypted.scale() * plain.scale();
            if (!is_scale_within_bounds(destination.scale(), context_data)) {
                throw std::invalid_argument("[Evaluator::multiply_plain_normal] Scale out of bounds.");
            }
        }
    }
    
    void Evaluator::multiply_plain_normal_batched(
        const std::vector<const Ciphertext*>& encrypted, const std::vector<const Plaintext*>& plain, 
        const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool
    ) const {
        if (encrypted.size() != plain.size() || encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::multiply_plain_normal_batched] Input vectors have different sizes.");
        }
        if (encrypted.empty()) return;
        check_no_seed_vec("[Evaluator::multiply_plain_normal_batched]", encrypted);
        
        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::multiply_plain_normal_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        size_t encrypted_size = get_vec_polynomial_count(encrypted);
        bool device = encrypted[0]->on_device();
        std::vector<Buffer<uint64_t>> temp(encrypted.size());
        for (size_t i = 0; i < encrypted.size(); i++) temp[i] = Buffer<uint64_t>(coeff_modulus_size, coeff_count, device, pool);
        auto temp_batched = batch_utils::rcollect_reference(temp);
        auto temp_const_batched = batch_utils::rcollect_const_reference(temp);

        bool skip_ntt_temp = false;

        auto plain_parms_id = get_vec_parms_id(plain);
        if (plain_parms_id == parms_id_zero) {

            // Note: the original implementation has an optimization
            // for plaintexts with only one term.
            // But we are reluctant to detect the number of non-zero terms
            // in the plaintext, so we just use the general implementation.
            
            // Generic case: any plaintext polynomial
            // Allocate temporary space for an entire RNS polynomial
            scaling_variant::centralize_batched(plain, context_data, temp_batched, coeff_count, pool);
        
        } else {

            // The plaintext is already conveyed to modulus Q.
            // Directly copy.
            auto plain_coeff_count = get_vec_coeff_count(plain);
            auto plain_batched = batch_utils::pcollect_const_reference(plain);
            if (plain_coeff_count == coeff_count) {
                utils::ntt_bp(plain_batched, coeff_count, ntt_tables, temp_batched, pool);
                skip_ntt_temp = true;
            } else {
                utils::scatter_partial_bp(plain_batched, plain_coeff_count, coeff_count, coeff_modulus_size, temp_batched);
            }
            
        }

        for (size_t i = 0; i < encrypted.size(); i++) {
            *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        }

        // Need to multiply each component in encrypted with temp; first step is to transform to NTT form
        // RNSIter temp_iter(temp.get(), coeff_count);
        
        if (!skip_ntt_temp) {
            utils::ntt_inplace_bp(temp_batched, coeff_count, ntt_tables, pool);
        }

        auto encrypted_batched = batch_utils::pcollect_const_reference(encrypted);
        auto destination_batched = batch_utils::pcollect_reference(destination);
        utils::ntt_bps(encrypted_batched, encrypted_size, coeff_count, ntt_tables, destination_batched, pool);

        utils::fgk::dyadic_convolute::dyadic_broadcast_product_inplace_bps(destination_batched, temp_const_batched, encrypted_size, coeff_count, coeff_modulus, pool);

        utils::intt_inplace_bps(destination_batched, encrypted_size, coeff_count, ntt_tables, pool);

        if (parms.scheme() == SchemeType::CKKS) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                destination[i]->scale() = encrypted[i]->scale() * plain[i]->scale();
                if (!is_scale_within_bounds(destination[i]->scale(), context_data)) {
                    throw std::invalid_argument("[Evaluator::multiply_plain_normal_batched] Scale out of bounds.");
                }
            }
        }
    }



    void Evaluator::multiply_plain_normal_accumulate(
        const std::vector<const Ciphertext*>& encrypted, const std::vector<const Plaintext*>& plain, 
        const std::vector<Ciphertext*>& destination, bool set_zero, MemoryPoolHandle pool
    ) const {
        if (encrypted.size() != plain.size() || encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::multiply_plain_normal_batched] Input vectors have different sizes.");
        }
        if (encrypted.empty()) return;
        check_no_seed_vec("[Evaluator::multiply_plain_normal_batched]", encrypted);
        
        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::multiply_plain_normal_batched]", parms_id);


        if (set_zero) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                *(destination[i]) = Ciphertext::like(*encrypted[i], false, pool);
            }
            {
                auto destination_batched = batch_utils::pcollect_reference(destination);
                utils::set_slice_b(0, destination_batched, pool);
            }
        } else {
            for (size_t i = 0; i < encrypted.size(); i++) {
                if (destination[i]->parms_id() != encrypted[i]->parms_id() || destination[i]->is_ntt_form() != encrypted[i]->is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::multiply_plain_normal_accumulate] Destination parameters do not match.");
                }
            }
        }

        std::vector<Ciphertext> temp(encrypted.size());
        this->multiply_plain_batched(encrypted, plain, batch_utils::collect_pointer(temp), pool);
        
        for (size_t i = 0; i < encrypted.size(); i++) {
            destination[i]->scale() = temp[i].scale();
            this->add_inplace(*destination[i], temp[i]);
        }
    }


    void Evaluator::multiply_plain_ntt(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::multiply_plain_ntt]", encrypted);
        if (encrypted.parms_id() != plain.parms_id()) {
            throw std::invalid_argument("[Evaluator::multiply_plain_ntt] Plaintext and ciphertext parameters do not match.");
        }

        ContextDataPointer context_data = this->get_context_data("[Evaluator::multiply_plain_ntt]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t encrypted_size = encrypted.polynomial_count();

        destination = Ciphertext::like(encrypted, false, pool);

        utils::fgk::dyadic_convolute::dyadic_broadcast_product_ps(encrypted.const_reference(), plain.poly(), encrypted_size, coeff_count, coeff_modulus, destination.reference());

        if (parms.scheme() == SchemeType::CKKS) {
            destination.scale() = encrypted.scale() * plain.scale();
            if (!is_scale_within_bounds(destination.scale(), context_data)) {
                throw std::invalid_argument("[Evaluator::multiply_plain_normal_inplace] Scale out of bounds.");
            }
        }
    }

    void Evaluator::multiply_plain_ntt_batched(
        const std::vector<const Ciphertext*>& encrypted, const std::vector<const Plaintext*>& plain, 
        const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool
    ) const {
        if (encrypted.size() != plain.size() || encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::multiply_plain_ntt_batched] Input vectors have different sizes.");
        }
        if (encrypted.empty()) return;
        check_no_seed_vec("[Evaluator::multiply_plain_ntt_batched]", encrypted);

        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::multiply_plain_ntt]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t encrypted_size = get_vec_polynomial_count(encrypted);

        for (Ciphertext* c: destination) 
            *c = Ciphertext::like(*encrypted[0], false, pool);
    
        auto encrypted_batch = batch_utils::pcollect_const_reference(encrypted);
        auto plain_batch = batch_utils::pcollect_const_poly(plain);
        auto destination_batch = batch_utils::pcollect_reference(destination);
        utils::fgk::dyadic_convolute::dyadic_broadcast_product_bps(
            encrypted_batch, plain_batch, encrypted_size, coeff_count, coeff_modulus, destination_batch
        );

        if (parms.scheme() == SchemeType::CKKS) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                destination[i]->scale() = encrypted[i]->scale() * plain[i]->scale();
                if (!is_scale_within_bounds(destination[i]->scale(), context_data)) {
                    throw std::invalid_argument("[Evaluator::multiply_plain_ntt_batched] Scale out of bounds.");
                }
            }
        }

    }
    
    void Evaluator::multiply_plain_ntt_accumulate(
        const std::vector<const Ciphertext*>& encrypted, const std::vector<const Plaintext*>& plain, 
        const std::vector<Ciphertext*>& destination, bool set_zero, MemoryPoolHandle pool
    ) const {
        if (encrypted.size() != plain.size() || encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::multiply_plain_ntt_batched] Input vectors have different sizes.");
        }
        if (encrypted.empty()) return;
        check_no_seed_vec("[Evaluator::multiply_plain_ntt_batched]", encrypted);

        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::multiply_plain_ntt]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t encrypted_size = get_vec_polynomial_count(encrypted);

        if (set_zero) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
            }
            {
                auto destination_batched = batch_utils::pcollect_reference(destination);
                utils::set_slice_b(0, destination_batched, pool);
            }
        } else {
            for (size_t i = 0; i < encrypted.size(); i++) {
                if (destination[i]->parms_id() != encrypted[i]->parms_id() || destination[i]->is_ntt_form() != encrypted[i]->is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::multiply_plain_normal_accumulate] Destination parameters do not match.");
                }
            }
        }
    
        auto encrypted_batch = batch_utils::pcollect_const_reference(encrypted);
        auto plain_batch = batch_utils::pcollect_const_poly(plain);
        auto destination_batch = batch_utils::pcollect_reference(destination);
        utils::fgk::dyadic_convolute::dyadic_broadcast_product_accumulate_bps(
            encrypted_batch, plain_batch, encrypted_size, coeff_count, coeff_modulus, destination_batch
        );

        if (parms.scheme() == SchemeType::CKKS) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                destination[i]->scale() = encrypted[i]->scale() * plain[i]->scale();
                if (!is_scale_within_bounds(destination[i]->scale(), context_data)) {
                    throw std::invalid_argument("[Evaluator::multiply_plain_ntt_batched] Scale out of bounds.");
                }
            }
        }

    }

    void Evaluator::multiply_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandle pool) const {
        bool encrypted_ntt = encrypted.is_ntt_form();
        bool plain_ntt = plain.is_ntt_form();
        if (encrypted_ntt && plain_ntt) {
            this->multiply_plain_ntt(encrypted, plain, destination, pool);
        } else if (!encrypted_ntt && !plain_ntt) {
            this->multiply_plain_normal(encrypted, plain, destination, pool);
        } else if (encrypted_ntt && !plain_ntt) {
            Plaintext plain_ntt;
            this->transform_plain_to_ntt(plain, encrypted.parms_id(), plain_ntt, pool);
            this->multiply_plain_ntt(encrypted, plain_ntt, destination, pool);
        } else { // !encrypted_ntt && plain_ntt
            Ciphertext encrypted_ntt;
            this->transform_to_ntt(encrypted, encrypted_ntt, pool);
            this->multiply_plain_ntt(encrypted_ntt, plain, destination, pool);
            this->transform_from_ntt_inplace(destination);
        }
    }

    void Evaluator::multiply_plain_batched(
        const std::vector<const Ciphertext*>& encrypted, 
        const std::vector<const Plaintext*>& plain, 
        const std::vector<Ciphertext*>& destination, 
        MemoryPoolHandle pool
    ) const {
        bool encrypted_ntt = get_is_ntt_form_vec(encrypted);
        bool plain_ntt = get_is_ntt_form_vec(plain);
        if (encrypted_ntt && plain_ntt) {
            this->multiply_plain_ntt_batched(encrypted, plain, destination, pool);
        } else if (!encrypted_ntt && !plain_ntt) {
            this->multiply_plain_normal_batched(encrypted, plain, destination, pool);
        } else if (encrypted_ntt && !plain_ntt) {
            std::vector<Plaintext> plain_ntt(plain.size());
            auto plain_ntt_ptr = batch_utils::collect_pointer(plain_ntt);
            auto plain_ntt_const_ptr = batch_utils::collect_const_pointer(plain_ntt);
            this->transform_plain_to_ntt_batched(plain, get_vec_parms_id(encrypted), plain_ntt_ptr, pool);
            this->multiply_plain_ntt_batched(encrypted, plain_ntt_const_ptr, destination, pool);
        } else { // !encrypted_ntt && plain_ntt
            std::vector<Ciphertext> encrypted_ntt(encrypted.size());
            auto encrypted_ntt_ptr = batch_utils::collect_pointer(encrypted_ntt);
            auto encrypted_ntt_const_ptr = batch_utils::collect_const_pointer(encrypted_ntt);
            this->transform_to_ntt_batched(encrypted, encrypted_ntt_ptr, pool);
            this->multiply_plain_ntt_batched(encrypted_ntt_const_ptr, plain, destination, pool);
            this->transform_from_ntt_inplace_batched(destination, pool);
        }
    }

    void Evaluator::multiply_plain_accumulate(
        const std::vector<const Ciphertext*>& encrypted, 
        const std::vector<const Plaintext*>& plain, 
        const std::vector<Ciphertext*>& destination, 
        bool set_zero,
        MemoryPoolHandle pool
    ) const {
        bool encrypted_ntt = get_is_ntt_form_vec(encrypted);
        bool plain_ntt = get_is_ntt_form_vec(plain);
        if (encrypted_ntt && plain_ntt) {
            this->multiply_plain_ntt_accumulate(encrypted, plain, destination, set_zero, pool);
        } else if (!encrypted_ntt && !plain_ntt) {
            this->multiply_plain_normal_accumulate(encrypted, plain, destination, set_zero, pool);
        } else if (encrypted_ntt && !plain_ntt) {
            std::vector<Plaintext> plain_ntt(plain.size());
            auto plain_ntt_ptr = batch_utils::collect_pointer(plain_ntt);
            auto plain_ntt_const_ptr = batch_utils::collect_const_pointer(plain_ntt);
            this->transform_plain_to_ntt_batched(plain, get_vec_parms_id(encrypted), plain_ntt_ptr, pool);
            this->multiply_plain_ntt_accumulate(encrypted, plain_ntt_const_ptr, destination, set_zero, pool);
        } else { // !encrypted_ntt && plain_ntt
            std::vector<Ciphertext> encrypted_ntt(encrypted.size());
            auto encrypted_ntt_ptr = batch_utils::collect_pointer(encrypted_ntt);
            auto encrypted_ntt_const_ptr = batch_utils::collect_const_pointer(encrypted_ntt);
            this->transform_to_ntt_batched(encrypted, encrypted_ntt_ptr, pool);
            this->multiply_plain_ntt_accumulate(encrypted_ntt_const_ptr, plain, destination, set_zero, pool);
            this->transform_from_ntt_inplace_batched(destination, pool);
        }
    }


}