#include "encryption_parameters.h"
#include "evaluator.h"
#include "evaluator_utils.h"
#include "batch_utils.h"
#include "utils/constants.h"


namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::RNSTool;

    void Evaluator::mod_switch_scale_to_next_internal(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        ParmsID parms_id = encrypted.parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_scale_to_next_internal]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                check_is_not_ntt_form("[Evaluator::mod_switch_scale_to_next_internal]", encrypted);
                break;
            }
            case SchemeType::CKKS: case SchemeType::BGV: {
                check_is_ntt_form("[Evaluator::mod_switch_scale_to_next_internal]", encrypted);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::mod_switch_scale_to_next_internal] Scheme not implemented.");
            }
        }
        if (!context_data->next_context_data().has_value()) {
            throw std::invalid_argument("[Evaluator::mod_switch_scale_to_next_internal] Next context data is not set.");
        }
        ContextDataPointer next_context_data = context_data->next_context_data().value();
        const EncryptionParameters& next_parms = next_context_data->parms();
        const RNSTool& rns_tool = context_data->rns_tool();
        
        size_t encrypted_size = encrypted.polynomial_count();

        bool device = encrypted.on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        destination.resize(this->context(), next_context_data->parms_id(), encrypted_size, false);

        switch (scheme) {
            case SchemeType::BFV: {
                rns_tool.divide_and_round_q_last(encrypted.reference(), encrypted_size, destination.reference());
                break;
            }
            case SchemeType::CKKS: {
                rns_tool.divide_and_round_q_last_ntt(encrypted.reference(), encrypted_size, destination.reference(), context_data->small_ntt_tables(), pool);
                break;
            }
            case SchemeType::BGV: {
                rns_tool.mod_t_and_divide_q_last_ntt(encrypted.reference(), encrypted_size, destination.reference(), context_data->small_ntt_tables(), pool);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::mod_switch_scale_to_next_internal] Scheme not implemented.");
            }
        }

        destination.is_ntt_form() = encrypted.is_ntt_form();
        if (scheme == SchemeType::CKKS) {
            // take the last modulus
            size_t id = parms.coeff_modulus().size() - 1;
            destination.scale() = encrypted.scale() / parms.coeff_modulus_host()[id].value();
        } else if (scheme == SchemeType::BGV) {
            destination.correction_factor() = utils::multiply_uint64_mod(
                encrypted.correction_factor(), rns_tool.inv_q_last_mod_t(), next_parms.plain_modulus_host()
            );
        }
    }


    void Evaluator::mod_switch_scale_to_next_internal_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::mod_switch_scale_to_next_internal_batched] Size mismatch.");
        }
        if (encrypted.empty()) return;
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_scale_to_next_internal]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                check_is_not_ntt_form_vec("[Evaluator::mod_switch_scale_to_next_internal]", encrypted);
                break;
            }
            case SchemeType::CKKS: case SchemeType::BGV: {
                check_is_ntt_form_vec("[Evaluator::mod_switch_scale_to_next_internal]", encrypted);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::mod_switch_scale_to_next_internal] Scheme not implemented.");
            }
        }
        if (!context_data->next_context_data().has_value()) {
            throw std::invalid_argument("[Evaluator::mod_switch_scale_to_next_internal] Next context data is not set.");
        }
        ContextDataPointer next_context_data = context_data->next_context_data().value();
        const EncryptionParameters& next_parms = next_context_data->parms();
        const RNSTool& rns_tool = context_data->rns_tool();
        
        size_t encrypted_size = get_vec_polynomial_count(encrypted);

        bool device = this->on_device();
        size_t n = encrypted.size();
        for (Ciphertext* d: destination) {
            *d = Ciphertext();
            if (device) d->to_device_inplace(pool);
            else d->to_host_inplace();
            d->resize(this->context(), next_context_data->parms_id(), encrypted_size, false);
        }

        auto encrypted_reference = batch_utils::pcollect_const_reference(encrypted);
        auto destination_reference = batch_utils::pcollect_reference(destination);
        switch (scheme) {
            case SchemeType::BFV: {
                rns_tool.divide_and_round_q_last_batched(encrypted_reference, encrypted_size, destination_reference, pool);
                break;
            }
            case SchemeType::CKKS: {
                rns_tool.divide_and_round_q_last_ntt_batched(encrypted_reference, encrypted_size, destination_reference, context_data->small_ntt_tables(), pool);
                break;
            }
            case SchemeType::BGV: {
                rns_tool.mod_t_and_divide_q_last_ntt_batched(encrypted_reference, encrypted_size, destination_reference, context_data->small_ntt_tables(), pool);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::mod_switch_scale_to_next_internal] Scheme not implemented.");
            }
        }

        for (size_t i = 0; i < n; i++) {
            destination[i]->is_ntt_form() = encrypted[i]->is_ntt_form();
            if (scheme == SchemeType::CKKS) {
                // take the last modulus
                size_t id = parms.coeff_modulus().size() - 1;
                destination[i]->scale() = encrypted[i]->scale() / parms.coeff_modulus_host()[id].value();
            } else if (scheme == SchemeType::BGV) {
                destination[i]->correction_factor() = utils::multiply_uint64_mod(
                    encrypted[i]->correction_factor(), rns_tool.inv_q_last_mod_t(), next_parms.plain_modulus_host()
                );
            }
        }
    }


    __device__ static void device_mod_switch_drop_to(ConstSlice<uint64_t> source, size_t poly_count, size_t source_modulus_size, size_t remain_modulus_size, size_t degree, Slice<uint64_t> destination) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= remain_modulus_size * degree) return;
        size_t i = global_index % degree;
        size_t j = global_index / degree;
        for (size_t p = 0; p < poly_count; p++) {
            size_t source_index = (p * source_modulus_size + j) * degree + i;
            size_t dest_index = (p * remain_modulus_size + j) * degree + i;
            destination[dest_index] = source[source_index];
        }
    }
    
    __global__ static void kernel_mod_switch_drop_to(ConstSlice<uint64_t> source, size_t poly_count, size_t source_modulus_size, size_t remain_modulus_size, size_t degree, Slice<uint64_t> destination) { 
        device_mod_switch_drop_to(source, poly_count, source_modulus_size, remain_modulus_size, degree, destination);
    }

    __global__ static void kernel_mod_switch_drop_to_batched(utils::ConstSliceArrayRef<uint64_t> source, size_t poly_count, size_t source_modulus_size, size_t remain_modulus_size, size_t degree, utils::SliceArrayRef<uint64_t> destination) {
        size_t i = blockIdx.y;
        device_mod_switch_drop_to(source[i], poly_count, source_modulus_size, remain_modulus_size, degree, destination[i]);
    }

    void Evaluator::mod_switch_drop_to_internal(const Ciphertext& encrypted, Ciphertext& destination, ParmsID target_parms_id, MemoryPoolHandle pool) const {
        ParmsID parms_id = encrypted.parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_scale_to_next_internal]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        if (scheme == SchemeType::CKKS) {
            check_is_ntt_form("[Evaluator::mod_switch_drop_to_internal]", encrypted);
        }
        if (!context_data->next_context_data().has_value()) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_next_internal] Next context data is not set.");
        }
        ContextDataPointer target_context_data = this->get_context_data("[Evaluator::mod_switch_drop_to_next_internal]", target_parms_id);
        const EncryptionParameters& target_parms = target_context_data->parms();
        if (!is_scale_within_bounds(encrypted.scale(), target_context_data)) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_internal] Scale out of bounds.");
        }
        
        size_t encrypted_size = encrypted.polynomial_count();
        size_t coeff_count = target_parms.poly_modulus_degree();
        size_t target_coeff_modulus_size = target_parms.coeff_modulus().size();

        destination = Ciphertext::like(encrypted, false, pool);

        destination.resize(this->context(), target_parms_id, encrypted_size, false, false);
        
        if (encrypted.on_device()) {
            size_t block_count = utils::ceil_div(target_coeff_modulus_size * coeff_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(encrypted.data().device_index());
            kernel_mod_switch_drop_to<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                encrypted.data().const_reference(), encrypted_size, parms.coeff_modulus().size(), target_coeff_modulus_size, coeff_count, destination.data().reference()
            );
            utils::stream_sync();
        } else {
            for (size_t p = 0; p < encrypted_size; p++) {
                for (size_t i = 0; i < coeff_count; i++) {
                    for (size_t j = 0; j < target_coeff_modulus_size; j++) {
                        size_t source_index = (p * parms.coeff_modulus().size() + j) * coeff_count + i;
                        size_t dest_index = (p * target_parms.coeff_modulus().size() + j) * coeff_count + i;
                        destination.data()[dest_index] = encrypted.data()[source_index];
                    }
                }
            }
        }

        destination.is_ntt_form() = encrypted.is_ntt_form();
        destination.scale() = encrypted.scale();
        destination.correction_factor() = encrypted.correction_factor();
    }

    
    void Evaluator::mod_switch_drop_to_internal_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Ciphertext*>& destination, ParmsID target_parms_id, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_internal_batched] Size mismatch.");
        }
        if (encrypted.empty()) return;
        if (!this->on_device() || encrypted.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                this->mod_switch_drop_to_internal(*encrypted[i], *destination[i], target_parms_id, pool);
            }
            return;
        }
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_drop_to_internal_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        if (scheme == SchemeType::CKKS) {
            check_is_ntt_form_vec("[Evaluator::mod_switch_drop_to_internal_batched]", encrypted);
        }
        if (!context_data->next_context_data().has_value()) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_internal_batched] Next context data is not set.");
        }
        ContextDataPointer target_context_data = this->get_context_data("[Evaluator::mod_switch_drop_to_internal_batched]", target_parms_id);
        const EncryptionParameters& target_parms = target_context_data->parms();
        double scale = get_vec_scale(encrypted);
        if (!is_scale_within_bounds(scale, target_context_data)) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_internal] Scale out of bounds.");
        }
        
        size_t encrypted_size = get_vec_polynomial_count(encrypted);
        size_t coeff_count = target_parms.poly_modulus_degree();
        size_t target_coeff_modulus_size = target_parms.coeff_modulus().size();

        size_t n = encrypted.size();
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
            destination[i]->resize(this->context(), target_parms_id, encrypted_size, false, false);
        }
        
        size_t block_count = utils::ceil_div(target_coeff_modulus_size * coeff_count, utils::KERNEL_THREAD_COUNT);
        dim3 block_dims(block_count, n);
        auto comp_ref = parms.coeff_modulus();
        auto encrypted_batched = batch_utils::construct_batch(batch_utils::pcollect_const_reference(encrypted), pool, comp_ref);
        auto destination_batched = batch_utils::construct_batch(batch_utils::pcollect_reference(destination), pool, comp_ref);
        utils::set_device(comp_ref.device_index());
        kernel_mod_switch_drop_to_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
            encrypted_batched, encrypted_size, parms.coeff_modulus().size(), target_coeff_modulus_size, coeff_count, destination_batched
        );
        utils::stream_sync();

        for (size_t i = 0; i < n; i++) {
            destination[i]->is_ntt_form() = encrypted[i]->is_ntt_form();
            destination[i]->scale() = encrypted[i]->scale();
            destination[i]->correction_factor() = encrypted[i]->correction_factor();
        }
    }

    void Evaluator::mod_switch_drop_to_plain_internal(const Plaintext& plain, Plaintext& destination, ParmsID target_parms_id, MemoryPoolHandle pool) const {
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_plain_internal] Plaintext is not in NTT form.");
        }
        ParmsID parms_id = plain.parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_drop_to_plain_internal]", parms_id);
        
        if (!context_data->next_context_data().has_value()) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_plain_internal] Next context data is not set.");
        }
        ContextDataPointer target_context_data = this->get_context_data("[Evaluator::mod_switch_drop_to_plain_internal]", target_parms_id);
        const EncryptionParameters& target_parms = target_context_data->parms();
        if (!is_scale_within_bounds(plain.scale(), target_context_data)) {
            throw std::invalid_argument("[Evaluator::mod_switch_drop_to_plain_internal] Scale out of bounds.");
        }

        destination = Plaintext::like(plain, false, pool);
        destination.resize_rns(*context(), target_parms_id, false);

        if (plain.on_device()) {
            size_t block_count = utils::ceil_div(target_parms.coeff_modulus().size() * target_parms.poly_modulus_degree(), utils::KERNEL_THREAD_COUNT);
            utils::set_device(plain.data().device_index());
            kernel_mod_switch_drop_to<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain.data().const_reference(), 1, 
                context_data->parms().coeff_modulus().size(), 
                target_parms.coeff_modulus().size(), target_parms.poly_modulus_degree(), destination.data().reference()
            );
            utils::stream_sync();
        } else {
            for (size_t i = 0; i < target_parms.coeff_modulus().size(); i++) {
                for (size_t j = 0; j < target_parms.poly_modulus_degree(); j++) {
                    size_t source_index = i * context_data->parms().poly_modulus_degree() + j;
                    size_t dest_index = i * target_parms.poly_modulus_degree() + j;
                    destination.data()[dest_index] = plain.data()[source_index];
                }
            }
        }
    }

    void Evaluator::mod_switch_to_next(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::mod_switch_to_next]", encrypted);
        if (this->context()->last_parms_id() == encrypted.parms_id()) {
            throw std::invalid_argument("[Evaluator::mod_switch_to_next] End of modulus switching chain reached.");
        }
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: 
                this->mod_switch_scale_to_next_internal(encrypted, destination, pool);
                break;
            case SchemeType::CKKS: {
                auto context_data = this->get_context_data("[Evaluator::mod_switch_to_next]", encrypted.parms_id());
                if (!context_data->next_context_data().has_value()) {
                    throw std::invalid_argument("[Evaluator::mod_switch_to_next] Next context data is not set.");
                }
                auto target_context_data = context_data->next_context_data().value();
                this->mod_switch_drop_to_internal(encrypted, destination, target_context_data->parms_id(), pool);
                break;
            }
            case SchemeType::BGV:
                this->mod_switch_scale_to_next_internal(encrypted, destination, pool);
                break;
            default:
                throw std::logic_error("[Evaluator::mod_switch_to_next] Scheme not implemented.");
        }
    }


    void Evaluator::mod_switch_to_next_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::mod_switch_to_next_batched] Size mismatch.");
        }
        if (encrypted.size() == 0) return;
        check_no_seed_vec("[Evaluator::mod_switch_to_next]", encrypted);
        ParmsID parms_id = get_vec_parms_id(encrypted);
        if (this->context()->last_parms_id() == parms_id) {
            throw std::invalid_argument("[Evaluator::mod_switch_to_next_batched] End of modulus switching chain reached.");
        }
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: 
                this->mod_switch_scale_to_next_internal_batched(encrypted, destination, pool);
                break;
            case SchemeType::CKKS: {
                auto context_data = this->get_context_data("[Evaluator::mod_switch_to_next]", parms_id);
                if (!context_data->next_context_data().has_value()) {
                    throw std::invalid_argument("[Evaluator::mod_switch_to_next] Next context data is not set.");
                }
                auto target_context_data = context_data->next_context_data().value();
                this->mod_switch_drop_to_internal_batched(encrypted, destination, target_context_data->parms_id(), pool);
                break;
            }
            case SchemeType::BGV:
                this->mod_switch_scale_to_next_internal_batched(encrypted, destination, pool);
                break;
            default:
                throw std::logic_error("[Evaluator::mod_switch_to_next] Scheme not implemented.");
        }
    }


    void Evaluator::mod_switch_to(const Ciphertext& encrypted, const ParmsID& parms_id, Ciphertext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_to_inplace]", encrypted.parms_id());
        ContextDataPointer target_context_data = this->get_context_data("[Evaluator::mod_switch_to_inplace]", parms_id);
        if (context_data->chain_index() < target_context_data->chain_index()) {
            throw std::invalid_argument("[Evaluator::mod_switch_to_inplace] Cannot switch to a higher level.");
        }
        if (encrypted.parms_id() == parms_id) {
            destination = encrypted.clone(pool); return;
        }
        if (context_data->parms().scheme() == SchemeType::CKKS) {
            this->mod_switch_drop_to_internal(encrypted, destination, parms_id, pool);
        } else {
            bool first = true;
            while (true) {
                if (first) {this->mod_switch_to_next(encrypted, destination, pool); first = false;}
                else this->mod_switch_to_next_inplace(destination, pool);
                if (destination.parms_id() == parms_id) break;
            }
        }
    }

    void Evaluator::mod_switch_to_batched(const std::vector<const Ciphertext*>& encrypted, const ParmsID& parms_id, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::mod_switch_to_batched] Size mismatch.");
        }
        if (encrypted.size() == 0) return;
        ParmsID original_parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_to_inplace]", original_parms_id);
        ContextDataPointer target_context_data = this->get_context_data("[Evaluator::mod_switch_to_inplace]", parms_id);
        if (context_data->chain_index() < target_context_data->chain_index()) {
            throw std::invalid_argument("[Evaluator::mod_switch_to_inplace] Cannot switch to a higher level.");
        }
        if (original_parms_id == parms_id) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                *destination[i] = encrypted[i]->clone(pool); 
            }
            return;
        }
        if (context_data->parms().scheme() == SchemeType::CKKS) {
            this->mod_switch_drop_to_internal_batched(encrypted, destination, parms_id, pool);
        } else {
            bool first = true;
            while (true) {
                if (first) {this->mod_switch_to_next_batched(encrypted, destination, pool); first = false;}
                else this->mod_switch_to_next_inplace_batched(destination, pool);
                if (destination[0]->parms_id() == parms_id) break;
            }
        }
    }


    void Evaluator::mod_switch_plain_to(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination, MemoryPoolHandle pool) const {
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("[Evaluator::mod_switch_plain_to_inplace] Plaintext is not in NTT form.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::mod_switch_plain_to_inplace]", plain.parms_id());
        ContextDataPointer target_context_data = this->get_context_data("[Evaluator::mod_switch_plain_to_inplace]", parms_id);
        if (context_data->chain_index() < target_context_data->chain_index()) {
            throw std::invalid_argument("[Evaluator::mod_switch_plain_to_inplace] Cannot switch to a higher level.");
        }
        if (plain.parms_id() == parms_id) {
            destination = plain.clone(); return;
        }
        this->mod_switch_drop_to_plain_internal(plain, destination, parms_id, pool);
    }

    void Evaluator::rescale_to_next(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::rescale_to_next]", encrypted);
        if (this->context()->last_parms_id() == encrypted.parms_id()) {
            throw std::invalid_argument("[Evaluator::rescale_to_next] End of modulus switching chain reached.");
        }
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: case SchemeType::BGV:
                throw std::invalid_argument("[Evaluator::rescale_to_next] Cannot rescale BFV/BGV ciphertext.");
                break;
            case SchemeType::CKKS:
                this->mod_switch_scale_to_next_internal(encrypted, destination, pool);
                break;
            default:
                throw std::logic_error("[Evaluator::rescale_to_next] Scheme not implemented.");
        }
    }
    
    void Evaluator::rescale_to(const Ciphertext& encrypted, const ParmsID& parms_id, Ciphertext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->get_context_data("[Evaluator::rescale_to]", encrypted.parms_id());
        ContextDataPointer target_context_data = this->get_context_data("[Evaluator::rescale_to]", parms_id);
        if (context_data->chain_index() < target_context_data->chain_index()) {
            throw std::invalid_argument("[Evaluator::rescale_to] Cannot rescale to a higher level.");
        }
        while (encrypted.parms_id() != parms_id) {
            this->rescale_to_next(encrypted, destination, pool);
        }
    }

}