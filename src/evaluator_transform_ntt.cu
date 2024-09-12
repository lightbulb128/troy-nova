#include "batch_utils.h"
#include "encryption_parameters.h"
#include "evaluator.h"
#include "evaluator_utils.h"
#include "fgk/translate_plain.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::NTTTables;
    using utils::ConstPointer;
    using utils::RNSTool;
    using utils::MultiplyUint64Operand;
    using utils::GaloisTool;





    /// Transforms a plaintext from normal form to NTT form.
    /// 
    /// The input plaintext must be from the BFV scheme, and in this function we do
    /// not scale by Delta = q/t. Two situations:
    /// 1. The plaintext is modulo t. In this case, it is converted to mod q NTT form by centralize, (so that
    ///    it may be multiplied with a ciphertext.) Note, since we don't scale by Delta,
    ///    one usually cannot directly add this converted plaintext to a ciphertext (which must
    ///    have already been scaled by Delta.) This equals first calling BatchEncoder.centralize
    ///    then invoking this (which is hence case 2).
    /// 2. The plaintext is already modulo some q. In this case, it is directly NTT-ed. You
    ///    must ensure the given parms_id is consistent with the plaintext's own parms_id. 
    ///    Usually, this form of plaintext is created with [BatchEncoder::scale_up],
    ///    so after NTT-ed it could still be added to a ciphertext, but it cannot be 
    ///    multiplied with a BFV ciphertext, since both the operands are already scaled.
    void Evaluator::transform_plain_to_ntt(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination, MemoryPoolHandle pool) const {
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("[Evaluator::transform_plain_to_ntt] Plaintext is already in NTT form.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        destination = Plaintext::like(plain, false, pool);

        if (plain.on_device()) destination.to_device_inplace(pool);
        destination.resize_rns(*context_, parms_id);
        if (plain.parms_id() == parms_id_zero) {
            scaling_variant::centralize(plain, context_data, destination.poly(), coeff_count, pool);
            utils::ntt_inplace_p(destination.poly(), coeff_count, ntt_tables);
            destination.is_ntt_form() = true;
            destination.coeff_modulus_size() = coeff_modulus_size;
            destination.poly_modulus_degree() = coeff_count;
        } else {
            if (plain.parms_id() != parms_id) {
                throw std::invalid_argument("[Evaluator::transform_plain_to_ntt] Plaintext parameters do not match.");
            }
            if (plain.coeff_count() != coeff_count) {
                Plaintext cloned; if (plain.on_device()) cloned.to_device_inplace(pool);
                cloned.resize_rns(*context_, plain.parms_id(), false);
                utils::fgk::translate_plain::scatter_translate_copy(nullptr, plain.const_poly(), coeff_count, plain.coeff_count(), coeff_modulus, cloned.poly(), false);
                utils::ntt_p(cloned.poly(), coeff_count, ntt_tables, destination.poly());
            } else {
                utils::ntt_p(plain.poly(), coeff_count, ntt_tables, destination.poly());
            }
            destination.is_ntt_form() = true;
        }
    }


    void Evaluator::transform_plain_to_ntt_batched(const std::vector<const Plaintext*>& plain, const ParmsID& parms_id, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (plain.size() != destination.size()) throw std::invalid_argument("[Evaluator::transform_plain_to_ntt_batched] The number of plaintexts does not match the number of destinations.");
        if (plain.size() == 0) return;
        check_is_not_ntt_form_vec("[Evaluator::transform_plain_to_ntt_batched]", plain);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        size_t n = plain.size();
        for (size_t i = 0; i < n; i++)
            *destination[i] = Plaintext::like(*plain[i], false, pool);

        auto plain_parms_id = get_vec_parms_id(plain);
        for (size_t i = 0; i < n; i++) {
            if (plain[i]->on_device()) destination[i]->to_device_inplace(pool);
            destination[i]->resize_rns(*context_, parms_id);
        }

        if (plain_parms_id == parms_id_zero) {
            auto destination_batched = batch_utils::pcollect_poly(destination);
            scaling_variant::centralize_batched(plain, context_data, destination_batched, coeff_count, pool);
            utils::ntt_inplace_bp(destination_batched, coeff_count, ntt_tables, pool);
            for (size_t i = 0; i < n; i++) {
                destination[i]->is_ntt_form() = true;
                destination[i]->coeff_modulus_size() = coeff_modulus_size;
                destination[i]->poly_modulus_degree() = coeff_count;
            }
        } else {
            if (plain_parms_id != parms_id) {
                throw std::invalid_argument("[Evaluator::transform_plain_to_ntt_batched] Plaintext parameters do not match.");
            }
            auto plain_coeff_count = get_vec_coeff_count(plain);
            if (plain_coeff_count != coeff_count) {
                std::vector<Plaintext> cloned(n); 
                for (size_t i = 0; i < n; i++) {
                    if (plain[i]->on_device()) cloned[i].to_device_inplace(pool);
                    cloned[i].resize_rns(*context_, plain_parms_id, false);
                }
                auto plain_batched = batch_utils::pcollect_const_poly(plain);
                auto cloned_batched = batch_utils::rcollect_poly(cloned);
                auto from_batched = std::vector<ConstSlice<uint64_t>>();
                for (size_t i = 0; i < n; i++) from_batched.push_back(nullptr);
                utils::fgk::translate_plain::scatter_translate_copy_batched(from_batched, plain_batched, coeff_count, plain_coeff_count, coeff_modulus, cloned_batched, false);
                auto destination_batched = batch_utils::pcollect_poly(destination);
                auto cloned_const_batched = batch_utils::rcollect_const_poly(cloned);
                utils::ntt_bp(cloned_const_batched, coeff_count, ntt_tables, destination_batched, pool);
            } else {
                auto plain_batched = batch_utils::pcollect_const_poly(plain);
                auto destination_batched = batch_utils::pcollect_poly(destination);
                utils::ntt_bp(plain_batched, coeff_count, ntt_tables, destination_batched, pool);
            }
            for (size_t i = 0; i < n; i++) destination[i]->is_ntt_form() = true;
        }
    }

    void Evaluator::bfv_centralize(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination, MemoryPoolHandle pool) const {
        check_is_not_ntt_form("[Evaluator::bfv_centralize]", plain);
        if (plain.parms_id() != parms_id_zero) {
            throw std::invalid_argument("[Evaluator::bfv_centralize] Plaintext is not modulo t.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();

        destination = Plaintext::like(plain, false, pool);

        if (plain.on_device()) destination.to_device_inplace(pool);
        destination.resize_rns(*context_, parms_id);
        scaling_variant::centralize(plain, context_data, destination.poly(), coeff_count, pool);
        destination.is_ntt_form() = false;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
    }

    void Evaluator::bfv_centralize_batched(const std::vector<const Plaintext*>& plain, const ParmsID& parms_id, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (plain.size() != destination.size()) throw std::invalid_argument("[Evaluator::transform_plain_to_ntt_batched] The number of plaintexts does not match the number of destinations.");
        if (plain.size() == 0) return;
        check_is_not_ntt_form_vec("[Evaluator::transform_plain_to_ntt_batched]", plain);
        for (size_t i = 0; i < plain.size(); i++) {
            if (plain[i]->parms_id() != parms_id_zero) {
                throw std::invalid_argument("[Evaluator::bfv_centralize_batched] Plaintext is not modulo t.");
            }
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();

        size_t n = plain.size();
        for (size_t i = 0; i < n; i++)
            *destination[i] = Plaintext::like(*plain[i], false, pool);

        for (size_t i = 0; i < n; i++) {
            if (plain[i]->on_device()) destination[i]->to_device_inplace(pool);
            destination[i]->resize_rns(*context_, parms_id);
        }

        auto destination_batched = batch_utils::pcollect_poly(destination);
        scaling_variant::centralize_batched(plain, context_data, destination_batched, coeff_count, pool);
        for (size_t i = 0; i < n; i++) {
            destination[i]->is_ntt_form() = false;
            destination[i]->coeff_modulus_size() = coeff_modulus_size;
            destination[i]->poly_modulus_degree() = coeff_count;
        }
    }

    void Evaluator::bfv_scale_up(const Plaintext& plain, const ParmsID& parms_id, Plaintext& destination, MemoryPoolHandle pool) const {
        check_is_not_ntt_form("[Evaluator::bfv_centralize]", plain);
        if (plain.parms_id() != parms_id_zero) {
            throw std::invalid_argument("[Evaluator::bfv_centralize] Plaintext is not modulo t.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();

        destination = Plaintext::like(plain, false, pool);

        if (plain.on_device()) destination.to_device_inplace(pool);
        destination.resize_rns(*context_, parms_id);
        scaling_variant::scale_up(plain, context_data, destination.poly(), coeff_count, nullptr, false);
        destination.is_ntt_form() = false;
        destination.coeff_modulus_size() = coeff_modulus_size;
        destination.poly_modulus_degree() = coeff_count;
    }

    void Evaluator::bfv_scale_up_batched(const std::vector<const Plaintext*>& plain, const ParmsID& parms_id, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (plain.size() != destination.size()) throw std::invalid_argument("[Evaluator::transform_plain_to_ntt_batched] The number of plaintexts does not match the number of destinations.");
        if (plain.size() == 0) return;
        check_is_not_ntt_form_vec("[Evaluator::transform_plain_to_ntt_batched]", plain);
        for (size_t i = 0; i < plain.size(); i++) {
            if (plain[i]->parms_id() != parms_id_zero) {
                throw std::invalid_argument("[Evaluator::bfv_centralize_batched] Plaintext is not modulo t.");
            }
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();

        size_t n = plain.size();
        for (size_t i = 0; i < n; i++)
            *destination[i] = Plaintext::like(*plain[i], false, pool);

        for (size_t i = 0; i < n; i++) {
            if (plain[i]->on_device()) destination[i]->to_device_inplace(pool);
            destination[i]->resize_rns(*context_, parms_id);
        }

        auto destination_batched = batch_utils::pcollect_poly(destination);
        std::vector<ConstSlice<uint64_t>> froms; for (size_t i = 0; i < n; i++) froms.push_back(nullptr);
        scaling_variant::scale_up_batched(plain, context_data, destination_batched, coeff_count, froms, false, pool);
        for (size_t i = 0; i < n; i++) {
            destination[i]->is_ntt_form() = false;
            destination[i]->coeff_modulus_size() = coeff_modulus_size;
            destination[i]->poly_modulus_degree() = coeff_count;
        }
    }













    // TODO: why not simply use transform_plain_to_ntt and then move to original place?
    void Evaluator::transform_plain_to_ntt_inplace(Plaintext& plain, const ParmsID& parms_id, MemoryPoolHandle pool) const {
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("[Evaluator::transform_plain_to_ntt_inplace] Plaintext is already in NTT form.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        if (plain.parms_id() == parms_id_zero) {
            Plaintext plain_copy; 
            if (plain.on_device()) plain_copy.to_device_inplace(pool);
            plain_copy.resize_rns(*context_, parms_id);
            scaling_variant::centralize(plain, context_data, plain_copy.poly(), coeff_count, pool);
            plain = std::move(plain_copy);

            utils::ntt_inplace_p(plain.poly(), coeff_count, ntt_tables);
            plain.is_ntt_form() = true;
            plain.coeff_modulus_size() = coeff_modulus_size;
            plain.poly_modulus_degree() = coeff_count;
        } else {
            if (plain.parms_id() != parms_id) {
                throw std::invalid_argument("[Evaluator::transform_plain_to_ntt_inplace] Plaintext parameters do not match.");
            }
            if (plain.coeff_count() != coeff_count) {
                Plaintext cloned; if (plain.on_device()) cloned.to_device_inplace(pool);
                cloned.resize_rns(*context_, plain.parms_id(), false);
                utils::fgk::translate_plain::scatter_translate_copy(nullptr, plain.const_poly(), coeff_count, plain.coeff_count(), coeff_modulus, cloned.poly(), false);
                plain = std::move(cloned);
            }
            utils::ntt_inplace_p(plain.poly(), coeff_count, ntt_tables);
            plain.is_ntt_form() = true;
        }
    }

    
    void Evaluator::transform_plain_to_ntt_inplace_batched(const std::vector<Plaintext*>& plain, const ParmsID& parms_id, MemoryPoolHandle pool) const {
        if (plain.size() == 0) return;
        check_is_not_ntt_form_vec("[Evaluator::transform_plain_to_ntt_batched]", plain);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        auto plain_parms_id = get_vec_parms_id(plain);
        size_t n = plain.size();
        if (plain_parms_id == parms_id_zero) {
            std::vector<Plaintext> plain_copy(n);
            for (size_t i = 0; i < n; i++) {
                if (plain[i]->on_device()) plain_copy[i].to_device_inplace(pool);
                plain_copy[i].resize_rns(*context_, parms_id);
            }
            
            auto plain_copy_batched = batch_utils::rcollect_poly(plain_copy);
            auto plain_const_batched = batch_utils::pcollect_const_pointer(plain);
            scaling_variant::centralize_batched(plain_const_batched, context_data, plain_copy_batched, coeff_count, pool);
            for (size_t i = 0; i < n; i++) {
                *plain[i] = std::move(plain_copy[i]);
            }

            auto plain_batched = batch_utils::pcollect_poly(plain);
            utils::ntt_inplace_bp(plain_batched, coeff_count, ntt_tables, pool);
            
            for (size_t i = 0; i < n; i++) {
                plain[i]->is_ntt_form() = true;
                plain[i]->coeff_modulus_size() = coeff_modulus_size;
                plain[i]->poly_modulus_degree() = coeff_count;
            }
        } else {
            if (plain_parms_id != parms_id) {
                throw std::invalid_argument("[Evaluator::transform_plain_to_ntt_inplace] Plaintext parameters do not match.");
            }
            auto plain_coeff_count = get_vec_coeff_count(plain);
            if (plain_coeff_count != coeff_count) {
                std::vector<Plaintext> cloned(n); 
                for (size_t i = 0; i < n; i++) {
                    if (plain[i]->on_device()) cloned[i].to_device_inplace(pool);
                    cloned[i].resize_rns(*context_, plain_parms_id, false);
                }
                auto plain_const = batch_utils::pcollect_const_pointer(plain);
                auto plain_batched = batch_utils::pcollect_const_poly(plain_const);
                auto cloned_batched = batch_utils::rcollect_poly(cloned);
                auto from_batched = std::vector<ConstSlice<uint64_t>>();
                for (size_t i = 0; i < n; i++) from_batched.push_back(nullptr);
                utils::fgk::translate_plain::scatter_translate_copy_batched(from_batched, plain_batched, coeff_count, plain_coeff_count, coeff_modulus, cloned_batched, false, pool);
                for (size_t i = 0; i < n; i++) {
                    *plain[i] = std::move(cloned[i]);
                }
            }
            auto plain_batched = batch_utils::pcollect_poly(plain);
            utils::ntt_inplace_bp(plain_batched, coeff_count, ntt_tables, pool);
            for (size_t i = 0; i < n; i++) plain[i]->is_ntt_form() = true;
        }
    }















    void Evaluator::transform_plain_from_ntt(const Plaintext& plain, Plaintext& destination, MemoryPoolHandle pool) const {
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("[Evaluator::transform_plain_from_ntt_inplace] Plaintext is already in NTT form.");
        }
        if (plain.parms_id() == parms_id_zero) {
            throw std::invalid_argument("[Evaluator::transform_plain_from_ntt_inplace] Invalid ParmsID, but this should never be reached.");
        }
        ParmsID parms_id = plain.parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_from_ntt_inplace]", parms_id);

        destination = Plaintext::like(plain, false, pool);

        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        utils::intt_p(plain.poly(), coeff_count, ntt_tables, destination.poly());
        destination.is_ntt_form() = false;
    }

    void Evaluator::transform_plain_from_ntt_batched(const std::vector<const Plaintext*>& plain, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        check_is_ntt_form_vec("[Evaluator::transform_plain_from_ntt_batched]", plain);
        auto parms_id = get_vec_parms_id(plain);
        if (parms_id == parms_id_zero) {
            throw std::invalid_argument("[Evaluator::transform_plain_from_ntt_inplace] Invalid ParmsID, but this should never be reached.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_from_ntt_inplace]", parms_id);

        size_t n = plain.size();
        for (size_t i = 0; i < n; i++) *destination[i] = Plaintext::like(*plain[i], false, pool);

        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        auto plain_batched = batch_utils::pcollect_const_poly(plain);
        auto destination_batched = batch_utils::pcollect_poly(destination);
        utils::intt_bp(plain_batched, coeff_count, ntt_tables, destination_batched, pool);
        for (size_t i = 0; i < n; i++) destination[i]->is_ntt_form() = false;
    }
    













    void Evaluator::transform_plain_from_ntt_inplace(Plaintext& plain) const {
        if (!plain.is_ntt_form()) {
            throw std::invalid_argument("[Evaluator::transform_plain_from_ntt_inplace] Plaintext is already in NTT form.");
        }
        if (plain.parms_id() == parms_id_zero) {
            throw std::invalid_argument("[Evaluator::transform_plain_from_ntt_inplace] Invalid ParmsID, but this should never be reached.");
        }
        ParmsID parms_id = plain.parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_from_ntt_inplace]", parms_id);

        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        utils::intt_inplace_p(plain.poly(), coeff_count, ntt_tables);
        plain.is_ntt_form() = false;
    }

    void Evaluator::transform_plain_from_ntt_inplace_batched(const std::vector<Plaintext*>& plain, MemoryPoolHandle pool) const {
        check_is_ntt_form_vec("[Evaluator::transform_plain_from_ntt_inplace_batched]", plain);
        auto parms_id = get_vec_parms_id(plain);
        if (parms_id == parms_id_zero) {
            throw std::invalid_argument("[Evaluator::transform_plain_from_ntt_inplace_batched] Invalid ParmsID, but this should never be reached.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_plain_from_ntt_inplace_batched]", parms_id);

        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();

        auto plain_batched = batch_utils::pcollect_poly(plain);
        utils::intt_inplace_bp(plain_batched, coeff_count, ntt_tables, pool);
        for (size_t i = 0; i < plain.size(); i++) plain[i]->is_ntt_form() = false;
    }















    void Evaluator::transform_to_ntt(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::transform_to_ntt]", encrypted);
        check_is_not_ntt_form("[Evaluator::transform_to_ntt]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        destination = Ciphertext::like(encrypted, false, pool);
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::ntt_ps(
            encrypted.const_reference(),
            encrypted.polynomial_count(), 
            coeff_count, ntt_tables,
            destination.reference()
        );
        destination.is_ntt_form() = true;
    }

    void Evaluator::transform_to_ntt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        check_no_seed_vec("[Evaluator::transform_to_ntt_batched]", encrypted);
        check_is_not_ntt_form_vec("[Evaluator::transform_to_ntt_batched]", encrypted);
        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t n = encrypted.size();
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        }
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::ntt_bps(
            batch_utils::pcollect_const_reference(encrypted),
            get_vec_polynomial_count(encrypted),
            coeff_count, ntt_tables,
            batch_utils::pcollect_reference(destination),
            pool
        );
        for (size_t i = 0; i < n; i++) 
            destination[i]->is_ntt_form() = true;


    }
    














    void Evaluator::transform_to_ntt_inplace(Ciphertext& encrypted) const {
        check_no_seed("[Evaluator::transform_to_ntt_inplace]", encrypted);
        check_is_not_ntt_form("[Evaluator::transform_to_ntt_inplace]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::ntt_inplace_ps(
            encrypted.polys(0, encrypted.polynomial_count()), 
            encrypted.polynomial_count(), 
            coeff_count, ntt_tables
        );
        encrypted.is_ntt_form() = true;
    }

    void Evaluator::transform_to_ntt_inplace_batched(const std::vector<Ciphertext*>& encrypted, MemoryPoolHandle pool) const {
        check_no_seed_vec("[Evaluator::transform_to_ntt_inplace_batched]", encrypted);
        check_is_not_ntt_form_vec("[Evaluator::transform_to_ntt_inplace_batched]", encrypted);
        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::ntt_inplace_bps(
            batch_utils::pcollect_reference(encrypted),
            get_vec_polynomial_count(encrypted),
            coeff_count, ntt_tables,
            pool
        );
        for (size_t i = 0; i < encrypted.size(); i++) 
            encrypted[i]->is_ntt_form() = true;
    }














    void Evaluator::transform_from_ntt(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::transform_to_ntt_inplace]", encrypted);
        check_is_ntt_form("[Evaluator::transform_to_ntt_inplace]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        destination = Ciphertext::like(encrypted, false, pool);
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::intt_ps(
            encrypted.const_reference(),
            encrypted.polynomial_count(), 
            coeff_count, ntt_tables,
            destination.reference()
        );
        destination.is_ntt_form() = false;
    }

    void Evaluator::transform_from_ntt_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        check_no_seed_vec("[Evaluator::transform_to_ntt_batched]", encrypted);
        check_is_ntt_form_vec("[Evaluator::transform_to_ntt_batched]", encrypted);
        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t n = encrypted.size();
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        }
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::intt_bps(
            batch_utils::pcollect_const_reference(encrypted),
            get_vec_polynomial_count(encrypted),
            coeff_count, ntt_tables,
            batch_utils::pcollect_reference(destination),
            pool
        );
        for (size_t i = 0; i < n; i++) 
            destination[i]->is_ntt_form() = false;
    }











    void Evaluator::transform_from_ntt_inplace(Ciphertext& encrypted) const {
        check_no_seed("[Evaluator::transform_to_ntt_inplace]", encrypted);
        check_is_ntt_form("[Evaluator::transform_to_ntt_inplace]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::intt_inplace_ps(
            encrypted.polys(0, encrypted.polynomial_count()), 
            encrypted.polynomial_count(), 
            coeff_count, ntt_tables
        );
        encrypted.is_ntt_form() = false;
    }    

    void Evaluator::transform_from_ntt_inplace_batched(const std::vector<Ciphertext*>& encrypted, MemoryPoolHandle pool) const {
        check_no_seed_vec("[Evaluator::transform_to_ntt_inplace_batched]", encrypted);
        check_is_ntt_form_vec("[Evaluator::transform_to_ntt_inplace_batched]", encrypted);
        auto parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::transform_to_ntt_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        utils::intt_inplace_bps(
            batch_utils::pcollect_reference(encrypted),
            get_vec_polynomial_count(encrypted),
            coeff_count, ntt_tables,
            pool
        );
        for (size_t i = 0; i < encrypted.size(); i++) 
            encrypted[i]->is_ntt_form() = false;
    }

}