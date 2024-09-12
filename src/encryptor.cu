#include "batch_utils.h"
#include "encryption_parameters.h"
#include "encryptor.h"
#include "utils/constants.h"
#include "utils/scaling_variant.h"

namespace troy {

    using utils::ConstSlice;
    using utils::RNSTool;

    void Encryptor::encrypt_zero_internal(
        const ParmsID& parms_id, 
        bool is_ntt_form,
        bool is_asymmetric, bool save_seed, 
        utils::RandomGenerator* u_prng, 
        Ciphertext& destination,
        MemoryPoolHandle pool
    ) const {
        // sanity check
        if (is_asymmetric && !this->public_key_.has_value()) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Public key not set for asymmetric encryption.");
        }
        if (!is_asymmetric && !this->secret_key_.has_value()) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Secret key not set for symmetric encryption.");
        }
        if (save_seed && is_asymmetric) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Cannot save seed when using asymmetric encryption.");
        }
        // verify params
        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] parms_id is not valid for encryption parameters.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        
        // Resize destination and save results
        destination.seed() = 0;
        bool device = (is_asymmetric) ? this->public_key_->on_device() : this->secret_key_->on_device();
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        destination.resize(this->context(), parms_id, 2, false, false);
        
        if (is_asymmetric) {
            std::optional<std::weak_ptr<const ContextData>> prev_context_optional = context_data->prev_context_data();
            if (prev_context_optional.has_value()) {
                // Requires modulus switching
                ContextDataPointer prev_context_data = prev_context_optional.value().lock();
                ParmsID prev_parms_id = prev_context_data->parms_id();
                const RNSTool& rns_tool = prev_context_data->rns_tool();

                // Zero encryption without modulus switching
                Ciphertext temp;
                if (u_prng == nullptr) {
                    rlwe::asymmetric(
                        this->public_key(), this->context(), prev_parms_id, is_ntt_form, temp, pool
                    );
                } else {
                    rlwe::asymmetric_with_u_prng(
                        this->public_key(), this->context(), prev_parms_id, is_ntt_form, *u_prng, temp, pool
                    );
                }

                if (parms.scheme() == SchemeType::BGV && !is_ntt_form) {
                    throw std::invalid_argument("[Encryptor::encrypt_zero_internal] BGV - Plaintext is not in NTT form.");
                }
                
                // Modulus switching
                switch (parms.scheme()) {
                    case SchemeType::CKKS: case SchemeType::BFV: {
                        if (is_ntt_form) {
                            rns_tool.divide_and_round_q_last_ntt(temp.const_reference(), temp.polynomial_count(), destination.reference(), prev_context_data->small_ntt_tables(), pool);
                        } else {
                            rns_tool.divide_and_round_q_last(temp.const_reference(), temp.polynomial_count(), destination.reference());
                        }
                        break;
                    }
                    case SchemeType::BGV: {
                        if (is_ntt_form) {
                            rns_tool.mod_t_and_divide_q_last_ntt(temp.const_reference(), temp.polynomial_count(), destination.reference(), prev_context_data->small_ntt_tables(), pool);
                        }
                        break;
                    }
                    default:
                        throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Unsupported scheme.");
                }

                destination.parms_id() = parms_id;
                destination.is_ntt_form() = is_ntt_form;
                destination.scale() = temp.scale();
                destination.correction_factor() = temp.correction_factor();
            } else {
                // Does not require modulus switching
                if (u_prng == nullptr) {
                    rlwe::asymmetric(
                        this->public_key(), this->context(), parms_id, is_ntt_form, destination, pool
                    );
                } else {
                    rlwe::asymmetric_with_u_prng(
                        this->public_key(), this->context(), parms_id, is_ntt_form, *u_prng, destination, pool
                    );
                }
            }
        } else {
            // Does not require modulus switching
            if (u_prng == nullptr) {
                rlwe::symmetric(
                    this->secret_key(), this->context(), parms_id, is_ntt_form, save_seed, destination, pool
                );
            } else {
                rlwe::symmetric_with_c1_prng(
                    this->secret_key(), this->context(), parms_id, is_ntt_form, *u_prng, save_seed, destination, pool
                );
            }
        }
    }



    void Encryptor::encrypt_zero_internal_batched(
        const ParmsID& parms_id, 
        bool is_ntt_form,
        bool is_asymmetric, bool save_seed, 
        utils::RandomGenerator* u_prng, 
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    ) const {

        if (destination.size() == 0) return;
        if (destination.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < destination.size(); i++) {
                this->encrypt_zero_internal(parms_id, is_ntt_form, is_asymmetric, save_seed, u_prng, *destination[i], pool);
            }
            return;
        }

        // sanity check
        if (is_asymmetric && !this->public_key_.has_value()) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Public key not set for asymmetric encryption.");
        }
        if (!is_asymmetric && !this->secret_key_.has_value()) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Secret key not set for symmetric encryption.");
        }
        if (save_seed && is_asymmetric) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Cannot save seed when using asymmetric encryption.");
        }
        // verify params
        std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] parms_id is not valid for encryption parameters.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        
        // Resize destination and save results
        size_t n = destination.size();
        for (size_t i = 0; i < n; i++) {
            destination[i]->seed() = 0;
            bool device = (is_asymmetric) ? this->public_key_->on_device() : this->secret_key_->on_device();
            if (device) destination[i]->to_device_inplace(pool);
            else destination[i]->to_host_inplace();
            destination[i]->resize(this->context(), parms_id, 2, false, false);
        }
        
        if (is_asymmetric) {
            std::optional<std::weak_ptr<const ContextData>> prev_context_optional = context_data->prev_context_data();
            if (prev_context_optional.has_value()) {
                // Requires modulus switching
                ContextDataPointer prev_context_data = prev_context_optional.value().lock();
                ParmsID prev_parms_id = prev_context_data->parms_id();
                const RNSTool& rns_tool = prev_context_data->rns_tool();

                // Zero encryption without modulus switching
                std::vector<Ciphertext> temp(n); auto temp_ptrs = batch_utils::collect_pointer(temp);
                if (u_prng == nullptr) {
                    rlwe::asymmetric_batched(
                        this->public_key(), this->context(), prev_parms_id, is_ntt_form, temp_ptrs, pool
                    );
                } else {
                    rlwe::asymmetric_with_u_prng_batched(
                        this->public_key(), this->context(), prev_parms_id, is_ntt_form, *u_prng, temp_ptrs, pool
                    );
                }

                if (parms.scheme() == SchemeType::BGV && !is_ntt_form) {
                    throw std::invalid_argument("[Encryptor::encrypt_zero_internal] BGV - Plaintext is not in NTT form.");
                }
                
                // Modulus switching
                size_t polynomial_count = temp[0].polynomial_count();
                auto temp_const_reference = batch_utils::rcollect_const_reference(temp);
                auto destination_reference = batch_utils::pcollect_reference(destination);
                switch (parms.scheme()) {
                    case SchemeType::CKKS: case SchemeType::BFV: {
                        if (is_ntt_form) {
                            rns_tool.divide_and_round_q_last_ntt_batched(temp_const_reference, polynomial_count, destination_reference, prev_context_data->small_ntt_tables(), pool);
                        } else {
                            rns_tool.divide_and_round_q_last_batched(temp_const_reference, polynomial_count, destination_reference);
                        }
                        break;
                    }
                    case SchemeType::BGV: {
                        if (is_ntt_form) {
                            rns_tool.mod_t_and_divide_q_last_ntt_batched(temp_const_reference, polynomial_count, destination_reference, prev_context_data->small_ntt_tables(), pool);
                        }
                        break;
                    }
                    default:
                        throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Unsupported scheme.");
                }

                for (size_t i = 0; i < n; i++) {
                    destination[i]->parms_id() = parms_id;
                    destination[i]->is_ntt_form() = is_ntt_form;
                    destination[i]->scale() = temp[i].scale();
                    destination[i]->correction_factor() = temp[i].correction_factor();
                }
            } else {
                // Does not require modulus switching
                if (u_prng == nullptr) {
                    rlwe::asymmetric_batched(
                        this->public_key(), this->context(), parms_id, is_ntt_form, destination, pool
                    );
                } else {
                    rlwe::asymmetric_with_u_prng_batched(
                        this->public_key(), this->context(), parms_id, is_ntt_form, *u_prng, destination, pool
                    );
                }
            }
        } else {
            // Does not require modulus switching
            if (u_prng == nullptr) {
                rlwe::symmetric_batched(
                    this->secret_key(), this->context(), parms_id, is_ntt_form, save_seed, destination, pool
                );
            } else {
                rlwe::symmetric_with_c1_prng_batched(
                    this->secret_key(), this->context(), parms_id, is_ntt_form, *u_prng, save_seed, destination, pool
                );
            }
        }
    }



    void Encryptor::encrypt_internal(
        const Plaintext& plain,
        bool is_asymmetric, bool save_seed,
        utils::RandomGenerator* u_prng,
        Ciphertext& destination,
        MemoryPoolHandle pool
    ) const {
        if (!utils::same(plain.on_device(), this->on_device(), this->context()->on_device())) {
            throw std::invalid_argument("[Encryptor::encrypt_internal] The arguments are not on the same device.");
        }
        SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                if (plain.parms_id() == parms_id_zero) {
                    if (plain.is_ntt_form()) {
                        throw std::invalid_argument("[Encryptor::encrypt_internal] BFV - Plaintext is in NTT form.");
                    }
                    this->encrypt_zero_internal(this->context()->first_parms_id(), false, is_asymmetric, save_seed, u_prng, destination, pool);
                    // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
                    // Result gets added into the c_0 term of ciphertext (c_0,c_1).
                    scaling_variant::multiply_add_plain_inplace(plain, this->context()->first_context_data().value(), destination.poly(0), destination.poly_modulus_degree());
                } else {
                    ParmsID parms_id = plain.parms_id();
                    std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
                    if (!context_data_optional.has_value()) {
                        throw std::invalid_argument("[Encryptor::encrypt_internal] BFV - Plaintext parms_id is not valid.");
                    }
                    ContextDataPointer context_data = context_data_optional.value();
                    const EncryptionParameters& parms = context_data->parms();
                    this->encrypt_zero_internal(parms_id, plain.is_ntt_form(), is_asymmetric, save_seed, u_prng, destination, pool);
                    utils::add_partial_inplace_p(
                        destination.poly(0), plain.poly(), parms.poly_modulus_degree(), plain.coeff_count(), parms.coeff_modulus()
                    );
                }
                break;
            }
            case SchemeType::CKKS: {
                std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(plain.parms_id());
                if (!context_data_optional.has_value()) {
                    throw std::invalid_argument("[Encryptor::encrypt_internal] CKKS - Plaintext parms_id is not valid.");
                }
                ContextDataPointer context_data = context_data_optional.value();
                this->encrypt_zero_internal(plain.parms_id(), plain.is_ntt_form(), is_asymmetric, save_seed, u_prng, destination, pool);
                const EncryptionParameters& parms = context_data->parms();
                ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
                size_t coeff_count = parms.poly_modulus_degree();
                // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
                utils::add_inplace_p(
                    destination.poly(0), plain.poly(), coeff_count, coeff_modulus
                );
                destination.scale() = plain.scale();
                break; 
            }
            case SchemeType::BGV: {
                this->encrypt_zero_internal(this->context()->first_parms_id(), true, is_asymmetric, save_seed, u_prng, destination, pool);
                
                ContextDataPointer context_data = this->context()->first_context_data().value();
                const EncryptionParameters& parms = context_data->parms();
                ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
                size_t coeff_modulus_size = coeff_modulus.size();
                size_t coeff_count = parms.poly_modulus_degree();
                ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
                
                if (!plain.is_ntt_form()) {
                    // c_{0} = pk_{0}*u + p*e_{0} + M
                    Plaintext plain_copy = Plaintext::like(plain, false, pool);
                    // Resize to fit the entire NTT transformed (ciphertext size) polynomial
                    // Note that the new coefficients are automatically set to 0
                    plain_copy.resize(coeff_count * coeff_modulus_size);
                    plain_copy.data().reference().set_zero();

                    scaling_variant::centralize(plain, context_data, plain_copy.reference(), coeff_count, pool);

                    // Transform to NTT domain
                    utils::ntt_inplace_p(plain_copy.reference(), coeff_count, ntt_tables);

                    // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
                    utils::add_inplace_p(
                        destination.poly(0), plain_copy.const_poly(), coeff_count, coeff_modulus
                    );
                } else {
                    // directly add the plaintext into the c_0 term of ciphertext (c_0,c_1)
                    utils::add_inplace_p(
                        destination.poly(0), plain.const_poly(), coeff_count, coeff_modulus
                    );
                }
                break; // case SchemeType::BGV
            }
            default: 
                throw std::invalid_argument("[Encryptor::encrypt_internal] Unsupported scheme.");
        }
    }




    void Encryptor::encrypt_internal_batched(
        const std::vector<const Plaintext*>& plain,
        bool is_asymmetric, bool save_seed,
        utils::RandomGenerator* u_prng,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    ) const {
        if (plain.size() != destination.size()) {
            throw std::invalid_argument("[Encryptor::encrypt_internal_batched] The number of plaintexts and ciphertexts do not match.");
        }
        if (plain.size() == 0) return;
        if (plain.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < plain.size(); i++) {
                this->encrypt_internal(*plain[i], is_asymmetric, save_seed, u_prng, *destination[i], pool);
            }
            return;
        }
        SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
        ParmsID parms_id = plain[0]->parms_id();
        bool is_ntt_form = plain[0]->is_ntt_form();
        size_t plain_coeff_count = plain[0]->coeff_count();
        for (size_t i = 0; i < plain.size(); i++) {
            if (plain[i]->parms_id() != parms_id) {
                throw std::invalid_argument("[Encryptor::encrypt_internal_batched] BFV - Plaintext parms_id is not valid.");
            }
            if (plain[i]->is_ntt_form() != is_ntt_form) {
                throw std::invalid_argument("[Encryptor::encrypt_internal_batched] BFV - Plaintext is not in NTT form.");
            }
            if (plain[i]->coeff_count() != plain_coeff_count) {
                throw std::invalid_argument("[Encryptor::encrypt_internal_batched] BFV - Plaintexts have different coeff_count.");
            }
        }
        switch (scheme) {
            case SchemeType::BFV: {
                if (parms_id == parms_id_zero) {
                    if (is_ntt_form) {
                        throw std::invalid_argument("[Encryptor::encrypt_internal] BFV - Plaintext is in NTT form.");
                    }
                    this->encrypt_zero_internal_batched(this->context()->first_parms_id(), false, is_asymmetric, save_seed, u_prng, destination, pool);
                    // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
                    // Result gets added into the c_0 term of ciphertext (c_0,c_1).
                    scaling_variant::multiply_add_plain_inplace_batched(plain, this->context()->first_context_data().value(), batch_utils::pcollect_poly(destination, 0), destination[0]->poly_modulus_degree(), pool);
                } else {
                    std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
                    if (!context_data_optional.has_value()) {
                        throw std::invalid_argument("[Encryptor::encrypt_internal] BFV - Plaintext parms_id is not valid.");
                    }
                    ContextDataPointer context_data = context_data_optional.value();
                    const EncryptionParameters& parms = context_data->parms();
                    this->encrypt_zero_internal_batched(parms_id, is_ntt_form, is_asymmetric, save_seed, u_prng, destination, pool);
                    size_t plain_coeff_count = plain[0]->coeff_count();
                    utils::add_partial_inplace_bp(
                        batch_utils::pcollect_poly(destination, 0), batch_utils::pcollect_const_poly(plain), parms.poly_modulus_degree(), plain_coeff_count, parms.coeff_modulus(), pool
                    );
                }
                break;
            }
            case SchemeType::CKKS: {
                std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
                if (!context_data_optional.has_value()) {
                    throw std::invalid_argument("[Encryptor::encrypt_internal] CKKS - Plaintext parms_id is not valid.");
                }
                ContextDataPointer context_data = context_data_optional.value();
                this->encrypt_zero_internal_batched(parms_id, is_ntt_form, is_asymmetric, save_seed, u_prng, destination, pool);
                const EncryptionParameters& parms = context_data->parms();
                ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
                size_t coeff_count = parms.poly_modulus_degree();
                // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
                utils::add_inplace_bp(
                    batch_utils::pcollect_poly(destination, 0), batch_utils::pcollect_const_poly(plain), coeff_count, coeff_modulus, pool
                );
                for (size_t i = 0; i < plain.size(); i++) {
                    destination[i]->scale() = plain[i]->scale();
                }
                break; 
            }
            case SchemeType::BGV: {
                this->encrypt_zero_internal_batched(this->context()->first_parms_id(), true, is_asymmetric, save_seed, u_prng, destination, pool);
                
                ContextDataPointer context_data = this->context()->first_context_data().value();
                const EncryptionParameters& parms = context_data->parms();
                ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
                size_t coeff_modulus_size = coeff_modulus.size();
                size_t coeff_count = parms.poly_modulus_degree();
                ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
                
                if (!is_ntt_form) {
                    // c_{0} = pk_{0}*u + p*e_{0} + M
                    std::vector<Plaintext> plain_copy(plain.size());
                    for (size_t i = 0; i < plain.size(); i++) {
                        plain_copy[i] = Plaintext::like(*plain[i], false, pool);
                        // Resize to fit the entire NTT transformed (ciphertext size) polynomial
                        // Note that the new coefficients are automatically set to 0
                        plain_copy[i].resize(coeff_count * coeff_modulus_size);
                        plain_copy[i].data().reference().set_zero();
                    }

                    auto plain_copy_reference = batch_utils::rcollect_reference(plain_copy);
                    scaling_variant::centralize_batched(plain, context_data, plain_copy_reference, coeff_count, pool);

                    // Transform to NTT domain
                    utils::ntt_inplace_bp(plain_copy_reference, coeff_count, ntt_tables, pool);

                    // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
                    utils::add_inplace_bp(
                        batch_utils::pcollect_poly(destination, 0), batch_utils::rcollect_const_reference(plain_copy), coeff_count, coeff_modulus, pool
                    );
                } else {
                    // directly add the plaintext into the c_0 term of ciphertext (c_0,c_1)
                    utils::add_inplace_bp(
                        batch_utils::pcollect_poly(destination, 0), batch_utils::pcollect_const_reference(plain), coeff_count, coeff_modulus, pool
                    );
                }
                break; // case SchemeType::BGV
            }
            default: 
                throw std::invalid_argument("[Encryptor::encrypt_internal] Unsupported scheme.");
        }
    }


}