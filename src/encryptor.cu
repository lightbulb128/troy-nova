#include "encryption_parameters.h"
#include "encryptor.h"
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
        destination.resize(this->context(), parms_id, 2);
        
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

}