#include "encryptor.cuh"
#include "utils/scaling_variant.cuh"

namespace troy {

    using utils::ConstSlice;
    using utils::RNSTool;

    void Encryptor::encrypt_zero_internal(
        const ParmsID& parms_id, 
        bool is_asymmetric, bool save_seed, 
        utils::RandomGenerator* u_prng, 
        Ciphertext& destination
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
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t poly_element_count = coeff_count * coeff_modulus_size;
        bool is_ntt_form = false;
        if (context_data->is_ckks() || context_data->is_bgv()) {
            is_ntt_form = true;
        }
        
        // Resize destination and save results
        destination.seed() = 0;
        bool device = (is_asymmetric) ? this->public_key_->on_device() : this->secret_key_->on_device();
        if (device) destination.to_device_inplace();
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
                        this->public_key(), this->context(), prev_parms_id, is_ntt_form, temp
                    );
                } else {
                    rlwe::asymmetric_with_u_prng(
                        this->public_key(), this->context(), prev_parms_id, is_ntt_form, *u_prng, temp
                    );
                }
                
                // Modulus switching
                for (size_t i = 0; i < temp.polynomial_count(); i++) {
                    switch (parms.scheme()) {
                        case SchemeType::CKKS: {
                            rns_tool.divide_and_round_q_last_ntt_inplace(temp.poly(i), prev_context_data->small_ntt_tables());
                            break;
                        }
                        case SchemeType::BFV: {
                            rns_tool.divide_and_round_q_last_inplace(temp.poly(i));
                            break;
                        }
                        case SchemeType::BGV: {
                            rns_tool.mod_t_and_divide_q_last_ntt_inplace(temp.poly(i), prev_context_data->small_ntt_tables());
                            break;
                        }
                        default:
                            throw std::invalid_argument("[Encryptor::encrypt_zero_internal] Unsupported scheme.");
                    }
                    destination.poly(i).copy_from_slice(
                        temp.poly(i).const_slice(0, poly_element_count)
                    );
                }

                destination.parms_id() = parms_id;
                destination.is_ntt_form() = is_ntt_form;
                destination.scale() = temp.scale();
                destination.correction_factor() = temp.correction_factor();
            } else {
                // Does not require modulus switching
                if (u_prng == nullptr) {
                    rlwe::asymmetric(
                        this->public_key(), this->context(), parms_id, is_ntt_form, destination
                    );
                } else {
                    rlwe::asymmetric_with_u_prng(
                        this->public_key(), this->context(), parms_id, is_ntt_form, *u_prng, destination
                    );
                }
            }
        } else {
            // Does not require modulus switching
            if (u_prng == nullptr) {
                rlwe::symmetric(
                    this->secret_key(), this->context(), parms_id, is_ntt_form, save_seed, destination
                );
            } else {
                rlwe::symmetric_with_c1_prng(
                    this->secret_key(), this->context(), parms_id, is_ntt_form, *u_prng, save_seed, destination
                );
            }
        }
    }

    void Encryptor::encrypt_internal(
        const Plaintext& plain,
        bool is_asymmetric, bool save_seed,
        utils::RandomGenerator* u_prng,
        Ciphertext& destination
    ) const {
        if (!utils::same(plain.on_device(), this->on_device(), this->context()->on_device())) {
            throw std::invalid_argument("[Encryptor::encrypt_internal] The arguments are not on the same device.");
        }
        SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                if (plain.is_ntt_form()) {
                    throw std::invalid_argument("[Encryptor::encrypt_internal] BFV - Plaintext is in NTT form.");
                }
                if (plain.parms_id() == parms_id_zero) {
                    this->encrypt_zero_internal(this->context()->first_parms_id(), is_asymmetric, save_seed, u_prng, destination);
                    // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
                    // Result gets added into the c_0 term of ciphertext (c_0,c_1).
                    scaling_variant::multiply_add_plain(plain, this->context()->first_context_data().value(), destination.poly(0));
                } else {
                    ParmsID parms_id = plain.parms_id();
                    std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(parms_id);
                    if (!context_data_optional.has_value()) {
                        throw std::invalid_argument("[Encryptor::encrypt_internal] BFV - Plaintext parms_id is not valid.");
                    }
                    ContextDataPointer context_data = context_data_optional.value();
                    const EncryptionParameters& parms = context_data->parms();
                    this->encrypt_zero_internal(parms_id, is_asymmetric, save_seed, u_prng, destination);
                    utils::add_inplace_p(
                        destination.poly(0), plain.poly(), parms.poly_modulus_degree(), parms.coeff_modulus()
                    );
                }
                break;
            }
            case SchemeType::CKKS: {
                if (!plain.is_ntt_form()) {
                    throw std::invalid_argument("[Encryptor::encrypt_internal] CKKS - Plaintext is not in NTT form.");
                }
                std::optional<ContextDataPointer> context_data_optional = this->context()->get_context_data(plain.parms_id());
                if (!context_data_optional.has_value()) {
                    throw std::invalid_argument("[Encryptor::encrypt_internal] CKKS - Plaintext parms_id is not valid.");
                }
                ContextDataPointer context_data = context_data_optional.value();
                this->encrypt_zero_internal(plain.parms_id(), is_asymmetric, save_seed, u_prng, destination);
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
                if (plain.is_ntt_form()) {
                    throw std::invalid_argument("[Encryptor::encrypt_internal] BGV - Plaintext is in NTT form.");
                }
                this->encrypt_zero_internal(this->context()->first_parms_id(), is_asymmetric, save_seed, u_prng, destination);
                
                ContextDataPointer context_data = this->context()->first_context_data().value();
                const EncryptionParameters& parms = context_data->parms();
                ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
                size_t coeff_modulus_size = coeff_modulus.size();
                size_t coeff_count = parms.poly_modulus_degree();
                size_t plain_coeff_count = plain.coeff_count();
                uint64_t plain_upper_half_threshold = context_data->plain_upper_half_threshold();
                ConstSlice<uint64_t> plain_upper_half_increment = context_data->plain_upper_half_increment();
                ConstSlice<utils::NTTTables> ntt_tables = context_data->small_ntt_tables();
                
                // c_{0} = pk_{0}*u + p*e_{0} + M
                Plaintext plain_copy = plain.clone();
                // Resize to fit the entire NTT transformed (ciphertext size) polynomial
                // Note that the new coefficients are automatically set to 0
                plain_copy.resize(coeff_count * coeff_modulus_size);
                plain_copy.data().reference().set_zero();

                scaling_variant::centralize(plain, context_data, plain_copy.reference());

                // Transform to NTT domain
                utils::ntt_negacyclic_harvey_p(plain_copy.reference(), coeff_count, ntt_tables);

                // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
                utils::add_inplace_p(
                    destination.poly(0), plain_copy.const_poly(), coeff_count, coeff_modulus
                );
                break; // case SchemeType::BGV
            }
            default: 
                throw std::invalid_argument("[Encryptor::encrypt_internal] Unsupported scheme.");
        }
    }

}