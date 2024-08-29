#include "evaluator.h"
#include "evaluator_utils.h"
#include "utils/polynomial_buffer.h"
#include "fgk/translate_plain.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::ConstPointer;
    using utils::Buffer;

    void Evaluator::translate_plain_inplace(Ciphertext& encrypted, const Plaintext& plain, bool subtract, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::translate_plain_inplace]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::translate_plain_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                if (encrypted.is_ntt_form() != plain.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain_inplace] Plaintext and ciphertext are not in the same NTT form.");
                }
                if (plain.parms_id() == parms_id_zero && encrypted.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain_inplace] When plain is mod t, encrypted must not be in NTT form.");
                }
                break;
            }
            case SchemeType::CKKS: {
                check_is_ntt_form("[Evaluator::translate_plain_inplace]", encrypted);
                if (!utils::are_close_double(plain.scale(), encrypted.scale())) {
                    throw std::invalid_argument("[Evaluator::translate_plain_inplace] Plaintext scale is not equal to the scale of the ciphertext.");
                }
                if (encrypted.is_ntt_form() != plain.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain_inplace] Plaintext and ciphertext are not in the same NTT form.");
                }
                break;
            }
            case SchemeType::BGV: {
                check_is_ntt_form("[Evaluator::translate_plain_inplace]", encrypted);
                if (plain.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain_inplace] Plaintext is in NTT form.");
                }
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::translate_plain_inplace] Scheme not implemented.");
            }
        }
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        switch (scheme) {
            case SchemeType::BFV: {
                if (plain.parms_id() == parms_id_zero) {
                    if (!subtract) {
                        scaling_variant::multiply_add_plain_inplace(plain, context_data, encrypted.poly(0), coeff_count);
                    } else {
                        scaling_variant::multiply_sub_plain_inplace(plain, context_data, encrypted.poly(0), coeff_count);
                    }
                } else {
                    if (plain.parms_id() != encrypted.parms_id()) {
                        throw std::invalid_argument("[Evaluator::translate_plain_inplace] Plaintext and ciphertext parameters do not match.");
                    }
                    if (!subtract) {
                        utils::add_partial_inplace_p(encrypted.poly(0), plain.poly(), coeff_count, plain.coeff_count(), coeff_modulus);
                    } else {
                        utils::sub_partial_inplace_p(encrypted.poly(0), plain.poly(), coeff_count, plain.coeff_count(), coeff_modulus);
                    }
                }
                break;
            }
            case SchemeType::CKKS: {
                if (!subtract) {
                    utils::add_inplace_p(encrypted.poly(0), plain.poly(), coeff_count, coeff_modulus);
                } else {
                    utils::sub_inplace_p(encrypted.poly(0), plain.poly(), coeff_count, coeff_modulus);
                }
                break;
            }
            case SchemeType::BGV: {
                Plaintext plain_copy = Plaintext::like(plain, false, pool);
                utils::multiply_scalar(plain.poly(), encrypted.correction_factor(), parms.plain_modulus(), plain_copy.poly());
                this->transform_plain_to_ntt_inplace(plain_copy, encrypted.parms_id(), pool);
                if (!subtract) {
                    utils::add_inplace_p(encrypted.poly(0), plain_copy.const_poly(), coeff_count, coeff_modulus);
                } else {
                    utils::sub_inplace_p(encrypted.poly(0), plain_copy.const_poly(), coeff_count, coeff_modulus);
                }
                break;
            }
            default: 
                throw std::logic_error("[Evaluator::translate_plain_inplace] Scheme not implemented.");
        }
    }
    
    void Evaluator::translate_plain(const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, bool subtract, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::translate_plain]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::translate_plain_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                if (encrypted.is_ntt_form() != plain.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain] Plaintext and ciphertext are not in the same NTT form.");
                }
                if (plain.parms_id() == parms_id_zero && encrypted.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain] When plain is mod t, encrypted must not be in NTT form.");
                }
                break;
            }
            case SchemeType::CKKS: {
                check_is_ntt_form("[Evaluator::translate_plain_inplace]", encrypted);
                if (!utils::are_close_double(plain.scale(), encrypted.scale())) {
                    throw std::invalid_argument("[Evaluator::translate_plain] Plaintext scale is not equal to the scale of the ciphertext.");
                }
                if (encrypted.is_ntt_form() != plain.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain] Plaintext and ciphertext are not in the same NTT form.");
                }
                break;
            }
            case SchemeType::BGV: {
                check_is_ntt_form("[Evaluator::translate_plain]", encrypted);
                if (plain.is_ntt_form()) {
                    throw std::invalid_argument("[Evaluator::translate_plain] Plaintext is in NTT form.");
                }
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::translate_plain] Scheme not implemented.");
            }
        }
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        destination = Ciphertext::like(encrypted, false, pool);
        switch (scheme) {
            case SchemeType::BFV: {
                if (plain.parms_id() == parms_id_zero) {
                    utils::fgk::translate_plain::multiply_translate_plain(encrypted.const_reference(), plain, context_data, destination.reference(), coeff_count, subtract);
                } else {
                    if (plain.parms_id() != encrypted.parms_id()) {
                        throw std::invalid_argument("[Evaluator::translate_plain_inplace] Plaintext and ciphertext parameters do not match.");
                    }
                    utils::fgk::translate_plain::scatter_translate_copy(encrypted.const_reference(), plain.poly(), coeff_count, plain.coeff_count(), coeff_modulus, destination.reference(), subtract);
                }
                break;
            }
            case SchemeType::CKKS: {
                utils::fgk::translate_plain::translate_copy(encrypted.const_reference(), plain.const_poly(), coeff_count, coeff_modulus, destination.reference(), subtract);
                break;
            }
            case SchemeType::BGV: {
                Plaintext plain_copy = Plaintext::like(plain, false, pool);
                utils::multiply_scalar(plain.poly(), encrypted.correction_factor(), parms.plain_modulus(), plain_copy.poly());
                this->transform_plain_to_ntt_inplace(plain_copy, encrypted.parms_id(), pool);
                utils::fgk::translate_plain::translate_copy(encrypted.const_reference(), plain_copy.const_poly(), coeff_count, coeff_modulus, destination.reference(), subtract);
                break;
            }
            default: 
                throw std::logic_error("[Evaluator::translate_plain_inplace] Scheme not implemented.");
        }
    }

}