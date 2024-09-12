#include "batch_utils.h"
#include "ciphertext.h"
#include "encryption_parameters.h"
#include "evaluator.h"
#include "utils/dynamic_array.h"
#include "utils/memory_pool.h"
#include "utils/ntt.h"
#include "utils/polynomial_buffer.h"
#include "fgk/dyadic_convolute.h"
#include "fgk/translate_plain.h"
#include "evaluator_utils.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::NTTTables;
    using utils::RNSTool;
    using utils::Buffer;

    ContextDataPointer Evaluator::get_context_data(const char* prompt, const ParmsID& encrypted) const {
        auto context_data_ptr = context_->get_context_data(encrypted);
        if (!context_data_ptr.has_value()) {
            throw std::invalid_argument(std::string(prompt) + " Context data not found parms id.");
        }
        return context_data_ptr.value();
    }

    void Evaluator::bfv_multiply(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_is_not_ntt_form("[Evaluator::bfv_multiply_inplace]", encrypted1);
        check_is_not_ntt_form("[Evaluator::bfv_multiply_inplace]", encrypted2);

        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::bfv_multiply_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> base_q = parms.coeff_modulus();
        size_t base_q_size = base_q.size();
        size_t encrypted1_size = encrypted1.polynomial_count();
        size_t encrypted2_size = encrypted2.polynomial_count();
        const RNSTool& rns_tool = context_data->rns_tool();
        ConstSlice<Modulus> base_Bsk = rns_tool.base_Bsk().base();
        size_t base_Bsk_size = base_Bsk.size();
        
        // Determine destination.size()
        size_t dest_size = encrypted1_size + encrypted2_size - 1;
        ConstSlice<NTTTables> base_q_ntt_tables = context_data->small_ntt_tables();
        ConstSlice<NTTTables> base_Bsk_ntt_tables = rns_tool.base_Bsk_ntt_tables();
        
        // Microsoft SEAL uses BEHZ-style RNS multiplication. This process is somewhat complex and consists of the
        // following steps:
        //
        // (1) Lift encrypted1 and encrypted2 (initially in base q) to an extended base q U Bsk U {m_tilde}
        // (2) Remove extra multiples of q from the results with Montgomery reduction, switching base to q U Bsk
        // (3) Transform the data to NTT form
        // (4) Compute the ciphertext polynomial product using dyadic multiplication
        // (5) Transform the data back from NTT form
        // (6) Multiply the result by t (plain_modulus)
        // (7) Scale the result by q using a divide-and-floor algorithm, switching base to Bsk
        // (8) Use Shenoy-Kumaresan method to convert the result to base q

        bool device = encrypted1.on_device();
        destination = Ciphertext::like(encrypted1, dest_size, false, pool);
        // Allocate space for a base q output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted1_q(encrypted1_size, base_q_size, coeff_count, device, pool);
        // Allocate space for a base Bsk output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted1_Bsk(encrypted1_size, base_Bsk_size, coeff_count, device, pool);

        // Perform BEHZ steps (1)-(3) for encrypted1
        // Make copy of input polynomial (in base q) and convert to NTT form
        utils::ntt_ps(encrypted1.const_reference(), encrypted1_size, coeff_count, base_q_ntt_tables, encrypted1_q.reference());
        // Allocate temporary space for a polynomial in the Bsk U {m_tilde} base
        for (size_t i = 0; i < encrypted1_size; i++) {
            // (1) Convert from base q to base Bsk U {m_tilde}
            rns_tool.fast_b_conv_m_tilde_sm_mrq(encrypted1.const_poly(i), encrypted1_Bsk.poly(i), pool);
        }
        // Transform to NTT form in base Bsk
        utils::ntt_inplace_ps(encrypted1_Bsk.reference(), encrypted1_size, coeff_count, base_Bsk_ntt_tables);

        // Repeat for encrypted2
        Buffer<uint64_t> encrypted2_q(encrypted2_size, base_q_size, coeff_count, device, pool);
        Buffer<uint64_t> encrypted2_Bsk(encrypted2_size, base_Bsk_size, coeff_count, device, pool);
        utils::ntt_ps(encrypted2.const_reference(), encrypted2_size, coeff_count, base_q_ntt_tables, encrypted2_q.reference());
        for (size_t i = 0; i < encrypted2_size; i++) {
            rns_tool.fast_b_conv_m_tilde_sm_mrq(encrypted2.poly(i), encrypted2_Bsk.poly(i), pool);
        }
        utils::ntt_inplace_ps(encrypted2_Bsk.reference(), encrypted2_size, coeff_count, base_Bsk_ntt_tables);

        // Allocate temporary space for the output of step (4)
        // We allocate space separately for the base q and the base Bsk components
        Buffer<uint64_t> temp_dest_q(dest_size, base_q_size, coeff_count, device, pool);
        Buffer<uint64_t> temp_dest_Bsk(dest_size, base_Bsk_size, coeff_count, device, pool);

        // Perform BEHZ step (4): dyadic multiplication on arbitrary size ciphertexts
        utils::fgk::dyadic_convolute::dyadic_convolute(
            encrypted1_q.const_reference(), encrypted2_q.const_reference(),
            encrypted1_size, encrypted2_size, base_q, coeff_count,
            temp_dest_q.reference(), pool
        );
        utils::fgk::dyadic_convolute::dyadic_convolute(
            encrypted1_Bsk.const_reference(), encrypted2_Bsk.const_reference(),
            encrypted1_size, encrypted2_size, base_Bsk, coeff_count,
            temp_dest_Bsk.reference(), pool
        );

        // Perform BEHZ step (5): transform data from NTT form
        // Lazy reduction here. The following multiplyPolyScalarCoeffmod will correct the value back to [0, p)
        utils::intt_inplace_ps(temp_dest_q.reference(), dest_size, coeff_count, base_q_ntt_tables);
        utils::intt_inplace_ps(temp_dest_Bsk.reference(), dest_size, coeff_count, base_Bsk_ntt_tables);

        // Perform BEHZ steps (6)-(8)
        rns_tool.fast_floor_fast_b_conv_sk(
            temp_dest_q.const_reference(), temp_dest_Bsk.const_reference(),
            destination.reference(), pool
        );
    }

    void Evaluator::ckks_multiply(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_is_ntt_form("[Evaluator::ckks_multiply_inplace]", encrypted1);
        check_is_ntt_form("[Evaluator::ckks_multiply_inplace]", encrypted2);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::ckks_multiply_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t encrypted1_size = encrypted1.polynomial_count();
        size_t encrypted2_size = encrypted2.polynomial_count();
        
        // Determine destination.size()
        size_t dest_size = encrypted1_size + encrypted2_size - 1;
        destination = Ciphertext::like(encrypted1, dest_size, false, pool);

        utils::fgk::dyadic_convolute::dyadic_convolute(
            encrypted1.const_reference(), encrypted2.const_reference(),
            encrypted1_size, encrypted2_size, coeff_modulus, coeff_count,
            destination.data().reference(), pool
        );

        destination.scale() = encrypted1.scale() * encrypted2.scale();
        if (!is_scale_within_bounds(destination.scale(), context_data)) {
            throw std::invalid_argument("[Evaluator::ckks_multiply] Scale out of bounds");
        }
    }
    
    void Evaluator::bgv_multiply(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_is_ntt_form("[Evaluator::bgv_multiply]", encrypted1);
        check_is_ntt_form("[Evaluator::bgv_multiply]", encrypted2);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::bgv_multiply_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t encrypted1_size = encrypted1.polynomial_count();
        size_t encrypted2_size = encrypted2.polynomial_count();
        
        // Determine destination.size()
        size_t dest_size = encrypted1_size + encrypted2_size - 1;
        destination = Ciphertext::like(encrypted1, dest_size, false, pool);

        utils::fgk::dyadic_convolute::dyadic_convolute(
            encrypted1.const_reference(), encrypted2.const_reference(),
            encrypted1_size, encrypted2_size, coeff_modulus, coeff_count,
            destination.data().reference(), pool
        );

        destination.correction_factor() = utils::multiply_uint64_mod(
            encrypted1.correction_factor(),
            encrypted2.correction_factor(),
            parms.plain_modulus_host()
        );
    }
    
    void Evaluator::multiply(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::multiply]", encrypted1);
        check_no_seed("[Evaluator::multiply]", encrypted2);
        check_same_parms_id("[Evaluator::multiply]", encrypted1, encrypted2);
        SchemeType scheme = this->context()->key_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                this->bfv_multiply(encrypted1, encrypted2, destination, pool);
                break;
            }
            case SchemeType::CKKS: {
                this->ckks_multiply(encrypted1, encrypted2, destination, pool);
                break;
            }
            case SchemeType::BGV: {
                this->bgv_multiply(encrypted1, encrypted2, destination, pool);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::multiply] Scheme not implemented.");
            }
        }
    }

    void Evaluator::bfv_square(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_is_not_ntt_form("[Evaluator::bfv_square_inplace]", encrypted);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::bfv_square_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> base_q = parms.coeff_modulus();
        size_t base_q_size = base_q.size();
        size_t encrypted_size = encrypted.polynomial_count();

        if (encrypted_size != 2) {
            this->multiply(encrypted, encrypted, destination, pool);
            return;
        }
        
        const RNSTool& rns_tool = context_data->rns_tool();
        ConstSlice<Modulus> base_Bsk = rns_tool.base_Bsk().base();
        size_t base_Bsk_size = base_Bsk.size();
        ConstSlice<Modulus> base_Bsk_m_tilde = rns_tool.base_Bsk_m_tilde().base();
        size_t base_Bsk_m_tilde_size = base_Bsk_m_tilde.size();
        
        // Determine destination.size()
        size_t dest_size = 2 * encrypted_size - 1;
        ConstSlice<NTTTables> base_q_ntt_tables = context_data->small_ntt_tables();
        ConstSlice<NTTTables> base_Bsk_ntt_tables = rns_tool.base_Bsk_ntt_tables();
        
        // Microsoft SEAL uses BEHZ-style RNS multiplication. This process is somewhat complex and consists of the
        // following steps:
        //
        // (1) Lift encrypted1 and encrypted2 (initially in base q) to an extended base q U Bsk U {m_tilde}
        // (2) Remove extra multiples of q from the results with Montgomery reduction, switching base to q U Bsk
        // (3) Transform the data to NTT form
        // (4) Compute the ciphertext polynomial product using dyadic multiplication
        // (5) Transform the data back from NTT form
        // (6) Multiply the result by t (plain_modulus)
        // (7) Scale the result by q using a divide-and-floor algorithm, switching base to Bsk
        // (8) Use Shenoy-Kumaresan method to convert the result to base q

        bool device = encrypted.on_device();
        destination = Ciphertext::like(encrypted, dest_size, false, pool);
        // Allocate space for a base q output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted_q(encrypted_size, base_q_size, coeff_count, device, pool);
        // Allocate space for a base Bsk output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted_Bsk(encrypted_size, base_Bsk_size, coeff_count, device, pool);

        // Perform BEHZ steps (1)-(3) for encrypted1
        // Make copy of input polynomial (in base q) and convert to NTT form
        utils::ntt_ps(encrypted.const_reference(), encrypted_size, coeff_count, base_q_ntt_tables, encrypted_q.reference());
        // Allocate temporary space for a polynomial in the Bsk U {m_tilde} base
        Buffer<uint64_t> temp(base_Bsk_m_tilde_size, coeff_count, device, pool);
        for (size_t i = 0; i < encrypted_size; i++) {
            // (1) Convert from base q to base Bsk U {m_tilde}
            rns_tool.fast_b_conv_m_tilde_sm_mrq(encrypted.const_poly(i), encrypted_Bsk.poly(i), pool);
        }
        // Transform to NTT form in base Bsk
        utils::ntt_inplace_ps(encrypted_Bsk.reference(), encrypted_size, coeff_count, base_Bsk_ntt_tables);

        // Allocate temporary space for the output of step (4)
        // We allocate space separately for the base q and the base Bsk components
        Buffer<uint64_t> temp_dest_q(dest_size, base_q_size, coeff_count, device, pool);
        Buffer<uint64_t> temp_dest_Bsk(dest_size, base_Bsk_size, coeff_count, device, pool);

        // Perform the BEHZ ciphertext square both for base q and base Bsk
        utils::fgk::dyadic_convolute::dyadic_square(
            encrypted_q.const_reference(), base_q, coeff_count,
            temp_dest_q.reference()
        );
        utils::fgk::dyadic_convolute::dyadic_square(
            encrypted_Bsk.const_reference(), base_Bsk, coeff_count,
            temp_dest_Bsk.reference()
        );
        
        // Perform BEHZ step (5): transform data from NTT form
        // Lazy reduction here. The following multiplyPolyScalarCoeffmod will correct the value back to [0, p)
        utils::intt_inplace_ps(temp_dest_q.reference(), dest_size, coeff_count, base_q_ntt_tables);
        utils::intt_inplace_ps(temp_dest_Bsk.reference(), dest_size, coeff_count, base_Bsk_ntt_tables);

        // Perform BEHZ steps (6)-(8)
        rns_tool.fast_floor_fast_b_conv_sk(
            temp_dest_q.const_reference(), temp_dest_Bsk.const_reference(),
            destination.reference(), pool
        );
    }

    void Evaluator::ckks_square(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_is_ntt_form("[Evaluator::ckks_square_inplace]", encrypted);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::ckks_square_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t encrypted_size = encrypted.polynomial_count();

        if (encrypted_size != 2) {
            this->multiply(encrypted, encrypted, destination, pool);
            return;
        }
        
        // Determine destination.size()
        size_t dest_size = 2 * encrypted_size - 1;
        destination = Ciphertext::like(encrypted, dest_size, false, pool);
        
        utils::fgk::dyadic_convolute::dyadic_square(
            encrypted.const_reference(), coeff_modulus, coeff_count,
            destination.data().reference()
        );

        destination.scale() = encrypted.scale() * encrypted.scale();
        if (!is_scale_within_bounds(destination.scale(), context_data)) {
            throw std::invalid_argument("[Evaluator::ckks_multiply_inplace] Scale out of bounds");
        }
    }
    
    void Evaluator::bgv_square(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_is_ntt_form("[Evaluator::bgv_square_inplace]", encrypted);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::bgv_square_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t encrypted_size = encrypted.polynomial_count();

        if (encrypted_size != 2) {
            this->multiply(encrypted, encrypted, destination, pool);
            return;
        }
        
        // Determine destination.size()
        size_t dest_size = 2 * encrypted_size - 1;
        destination = Ciphertext::like(encrypted, dest_size, false, pool);

        utils::fgk::dyadic_convolute::dyadic_square(
            encrypted.const_reference(), coeff_modulus, coeff_count,
            destination.data().reference()
        );

        destination.correction_factor() = utils::multiply_uint64_mod(
            encrypted.correction_factor(),
            encrypted.correction_factor(),
            parms.plain_modulus_host()
        );
    }

    void Evaluator::square(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::square_inplace]", encrypted);
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                this->bfv_square(encrypted, destination, pool);
                break;
            }
            case SchemeType::CKKS: {
                this->ckks_square(encrypted, destination, pool);
                break;
            }
            case SchemeType::BGV: {
                this->bgv_square(encrypted, destination, pool);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::square_inplace] Scheme not implemented.");
            }
        }
    }

}