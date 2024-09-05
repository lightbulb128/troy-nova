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
    using utils::GaloisTool;

    ContextDataPointer Evaluator::get_context_data(const char* prompt, const ParmsID& encrypted) const {
        auto context_data_ptr = context_->get_context_data(encrypted);
        if (!context_data_ptr.has_value()) {
            throw std::invalid_argument(std::string(prompt) + " Context data not found parms id.");
        }
        return context_data_ptr.value();
    }

    void Evaluator::negate_inplace(Ciphertext& encrypted) const {
        check_ciphertext("[Evaluator::negate_inplace]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t poly_count = encrypted.polynomial_count();
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_inplace_ps(encrypted.data().reference(), poly_count, poly_degree, coeff_modulus);
    }

    void Evaluator::negate(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_ciphertext("[Evaluator::negate]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        destination = Ciphertext::like(encrypted, false, pool);
        size_t poly_count = encrypted.polynomial_count();
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_ps(encrypted.data().const_reference(), poly_count, poly_degree, coeff_modulus, destination.data().reference());
    }

    
    void Evaluator::negate_inplace_batched(const std::vector<Ciphertext*>& encrypted, MemoryPoolHandle pool) const {
        if (encrypted.size() == 0) return;
        check_ciphertext_vec("[Evaluator::negate_inplace_batched]", encrypted);
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate_inplace_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t poly_count = get_vec_polynomial_count(encrypted);
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_inplace_bps(batch_utils::pcollect_reference(encrypted), poly_count, poly_degree, coeff_modulus, pool);
    }

    void Evaluator::negate_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::negate_batched] Size mismatch");
        }
        if (encrypted.size() == 0) return;
        check_ciphertext_vec("[Evaluator::negate_batched]", encrypted);
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        for (size_t i = 0; i < encrypted.size(); i++) *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        size_t poly_count = get_vec_polynomial_count(encrypted);
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_bps(batch_utils::pcollect_const_reference(encrypted), poly_count, poly_degree, coeff_modulus, batch_utils::pcollect_reference(destination));
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

    void Evaluator::apply_keyswitching(const Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        if (kswitch_keys.data().size() != 1) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys size must be 1.");
        }
        if (encrypted.polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Ciphertext polynomial count must be 2.");
        }
        if (kswitch_keys.data()[0][0].as_ciphertext().polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys polynomial count must be 2. Check the key switch key generation for problems.");
        }
        destination = Ciphertext::like(encrypted, false, pool);
        this->switch_key_internal(encrypted, encrypted.poly(1), kswitch_keys, 0, Evaluator::SwitchKeyDestinationAssignMethod::Overwrite, destination, pool);
        
        ContextDataPointer context_data = this->get_context_data("[Evaluator::switch_key_inplace_internal]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();

        utils::add_inplace_p(destination.poly(0), encrypted.poly(0), parms.poly_modulus_degree(), parms.coeff_modulus());
    }

    void Evaluator::apply_keyswitching_inplace(Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, MemoryPoolHandle pool) const {
        if (kswitch_keys.data().size() != 1) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys size must be 1.");
        }
        if (encrypted.polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Ciphertext polynomial count must be 2.");
        }
        if (kswitch_keys.data()[0][0].as_ciphertext().polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys polynomial count must be 2. Check the key switch key generation for problems.");
        }
        this->switch_key_internal(encrypted, encrypted.poly(1), kswitch_keys, 0, Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst, encrypted, pool);
        
    }

    void Evaluator::relinearize_inplace_internal(Ciphertext& encrypted, const RelinKeys& relin_keys, size_t destination_size, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::relinearize_inplace_internal]", encrypted);
        if (relin_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Relin keys has incorrect parms id.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::relinearize_inplace_internal]", encrypted.parms_id());
        size_t encrypted_size = encrypted.polynomial_count();
        if (encrypted_size < 2 || destination_size > encrypted_size) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Destination size must be at least 2 and less/equal to the size of the encrypted polynomial.");
        }
        if (destination_size == encrypted_size) {
            return;
        }
        size_t relins_needed = encrypted_size - destination_size;
        for (size_t i = 0; i < relins_needed; i++) {
            this->switch_key_internal(
                encrypted, encrypted.const_poly(encrypted_size - 1),
                relin_keys.as_kswitch_keys(), RelinKeys::get_index(encrypted_size - 1), Evaluator::SwitchKeyDestinationAssignMethod::AddInplace, encrypted, pool);
            encrypted_size -= 1;
        }
        encrypted.resize(this->context(), context_data->parms_id(), destination_size);
    }

    void Evaluator::relinearize_internal(const Ciphertext& encrypted, const RelinKeys& relin_keys, size_t destination_size, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::relinearize_inplace_internal]", encrypted);
        if (relin_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Relin keys has incorrect parms id.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::relinearize_inplace_internal]", encrypted.parms_id());
        size_t encrypted_size = encrypted.polynomial_count();
        if (encrypted_size < 2 || destination_size > encrypted_size) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Destination size must be at least 2 and less/equal to the size of the encrypted polynomial.");
        }
        if (destination_size == encrypted_size) {
            return;
        }
        size_t relins_needed = encrypted_size - destination_size;
        destination = Ciphertext::like(encrypted, destination_size, false, pool);
        for (size_t i = 0; i < relins_needed; i++) {
            this->switch_key_internal(
                encrypted, encrypted.const_poly(encrypted_size - 1),
                relin_keys.as_kswitch_keys(), RelinKeys::get_index(encrypted_size - 1), 
                i == 0 ? Evaluator::SwitchKeyDestinationAssignMethod::Overwrite : Evaluator::SwitchKeyDestinationAssignMethod::AddInplace, 
                destination, pool);
            encrypted_size -= 1;
        }
        const EncryptionParameters& parms = context_data->parms();
        utils::add_inplace_ps(destination.polys(0, destination_size), encrypted.const_polys(0, destination_size), destination_size, parms.poly_modulus_degree(), parms.coeff_modulus());
    }

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

    __global__ static void kernel_mod_switch_drop_to(ConstSlice<uint64_t> source, size_t poly_count, size_t source_modulus_size, size_t remain_modulus_size, size_t degree, Slice<uint64_t> destination) {
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

    
    void Evaluator::apply_galois(const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::apply_galois_inplace]", encrypted);
        if (galois_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois keys has incorrect parms id.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::apply_galois_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.polynomial_count();
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const GaloisTool& galois_tool = key_context_data->galois_tool();

        if (!galois_keys.has_key(galois_element)) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois key not present.");
        }
        size_t m = coeff_count * 2;
        if ((galois_element & 1) == 0 || galois_element > m) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois element is not valid.");
        }
        if (encrypted_size > 2) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Ciphertext size must be 2.");
        }

        destination = Ciphertext::like(encrypted, false, pool);
        if (!encrypted.is_ntt_form()) {
            galois_tool.apply_ps(encrypted.const_polys(0, 1), 2, galois_element, coeff_modulus, destination.polys(0, 1));
        } else {
            galois_tool.apply_ntt_ps(encrypted.const_polys(0, 1), 2, coeff_modulus_size, galois_element, destination.polys(0, 1), pool);
        }

        this->switch_key_internal(encrypted, destination.poly(1), galois_keys.as_kswitch_keys(), GaloisKeys::get_index(galois_element), Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst, destination, pool);
    }
    
    void Evaluator::apply_galois_plain(const Plaintext& plain, size_t galois_element, Plaintext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = plain.is_ntt_form()
            ? this->get_context_data("[Evaluator::apply_galois_plain_inplace]", plain.parms_id())
            : this->context()->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const GaloisTool& galois_tool = key_context_data->galois_tool();
        
        size_t m = coeff_count * 2;
        if ((galois_element & 1) == 0 || galois_element > m) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois element is not valid.");
        }

        destination = Plaintext::like(plain, false, pool);
        if (!plain.is_ntt_form()) {
            if (context_data->is_ckks()) {
                galois_tool.apply_p(plain.const_poly(), galois_element, coeff_modulus, destination.reference());
            } else {
                galois_tool.apply(plain.const_poly(), galois_element, context_data->parms().plain_modulus(), destination.reference());
            }
        } else {
            galois_tool.apply_ntt_p(plain.const_poly(), coeff_modulus_size, galois_element, destination.reference(), pool);
        }
    }

    void Evaluator::rotate_internal(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->get_context_data("[Evaluator::rotate_inplace_internal]", encrypted.parms_id());
        if (!context_data->qualifiers().using_batching) {
            throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Batching must be enabled to use rotate.");
        }
        if (galois_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Galois keys has incorrect parms id.");
        }
        if (steps == 0) return;
        const GaloisTool& galois_tool = context_data->galois_tool();
        if (galois_keys.has_key(galois_tool.get_element_from_step(steps))) {
            size_t element = galois_tool.get_element_from_step(steps);
            this->apply_galois(encrypted, element, galois_keys, destination, pool);
        } else {
            // Convert the steps to NAF: guarantees using smallest HW
            std::vector<int> naf_steps = utils::naf(steps);
            if (naf_steps.size() == 1) {
                throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Galois key not present.");
            }
            bool done_flag = false;
            for (int naf_step : naf_steps) {
                if (!done_flag) {
                    this->rotate_internal(encrypted, naf_step, galois_keys, destination, pool);
                    done_flag = true;
                } else {
                    Ciphertext temp;
                    this->rotate_internal(destination, naf_step, galois_keys, temp, pool);
                    destination = std::move(temp);
                }
            }
        }
    }
    
    void Evaluator::conjugate_internal(const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->get_context_data("Evaluator::conjugate_inplace_internal", encrypted.parms_id());
        if (!context_data->qualifiers().using_batching) {
            throw std::logic_error("[Evaluator::conjugate_inplace_internal] Batching is not enabled.");
        }
        const GaloisTool& galois_tool = context_data->galois_tool();
        this->apply_galois(encrypted, galois_tool.get_element_from_step(0), galois_keys, destination, pool);
    }

    void Evaluator::negacyclic_shift(const Ciphertext& encrypted, size_t shift, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::negacyclic_shift]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negacyclic_shift]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();

        destination = Ciphertext::like(encrypted, false, pool);
        utils::negacyclic_shift_ps(
            encrypted.polys(0, encrypted.polynomial_count()),
            shift, encrypted.polynomial_count(), coeff_count, coeff_modulus, 
            destination.polys(0, destination.polynomial_count())
        );
    }
}