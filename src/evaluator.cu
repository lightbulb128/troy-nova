#include "ciphertext.h"
#include "encryption_parameters.h"
#include "evaluator.h"
#include "utils/dynamic_array.h"
#include "utils/memory_pool.h"
#include "utils/ntt.h"
#include "utils/polynomial_buffer.h"
#include "fgk/dyadic_convolute.h"
#include "fgk/switch_key.h"
#include "fgk/translate_plain.h"
#include "evaluator_utils.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::NTTTables;
    using utils::ConstPointer;
    using utils::RNSTool;
    using utils::Buffer;
    using utils::MultiplyUint64Operand;
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

    __global__ static void kernel_ski_util1(
        Slice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        ConstSlice<uint64_t> key_vector_j,
        size_t key_poly_coeff_size,
        ConstSlice<uint64_t> t_operand,
        size_t key_index,
        ConstPointer<Modulus> key_modulus
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, true, nullptr);
        utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
        utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
        accumulator_l[0] = key_modulus->reduce_uint128_limbs(qword_slice.as_const());
        accumulator_l[1] = 0;
    }

    static void ski_util1(
        Slice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        ConstSlice<uint64_t> key_vector_j,
        size_t key_poly_coeff_size,
        ConstSlice<uint64_t> t_operand,
        size_t key_index,
        ConstPointer<Modulus> key_modulus
    ) {
        bool device = t_poly_lazy.on_device();
        if (!device) {
            uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, false, nullptr);
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t k = 0; k < key_component_count; k++) {
                    utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
                    size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
                    Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
                    utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
                    accumulator_l[0] = key_modulus->reduce_uint128_limbs(qword_slice.as_const());
                    accumulator_l[1] = 0;
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                key_vector_j, key_poly_coeff_size, t_operand, key_index, key_modulus
            );
            utils::stream_sync();
        }
    }
    
    __global__ static void kernel_ski_util2(
        Slice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        ConstSlice<uint64_t> key_vector_j,
        size_t key_poly_coeff_size,
        ConstSlice<uint64_t> t_operand,
        size_t key_index
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, true, nullptr);
        utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
        utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
        accumulator_l[0] = qword_slice[0];
        accumulator_l[1] = qword_slice[1];
    }

    static void ski_util2(
        Slice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        ConstSlice<uint64_t> key_vector_j,
        size_t key_poly_coeff_size,
        ConstSlice<uint64_t> t_operand,
        size_t key_index
    ) {
        bool device = t_poly_lazy.on_device();
        if (!device) {
            uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, false, nullptr);
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t k = 0; k < key_component_count; k++) {
                    utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
                    size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
                    Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
                    utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
                    accumulator_l[0] = qword_slice[0];
                    accumulator_l[1] = qword_slice[1];
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                key_vector_j, key_poly_coeff_size, t_operand, key_index
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_ski_util3(
        ConstSlice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        size_t rns_modulus_size,
        Slice<uint64_t> t_poly_prod_iter
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = t_poly_lazy[accumulator_l_offset];
    }

    static void ski_util3(
        ConstSlice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        size_t rns_modulus_size,
        Slice<uint64_t> t_poly_prod_iter
    ) {
        bool device = t_poly_lazy.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t k = 0; k < key_component_count; k++) {
                    size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
                    t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = t_poly_lazy[accumulator_l_offset];
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util3<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, rns_modulus_size, t_poly_prod_iter
            );
            utils::stream_sync();
        }
    }


    __global__ static void kernel_ski_util4(
        ConstSlice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        size_t rns_modulus_size,
        Slice<uint64_t> t_poly_prod_iter,
        ConstPointer<Modulus> key_modulus
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = key_modulus->reduce_uint128_limbs(
            t_poly_lazy.const_slice(accumulator_l_offset, accumulator_l_offset + 2)
        );
    }

    static void ski_util4(
        ConstSlice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        size_t rns_modulus_size,
        Slice<uint64_t> t_poly_prod_iter,
        ConstPointer<Modulus> key_modulus
    ) {
        bool device = t_poly_lazy.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t k = 0; k < key_component_count; k++) {
                    size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
                    t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = key_modulus->reduce_uint128_limbs(
                        t_poly_lazy.const_slice(accumulator_l_offset, accumulator_l_offset + 2)
                    );
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util4<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                rns_modulus_size, t_poly_prod_iter, key_modulus
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_ski_util5_step1(
        ConstSlice<uint64_t> t_last,
        size_t coeff_count,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        uint64_t qk_inv_qp,
        uint64_t qk,
        Slice<uint64_t> delta_array
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        uint64_t k = utils::barrett_reduce_uint64(t_last[i], *plain_modulus);
        k = utils::negate_uint64_mod(k, *plain_modulus);
        if (qk_inv_qp != 1) {
            k = utils::multiply_uint64_mod(k, qk_inv_qp, *plain_modulus);
        }
        uint64_t delta = utils::barrett_reduce_uint64(k, key_modulus[j]);
        delta = utils::multiply_uint64_mod(delta, qk, key_modulus[j]);
        uint64_t c_mod_qi = utils::barrett_reduce_uint64(t_last[i], key_modulus[j]);
        delta = utils::add_uint64_mod(delta, c_mod_qi, key_modulus[j]);
        delta_array[global_index] = delta;
    }


    __global__ static void kernel_ski_util5_step2(
        Slice<uint64_t> t_poly_prod_i,
        size_t coeff_count,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> encrypted_i,
        ConstSlice<uint64_t> delta_array,
        bool add_inplace
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        uint64_t delta = delta_array[global_index];
        uint64_t& target = t_poly_prod_i[j * coeff_count + i];
        target = utils::sub_uint64_mod(target, delta, key_modulus[j]);
        target = utils::multiply_uint64operand_mod(target, modswitch_factors[j], key_modulus[j]);
        if (add_inplace) {
            encrypted_i[global_index] = utils::add_uint64_mod(target, encrypted_i[global_index], key_modulus[j]);
        } else {
            encrypted_i[global_index] = target;
        }
    }


    static void ski_util5(
        ConstSlice<uint64_t> t_last,
        Slice<uint64_t> t_poly_prod_i,
        size_t coeff_count,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<NTTTables> key_ntt_tables,
        size_t decomp_modulus_size,
        uint64_t qk_inv_qp,
        uint64_t qk,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> destination_i,
        bool add_inplace,
        MemoryPoolHandle pool
    ) {
        bool device = t_last.on_device();
        if (!device) {

            Buffer<uint64_t> buffer(coeff_count * 3, false, nullptr);
            Slice<uint64_t> delta = buffer.slice(0, coeff_count);
            Slice<uint64_t> c_mod_qi = buffer.slice(coeff_count, 2 * coeff_count);
            Slice<uint64_t> k = buffer.slice(2 * coeff_count, 3 * coeff_count);

            utils::modulo(t_last, plain_modulus, k);
            utils::negate_inplace(k, plain_modulus);
            if (qk_inv_qp != 1) {
                utils::multiply_scalar_inplace(k, qk_inv_qp, plain_modulus);
            }

            for (size_t j = 0; j < decomp_modulus_size; j++) {
                // delta = k mod q_i
                utils::modulo(k.as_const(), key_modulus.at(j), delta);
                // delta = k * q_k mod q_i
                utils::multiply_scalar_inplace(delta, qk, key_modulus.at(j));
                // c mod q_i
                utils::modulo(t_last, key_modulus.at(j), c_mod_qi);
                // delta = c + k * q_k mod q_i
                // c_{i} = c_{i} - delta mod q_i
                utils::add_inplace(delta, c_mod_qi.as_const(), key_modulus.at(j));
                utils::ntt_inplace(delta, coeff_count, key_ntt_tables.at(j));
                Slice<uint64_t> t_poly_prod_i_comp_j = t_poly_prod_i.slice(j * coeff_count, (j + 1) * coeff_count);
                utils::sub_inplace(t_poly_prod_i_comp_j, delta.as_const(), key_modulus.at(j));
                utils::multiply_uint64operand_inplace(t_poly_prod_i_comp_j, modswitch_factors.at(j), key_modulus.at(j));
                if (add_inplace) {
                    utils::add_inplace(destination_i.slice(j * coeff_count, (j + 1) * coeff_count), t_poly_prod_i_comp_j.as_const(), key_modulus.at(j));
                } else {
                    destination_i.slice(j * coeff_count, (j + 1) * coeff_count).copy_from_slice(t_poly_prod_i_comp_j.as_const());
                }
            }

        } else {
            Buffer<uint64_t> delta(coeff_count * decomp_modulus_size, true, pool);
            size_t block_count = utils::ceil_div(coeff_count * decomp_modulus_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_last.device_index());
            kernel_ski_util5_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, coeff_count, plain_modulus, key_modulus, 
                decomp_modulus_size, qk_inv_qp, qk,
                delta.reference()
            );
            utils::stream_sync();
            utils::ntt_inplace_p(delta.reference(), coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size));
            utils::set_device(t_poly_prod_i.device_index());
            kernel_ski_util5_step2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_prod_i, coeff_count, key_modulus, 
                decomp_modulus_size, modswitch_factors, destination_i, delta.const_reference(), add_inplace
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_ski_util6(
        Slice<uint64_t> t_last,
        size_t coeff_count,
        ConstPointer<Modulus> qk,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        Slice<uint64_t> t_ntt
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count) return;
        size_t i = global_index % coeff_count;
        uint64_t qk_half = qk->value() >> 1;
        t_last[i] = utils::barrett_reduce_uint64(t_last[i] + qk_half, *qk);
        for (size_t j = 0; j < decomp_modulus_size; j++) {
            const Modulus& qi = key_modulus[j];
            if (qk->value() > qi.value()) {
                t_ntt[j * coeff_count + i] = utils::barrett_reduce_uint64(t_last[i], qi);
            } else {
                t_ntt[j * coeff_count + i] = t_last[i];
            }
            uint64_t fix = qi.value() - utils::barrett_reduce_uint64(qk_half, key_modulus[j]);
            t_ntt[j * coeff_count + i] += fix;
        }
    }

    static void ski_util6(
        Slice<uint64_t> t_last,
        size_t coeff_count,
        ConstPointer<Modulus> qk,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        Slice<uint64_t> t_ntt
    ) {
        bool device = t_last.on_device();
        if (!device) {
            uint64_t qk_half = qk->value() >> 1;
            for (size_t i = 0; i < coeff_count; i++) {
                t_last[i] = utils::barrett_reduce_uint64(t_last[i] + qk_half, *qk);
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    const Modulus& qi = key_modulus[j];
                    if (qk->value() > qi.value()) {
                        t_ntt[j * coeff_count + i] = utils::barrett_reduce_uint64(t_last[i], qi);
                    } else {
                        t_ntt[j * coeff_count + i] = t_last[i];
                    }
                    uint64_t fix = qi.value() - utils::barrett_reduce_uint64(qk_half, key_modulus[j]);
                    t_ntt[j * coeff_count + i] += fix;
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_last.device_index());
            kernel_ski_util6<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, coeff_count, qk, key_modulus, decomp_modulus_size, t_ntt
            );
            utils::stream_sync();
        }
    }

    
    void kernel_ski_util6_merged(
        Slice<uint64_t> t_last,
        size_t coeff_count,
        ConstPointer<Modulus> qk,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        Slice<uint64_t> t_ntt
    ) {
        bool device = t_last.on_device();
        if (!device) {
            uint64_t qk_half = qk->value() >> 1;
            for (size_t i = 0; i < coeff_count; i++) {
                t_last[i] = utils::barrett_reduce_uint64(t_last[i] + qk_half, *qk);
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    const Modulus& qi = key_modulus[j];
                    if (qk->value() > qi.value()) {
                        t_ntt[j * coeff_count + i] = utils::barrett_reduce_uint64(t_last[i], qi);
                    } else {
                        t_ntt[j * coeff_count + i] = t_last[i];
                    }
                    uint64_t fix = qi.value() - utils::barrett_reduce_uint64(qk_half, key_modulus[j]);
                    t_ntt[j * coeff_count + i] += fix;
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_last.device_index());
            kernel_ski_util6<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, coeff_count, qk, key_modulus, decomp_modulus_size, t_ntt
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_ski_util5_merged_step1(
        ConstSlice<uint64_t> poly_prod_intt,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        uint64_t qk_inv_qp,
        Slice<uint64_t> delta
     ) {

        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= key_component_count * coeff_count) return;
        size_t k = global_index / coeff_count;
        size_t i = global_index % coeff_count;
        size_t t_last_offset = coeff_count * (decomp_modulus_size + 1) * k + decomp_modulus_size * coeff_count;
        
        uint64_t kt = plain_modulus->reduce(poly_prod_intt[i + t_last_offset]);
        kt = utils::negate_uint64_mod(kt, *plain_modulus);
        if (qk_inv_qp != 1) {
            kt = utils::multiply_uint64_mod(kt, qk_inv_qp, *plain_modulus);
        }

        uint64_t qk = key_modulus[key_modulus.size() - 1].value();

        for (size_t j = 0; j < decomp_modulus_size; j++) {
            uint64_t delta_result = 0;
            const Modulus& qi = key_modulus[j];
            // delta = k mod q_i
            delta_result = qi.reduce(kt);
            // delta = k * q_k mod q_i
            delta_result = utils::multiply_uint64_mod(delta_result, qk, qi);
            // c mod q_i
            uint64_t c_mod_qi = qi.reduce(poly_prod_intt[i + t_last_offset]);
            // delta = c + k * q_k mod q_i
            // c_{i} = c_{i} - delta mod q_i
            delta_result = utils::add_uint64_mod(delta_result, c_mod_qi, qi);
            delta[(k * decomp_modulus_size + j) * coeff_count + i] = delta_result;
        }

    }
    
    __global__ static void kernel_ski_util5_merged_step2(
        ConstSlice<uint64_t> delta_array,
        Slice<uint64_t> poly_prod,
        size_t coeff_count,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> destination,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        
        bool add_inplace = (assign_method == Evaluator::SwitchKeyDestinationAssignMethod::AddInplace) || (
            k == 0 && assign_method == Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst
        );

        for (size_t j = 0; j < decomp_modulus_size; j++) {
            size_t index = (k * decomp_modulus_size + j) * coeff_count + i;
            uint64_t delta = delta_array[index];
            uint64_t& target = poly_prod[(k * (decomp_modulus_size + 1) + j) * coeff_count + i];
            target = utils::sub_uint64_mod(target, delta, key_modulus[j]);
            target = utils::multiply_uint64operand_mod(target, modswitch_factors[j], key_modulus[j]);
            if (add_inplace) {
                destination[index] = utils::add_uint64_mod(target, destination[index], key_modulus[j]);
            } else {
                destination[index] = target;
            }
        }
    }

    __global__ static void kernel_ski_util6_merged(
        Slice<uint64_t> poly_prod_intt,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        ConstPointer<Modulus> qk,
        ConstSlice<Modulus> key_modulus,
        Slice<uint64_t> temp_last
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= key_component_count * coeff_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        uint64_t qk_half = qk->value() >> 1;
        size_t offset = coeff_count * (decomp_modulus_size + 1) * k + decomp_modulus_size * coeff_count;
        size_t poly_prod_intt_i = utils::barrett_reduce_uint64(poly_prod_intt[offset + i] + qk_half, *qk);
        for (size_t j = 0; j < decomp_modulus_size; j++) {
            uint64_t result;
            const Modulus& qi = key_modulus[j];
            if (qk->value() > qi.value()) {
                result = utils::barrett_reduce_uint64(poly_prod_intt_i, qi);
            } else {
                result = poly_prod_intt_i;
            }
            uint64_t fix = qi.value() - utils::barrett_reduce_uint64(qk_half, key_modulus[j]);
            result += fix;
            temp_last[(k * decomp_modulus_size + j) * coeff_count + i] = result;
        }
        
    }

    __global__ static void kernel_ski_util7_merged(
        Slice<uint64_t> poly_prod,
        ConstSlice<uint64_t> temp_last,
        size_t coeff_count, 
        Slice<uint64_t> destination,
        bool is_ckks,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        for (size_t k = 0; k < key_component_count; k++) {
            size_t offset = k * decomp_modulus_size * coeff_count;
            uint64_t& dest = poly_prod[k * (decomp_modulus_size + 1) * coeff_count + j * coeff_count + i];
            uint64_t qi = key_modulus[j].value();
            dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - temp_last[offset + j * coeff_count + i];
            dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
            bool add_inplace = (assign_method == Evaluator::SwitchKeyDestinationAssignMethod::AddInplace) || (
                k == 0 && assign_method == Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst
            );
            if (add_inplace) {
                destination[offset + j * coeff_count + i] = utils::add_uint64_mod(
                    destination[offset + j * coeff_count + i], dest, key_modulus[j]
                );
            } else {
                destination[offset + j * coeff_count + i] = dest;
            }
        }
    }

    __global__ static void kernel_ski_util7(
        Slice<uint64_t> t_poly_prod_i,
        ConstSlice<uint64_t> t_ntt,
        size_t coeff_count, 
        Slice<uint64_t> destination_i,
        bool is_ckks,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        bool add_inplace
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        uint64_t& dest = t_poly_prod_i[j*coeff_count + i];
        uint64_t qi = key_modulus[j].value();
        dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - t_ntt[j * coeff_count + i];
        dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
        if (add_inplace) {
            destination_i[j * coeff_count + i] = utils::add_uint64_mod(
                destination_i[j * coeff_count + i], dest, key_modulus[j]
            );
        } else {
            destination_i[j * coeff_count + i] = dest;
        }
    }

    static void ski_util7(
        Slice<uint64_t> t_poly_prod_i,
        ConstSlice<uint64_t> t_ntt,
        size_t coeff_count, 
        Slice<uint64_t> destination_i,
        bool is_ckks,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        bool add_inplace
    ) {
        bool device = t_poly_prod_i.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    uint64_t& dest = t_poly_prod_i[j*coeff_count + i];
                    uint64_t qi = key_modulus[j].value();
                    dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - t_ntt[j * coeff_count + i];
                    dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
                    if (add_inplace) {
                        destination_i[j * coeff_count + i] = utils::add_uint64_mod(
                            destination_i[j * coeff_count + i], dest, key_modulus[j]
                        );
                    } else {
                        destination_i[j * coeff_count + i] = dest;
                    }
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * decomp_modulus_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_prod_i.device_index());
            kernel_ski_util7<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_prod_i, t_ntt, coeff_count, destination_i, is_ckks, 
                decomp_modulus_size, key_modulus, modswitch_factors, add_inplace
            );
            utils::stream_sync();
        }
    }

    void Evaluator::switch_key_internal(
        const Ciphertext& encrypted, utils::ConstSlice<uint64_t> target, 
        const KSwitchKeys& kswitch_keys, size_t kswitch_keys_index, 
        SwitchKeyDestinationAssignMethod assign_method, 
        Ciphertext& destination, MemoryPoolHandle pool
    ) const {
        check_no_seed("[Evaluator::switch_key_inplace_internal]", encrypted);
        if (!this->context()->using_keyswitching()) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Keyswitching is not supported.");
        }
        if (kswitch_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Keyswitching key has incorrect parms id.");
        }
        if (kswitch_keys_index >= kswitch_keys.data().size()) {
            throw std::out_of_range("[Evaluator::switch_key_inplace_internal] Key switch keys index out of range.");
        }

        ParmsID parms_id = encrypted.parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::switch_key_inplace_internal]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const EncryptionParameters& key_parms = key_context_data->parms();
        SchemeType scheme = parms.scheme();
        bool is_ntt_form = encrypted.is_ntt_form();

        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        ConstSlice<Modulus> key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        ConstSlice<NTTTables> key_ntt_tables = key_context_data->small_ntt_tables();
        ConstSlice<MultiplyUint64Operand> modswitch_factors = key_context_data->rns_tool().inv_q_last_mod_q();

        const std::vector<PublicKey>& key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].as_ciphertext().polynomial_count();

        if (destination.polynomial_count() < key_component_count) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Destination should have at least same amount of polys as the key switching key.");
        }
        if (destination.polynomial_count() > key_component_count && assign_method != SwitchKeyDestinationAssignMethod::AddInplace) {
            destination.polys(key_component_count, destination.polynomial_count()).set_zero();
        }
        if (destination.parms_id() != parms_id) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Destination parms_id should match the input parms_id.");
        }
        if (!utils::device_compatible(encrypted, destination, target)) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Incompatible encryption parameters.");
        }

        for (size_t i = 0; i < key_vector.size(); i++) {
            check_no_seed("[Evaluator::switch_key_inplace_internal]", key_vector[i].as_ciphertext());
        }

        if (target.size() != decomp_modulus_size * coeff_count) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Invalid target size.");
        }
        Buffer<uint64_t> target_copied_underlying(target.size(), target.on_device(), pool);
        ConstSlice<uint64_t> target_copied(nullptr, 0, false, nullptr);

        // If target is in NTT form; switch back to normal form
        if (is_ntt_form) {
            utils::intt_p(
                target, coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size), target_copied_underlying.reference()
            );
            target_copied = target_copied_underlying.const_reference();
        } else {
            target_copied = target;
        }

        // Temporary result
        bool device = target.on_device();

        // We fuse some kernels for device only. This is not necessary for host execution.

        Buffer<uint64_t> poly_prod(key_component_count, rns_modulus_size, coeff_count, device, pool);

        if (!device) { // host

            Buffer<uint64_t> poly_lazy(key_component_count * coeff_count * 2, device, pool);
            Buffer<uint64_t> temp_ntt(coeff_count, device, pool);

            for (size_t i = 0; i < rns_modulus_size; i++) {
                size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);

                // Product of two numbers is up to 60 + 60 = 120 bits, so we can sum up to 256 of them without reduction.
                size_t lazy_reduction_summand_bound = utils::HE_MULTIPLY_ACCUMULATE_USER_MOD_MAX;
                size_t lazy_reduction_counter = lazy_reduction_summand_bound;

                // Allocate memory for a lazy accumulator (128-bit coefficients)
                poly_lazy.set_zero();

                // Multiply with keys and perform lazy reduction on product's coefficients
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    ConstSlice<uint64_t> temp_operand(nullptr, 0, device, nullptr);
                    if (is_ntt_form && (i == j)) {
                        temp_operand = target.const_slice(j * coeff_count, (j + 1) * coeff_count);
                    } else {
                        if (key_modulus[j].value() <= key_modulus[key_index].value()) {
                            temp_ntt.copy_from_slice(target_copied.const_slice(j * coeff_count, (j + 1) * coeff_count));
                        } else {
                            utils::modulo(target_copied.const_slice(j * coeff_count, (j + 1) * coeff_count), key_modulus.at(key_index), temp_ntt.reference());
                        }
                        utils::ntt_inplace(temp_ntt.reference(), coeff_count, key_ntt_tables.at(key_index));
                        temp_operand = temp_ntt.const_reference();
                    }
                    
                    // Multiply with keys and modular accumulate products in a lazy fashion
                    size_t key_vector_poly_coeff_size = key_modulus_size * coeff_count;

                    if (!lazy_reduction_counter) {
                        ski_util1(
                            poly_lazy.reference(), coeff_count, key_component_count,
                            key_vector[j].as_ciphertext().const_reference(),
                            key_vector_poly_coeff_size,
                            temp_operand, key_index, key_modulus.at(key_index)
                        );
                    } else {
                        ski_util2(
                            poly_lazy.reference(), coeff_count, key_component_count,
                            key_vector[j].as_ciphertext().const_reference(),
                            key_vector_poly_coeff_size,
                            temp_operand, key_index
                        );
                    }

                    lazy_reduction_counter -= 1;
                    if (lazy_reduction_counter == 0) {
                        lazy_reduction_counter = lazy_reduction_summand_bound;
                    }
                }
                
                Slice<uint64_t> t_poly_prod_iter = poly_prod.slice(i * coeff_count, poly_prod.size());

                if (lazy_reduction_counter == lazy_reduction_summand_bound) {
                    ski_util3(
                        poly_lazy.const_reference(), coeff_count, key_component_count,
                        rns_modulus_size, t_poly_prod_iter
                    );
                } else {
                    ski_util4(
                        poly_lazy.const_reference(), coeff_count, key_component_count,
                        rns_modulus_size, t_poly_prod_iter,
                        key_modulus.at(key_index)
                    );
                }
            } // i

        } else { // device

            Buffer<uint64_t> poly_lazy(key_component_count * coeff_count * 2, device, pool);
            Buffer<uint64_t> temp_ntt(rns_modulus_size, decomp_modulus_size, coeff_count, device, pool);

            utils::fgk::switch_key::set_accumulate(decomp_modulus_size, coeff_count, target_copied, temp_ntt, key_modulus);

            utils::ntt_inplace_ps(temp_ntt.reference(), rns_modulus_size, decomp_modulus_size, coeff_count, 
                utils::NTTTableIndexer::key_switching_set_products(key_ntt_tables, decomp_modulus_size));

            utils::fgk::switch_key::accumulate_products(
                decomp_modulus_size, key_component_count, coeff_count, temp_ntt.const_reference(), 
                key_modulus, kswitch_keys.get_data_ptrs(kswitch_keys_index), poly_prod.reference()
            );

        }
        
        // Accumulated products are now stored in t_poly_prod

        if (!device) { // host
            Buffer<uint64_t> temp_ntt = Buffer<uint64_t>(decomp_modulus_size, coeff_count, device, pool);
            for (size_t i = 0; i < key_component_count; i++) {
                
                bool add_inplace = (assign_method == SwitchKeyDestinationAssignMethod::AddInplace) || 
                    (i == 0 && assign_method == SwitchKeyDestinationAssignMethod::OverwriteExceptFirst);

                if (scheme == SchemeType::BGV) {
                    // qk is the special prime
                    uint64_t qk = key_modulus[key_modulus_size - 1].value();
                    uint64_t qk_inv_qp = this->context()->key_context_data().value()->rns_tool().inv_q_last_mod_t();

                    // Lazy reduction; this needs to be then reduced mod qi
                    size_t t_last_offset = coeff_count * rns_modulus_size * i + decomp_modulus_size * coeff_count;
                    Slice<uint64_t> t_last = poly_prod.slice(t_last_offset, t_last_offset + coeff_count);
                    utils::intt_inplace(t_last, coeff_count, key_ntt_tables.at(key_modulus_size - 1));
                    ConstPointer<Modulus> plain_modulus = parms.plain_modulus();

                    ski_util5(
                        t_last.as_const(), poly_prod.slice(i * coeff_count * rns_modulus_size, poly_prod.size()),
                        coeff_count, plain_modulus, key_modulus, key_ntt_tables,
                        decomp_modulus_size, qk_inv_qp, qk,
                        modswitch_factors, destination.poly(i),
                        add_inplace,
                        pool
                    );
                } else {
                    // Lazy reduction; this needs to be then reduced mod qi
                    size_t t_last_offset = coeff_count * rns_modulus_size * i + decomp_modulus_size * coeff_count;
                    Slice<uint64_t> t_last = poly_prod.slice(t_last_offset, t_last_offset + coeff_count);
                    // temp_ntt.set_zero();
                    utils::intt_inplace(t_last, coeff_count, key_ntt_tables.at(key_modulus_size - 1));

                    ski_util6(
                        t_last, coeff_count, key_modulus.at(key_modulus_size - 1),
                        key_modulus,
                        decomp_modulus_size,
                        temp_ntt.reference()
                    );
                    
                    if (is_ntt_form) {
                        utils::ntt_inplace_p(temp_ntt.reference(), coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size));
                    } else {
                        utils::intt_inplace_p(
                            poly_prod.slice(
                                i * coeff_count * rns_modulus_size, 
                                i * coeff_count * rns_modulus_size + decomp_modulus_size * coeff_count
                            ), 
                            coeff_count, 
                            key_ntt_tables.const_slice(0, decomp_modulus_size)
                        );
                    }

                    ski_util7(
                        poly_prod.slice(i * coeff_count * rns_modulus_size, poly_prod.size()),
                        temp_ntt.const_reference(),
                        coeff_count, destination.poly(i),
                        scheme==SchemeType::CKKS, decomp_modulus_size, key_modulus,
                        modswitch_factors,
                        add_inplace
                    );
                }
            }

        } else { // device

            Buffer<uint64_t> poly_prod_intt(key_component_count, rns_modulus_size, coeff_count, device, pool);
            
            // When is_ntt_form is true, we actually only need the INTT of the every polynomial's last component. 
            // but iteration over the polynomial and then conduct INTT will be inefficient because of multiple kernel calls.
            utils::intt_ps(
                poly_prod.const_reference(), key_component_count, rns_modulus_size, coeff_count, 
                poly_prod_intt.reference(), utils::NTTTableIndexer::key_switching_skip_finals(key_ntt_tables, decomp_modulus_size)
            );
            
            if (scheme == SchemeType::BGV) {

                Buffer<uint64_t> delta(key_component_count, decomp_modulus_size, coeff_count, device, pool);
                
                size_t total = key_component_count * coeff_count;
                size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
                utils::set_device(poly_prod_intt.device_index());
                kernel_ski_util5_merged_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    poly_prod_intt.const_reference(),
                    parms.plain_modulus(), key_modulus, key_component_count, decomp_modulus_size, coeff_count,
                    this->context()->key_context_data().value()->rns_tool().inv_q_last_mod_t(),
                    delta.reference()
                );
                utils::stream_sync();
                utils::ntt_inplace_ps(delta.reference(), key_component_count, coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size));
                utils::set_device(poly_prod.device_index());
                kernel_ski_util5_merged_step2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    delta.const_reference(), poly_prod.reference(), coeff_count, key_modulus, key_component_count, decomp_modulus_size, modswitch_factors, 
                    destination.data().reference(), assign_method
                );

            } else {

                Buffer<uint64_t> temp_last(key_component_count, decomp_modulus_size, coeff_count, device, pool);
                
                size_t block_count = utils::ceil_div(key_component_count * coeff_count, utils::KERNEL_THREAD_COUNT);
                kernel_ski_util6_merged<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    poly_prod_intt.reference(),
                    key_component_count, decomp_modulus_size, coeff_count, key_modulus.at(key_modulus_size - 1),
                    key_modulus,
                    temp_last.reference()
                );
                if (is_ntt_form) {
                    utils::ntt_inplace_ps(temp_last.reference(), key_component_count, coeff_count, 
                        key_ntt_tables.const_slice(0, decomp_modulus_size));
                }
                if (!is_ntt_form) {
                    poly_prod = std::move(poly_prod_intt);
                }

                block_count = utils::ceil_div(coeff_count * decomp_modulus_size, utils::KERNEL_THREAD_COUNT);
                kernel_ski_util7_merged<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    poly_prod.reference(),
                    temp_last.const_reference(),
                    coeff_count, destination.data().reference(),
                    scheme==SchemeType::CKKS, key_component_count, decomp_modulus_size,
                    key_modulus, modswitch_factors, assign_method
                );

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
            galois_tool.apply_p(encrypted.const_poly(0), galois_element, coeff_modulus, destination.poly(0));
            galois_tool.apply_p(encrypted.const_poly(1), galois_element, coeff_modulus, destination.poly(1));
        } else {
            galois_tool.apply_ntt_p(encrypted.const_poly(0), coeff_modulus_size, galois_element, destination.poly(0), pool);
            galois_tool.apply_ntt_p(encrypted.const_poly(1), coeff_modulus_size, galois_element, destination.poly(1), pool);
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

    __global__ static void kernel_extract_lwe_gather_c0(
        size_t coeff_modulus_size, size_t coeff_count, size_t term,
        ConstSlice<uint64_t> rlwe_c0, Slice<uint64_t> c0
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_modulus_size) return;
        c0[i] = rlwe_c0[coeff_count * i + term];
    }

    static void extract_lwe_gather_c0(
        size_t coeff_modulus_size, size_t coeff_count, size_t term,
        ConstSlice<uint64_t> rlwe_c0, Slice<uint64_t> c0
    ) {
        bool device = rlwe_c0.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                c0[i] = rlwe_c0[coeff_count * i + term];
            }
        } else {
            if (coeff_modulus_size >= utils::KERNEL_THREAD_COUNT) {
                size_t block_count = utils::ceil_div(coeff_modulus_size, utils::KERNEL_THREAD_COUNT);
                utils::set_device(c0.device_index());
                kernel_extract_lwe_gather_c0<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    coeff_modulus_size, coeff_count, term, rlwe_c0, c0
                );
                utils::stream_sync();
            } else {
                utils::set_device(c0.device_index());
                kernel_extract_lwe_gather_c0<<<1, coeff_modulus_size>>>(
                    coeff_modulus_size, coeff_count, term, rlwe_c0, c0
                );
                utils::stream_sync();
            }
        }
    }
    
    LWECiphertext Evaluator::extract_lwe_new(const Ciphertext& encrypted, size_t term, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::extract_lwe_new]", encrypted);
        if (encrypted.polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::extract_lwe_new] Ciphertext size must be 2.");
        }
        if (encrypted.is_ntt_form()) {
            Ciphertext transformed;
            this->transform_from_ntt(encrypted, transformed, pool);
            return this->extract_lwe_new(transformed, term, pool);
        }
        // else
        ContextDataPointer context_data = this->get_context_data("[Evaluator::extract_lwe_new]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = parms.coeff_modulus().size();

        // gather c1
        size_t shift = (term == 0) ? 0 : (coeff_count * 2 - term);
        bool device = encrypted.on_device();
        utils::DynamicArray<uint64_t> c1 = utils::DynamicArray<uint64_t>::create_uninitialized(coeff_count * coeff_modulus_size, device, pool);
        utils::negacyclic_shift_p(
            encrypted.const_poly(1), shift, coeff_count, coeff_modulus, c1.reference()
        );

        // gather c0
        utils::DynamicArray<uint64_t> c0 = utils::DynamicArray<uint64_t>::create_uninitialized(coeff_modulus_size, device, pool);
        extract_lwe_gather_c0(
            coeff_modulus_size, coeff_count, term,
            encrypted.const_poly(0), c0.reference()
        );

        // set lwe
        LWECiphertext ret;
        ret.coeff_modulus_size() = coeff_modulus_size;
        ret.poly_modulus_degree() = coeff_count;
        ret.c0_dyn() = std::move(c0);
        ret.c1_dyn() = std::move(c1);
        ret.parms_id() = encrypted.parms_id();
        ret.scale() = encrypted.scale();
        ret.correction_factor() = encrypted.correction_factor();
        return ret;
    }

    
    void Evaluator::field_trace_inplace(Ciphertext& encrypted, const GaloisKeys& automorphism_keys, size_t logn, MemoryPoolHandle pool) const {
        size_t poly_degree = encrypted.poly_modulus_degree();
        Ciphertext temp;
        while (poly_degree > (static_cast<size_t>(1) << logn)) {
            size_t galois_element = poly_degree + 1;
            this->apply_galois(encrypted, galois_element, automorphism_keys, temp, pool);
            this->add_inplace(encrypted, temp, pool);
            poly_degree >>= 1;
        }
    }
    
    void Evaluator::divide_by_poly_modulus_degree_inplace(Ciphertext& encrypted, uint64_t mul) const {
        ContextDataPointer context_data = this->get_context_data("[Evaluator::divide_by_poly_modulus_degree_inplace]", encrypted.parms_id());
        size_t size = encrypted.polynomial_count();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        ConstSlice<Modulus> coeff_modulus = context_data->parms().coeff_modulus();
        size_t n = context_data->parms().poly_modulus_degree();
        size_t logn = static_cast<size_t>(utils::get_power_of_two(n));
        utils::ntt_multiply_inv_degree(
            encrypted.polys(0, size), size, logn, ntt_tables
        );
        if (mul != 1) {
            utils::multiply_scalar_ps(encrypted.const_polys(0, size), mul, size, n, coeff_modulus, encrypted.polys(0, size));
        }
    }
    
    Ciphertext Evaluator::pack_lwe_ciphertexts_new(const std::vector<LWECiphertext>& lwes, const GaloisKeys& automorphism_keys, MemoryPoolHandle pool) const {
        
        size_t lwes_count = lwes.size();
        if (lwes_count == 0) {
            throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must not be empty.");
        }
        ParmsID lwe_parms_id = lwes[0].parms_id();
        // check all have same parms id
        for (size_t i = 1; i < lwes_count; i++) {
            if (lwes[i].parms_id() != lwe_parms_id) {
                throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must have same parms id.");
            }
        }

        ContextDataPointer context_data = this->get_context_data("[Evaluator::pack_lwe_ciphertexts_new]", lwe_parms_id);
        SchemeType scheme = context_data->parms().scheme();
        bool ntt_form = scheme == SchemeType::CKKS || scheme == SchemeType::BGV;
        if (scheme == SchemeType::CKKS) {
            // all should have same scale
            double scale = lwes[0].scale();
            for (size_t i = 1; i < lwes_count; i++) {
                if (!utils::are_close_double(lwes[i].scale(), scale)) {
                    throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must have same scale.");
                }
            }
        }
        if (scheme == SchemeType::BGV) {
            // all should have same correction factor
            uint64_t cf = lwes[0].correction_factor();
            for (size_t i = 1; i < lwes_count; i++) {
                if (lwes[i].correction_factor() != cf) {
                    throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must have same correction factor.");
                }
            }
        }
        size_t poly_modulus_degree = context_data->parms().poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = context_data->parms().coeff_modulus();
        if (lwes_count > poly_modulus_degree) {
            throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts count must be less than poly_modulus_degree.");
        }
        size_t l = 0;
        while ((static_cast<size_t>(1) << l) < lwes_count) l += 1;
        std::vector<Ciphertext> rlwes(1 << l);
        Ciphertext zero_rlwe = this->assemble_lwe_new(lwes[0], pool);
        zero_rlwe.data().reference().set_zero();
        for (size_t i = 0; i < (static_cast<size_t>(1)<<l); i++) {
            size_t index = static_cast<size_t>(utils::reverse_bits_uint64(static_cast<uint64_t>(i), l));
            if (index < lwes_count) {
                rlwes[i] = this->assemble_lwe_new(lwes[index], pool);
                this->divide_by_poly_modulus_degree_inplace(rlwes[i]);
            } else {
                rlwes[i] = zero_rlwe.clone(pool);
            }
        }
        Ciphertext temp(std::move(zero_rlwe));
        for (size_t layer = 0; layer < l; layer++) {
            size_t gap = 1 << layer;
            size_t offset = 0;
            size_t shift = poly_modulus_degree >> (layer + 1);
            while (offset < (static_cast<size_t>(1) << l)) {
                Ciphertext& even = rlwes[offset];
                Ciphertext& odd = rlwes[offset + gap];
                utils::negacyclic_shift_ps(
                    odd.const_reference(), shift, odd.polynomial_count(), 
                    poly_modulus_degree, coeff_modulus, temp.reference()
                );
                this->sub(even, temp, odd, pool);
                this->add_inplace(even, temp, pool);
                if (ntt_form) {
                    this->transform_to_ntt_inplace(odd);
                }
                this->apply_galois_inplace(odd, (1 << (layer + 1)) + 1, automorphism_keys, pool);
                if (ntt_form) {
                    this->transform_from_ntt_inplace(odd);
                }
                this->add_inplace(even, odd, pool);
                offset += (gap << 1);
            }
        }
        // take the first element
        Ciphertext ret = std::move(rlwes[0]);
        if (ntt_form) {
            this->transform_to_ntt_inplace(ret);
        }
        field_trace_inplace(ret, automorphism_keys, l, pool);
        return ret;
    }
}