#include "evaluator.cuh"
#include "utils/polynomial_buffer.cuh"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::Array;
    using utils::NTTTables;
    using utils::Pointer;
    using utils::ConstPointer;
    using utils::RNSTool;
    using utils::Buffer;
    using utils::MultiplyUint64Operand;

    template <typename C>
    inline static void check_no_seed(const char* prompt, const C& c) {
        if (c.contains_seed()) {
            throw std::invalid_argument(std::string(prompt) + " Argument contains seed.");
        }
    }

    inline void check_ciphertext(const char* prompt, const Ciphertext& ciphertext) {
        check_no_seed(prompt, ciphertext);
    }

    template <typename C1, typename C2>
    inline static void check_same_parms_id(const char* prompt, const C1& a, const C2& b) {
        if (a.parms_id() != b.parms_id()) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different parms ID.");
        }
    }

    inline static void check_same_scale(const char* prompt, const Ciphertext& a, const Ciphertext& b) {
        if (!utils::are_close_double(a.scale(), b.scale())) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different scales.");
        }
    }

    inline static void check_same_ntt_form(const char* prompt, const Ciphertext& a, const Ciphertext& b) {
        if (a.is_ntt_form() != b.is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Arguments have different NTT form.");
        }
    }

    inline static void check_is_ntt_form(const char* prompt, const Ciphertext& a) {
        if (!a.is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Argument is not in NTT form.");
        }
    }

    inline static void check_is_not_ntt_form(const char* prompt, const Ciphertext& a) {
        if (a.is_ntt_form()) {
            throw std::invalid_argument(std::string(prompt) + " Argument is in NTT form.");
        }
    }

    static void balance_correction_factors(
        uint64_t factor1, uint64_t factor2, const Modulus& plain_modulus,
        uint64_t& prod, uint64_t& e1, uint64_t& e2
    ) {
        uint64_t t = plain_modulus.value();
        uint64_t half_t = t >> 1;
        auto sum_abs = [half_t, t](uint64_t x, uint64_t y) -> uint64_t {
            int64_t x_bal = x > half_t ? static_cast<int64_t>(x - t) : static_cast<int64_t>(x);
            int64_t y_bal = y > half_t ? static_cast<int64_t>(y - t) : static_cast<int64_t>(y);
            return static_cast<uint64_t>(std::abs(x_bal) + std::abs(y_bal));
        };
        uint64_t ratio = 1;
        if (!utils::try_invert_uint64_mod(factor1, plain_modulus, ratio)) {
            throw std::logic_error("[balance_correction_factors] Failed to invert factor1.");
        }
        ratio = utils::multiply_uint64_mod(ratio, factor2, plain_modulus);
        e1 = ratio;
        e2 = 1;
        uint64_t sum = sum_abs(factor1, factor2);
        
        // Extended Euclidean
        int64_t prev_a = static_cast<int64_t>(plain_modulus.value());
        int64_t prev_b = static_cast<int64_t>(0);
        int64_t a = static_cast<int64_t>(ratio);
        int64_t b = static_cast<int64_t>(1);
        while (a != 0) {
            int64_t q = prev_a / a;
            int64_t temp = prev_a % a;
            prev_a = a;
            a = temp;
            temp = prev_b - q * b;
            prev_b = b;
            b = temp;
            uint64_t a_mod = plain_modulus.reduce(static_cast<uint64_t>(std::abs(a)));
            if (a < 0) {a_mod = utils::negate_uint64_mod(a_mod, plain_modulus);}
            uint64_t b_mod = plain_modulus.reduce(static_cast<uint64_t>(std::abs(b)));
            if (b < 0) {b_mod = utils::negate_uint64_mod(b_mod, plain_modulus);}
            if ((a_mod != 0) && (utils::gcd(a_mod, t) == 1)) {
                uint64_t new_sum = sum_abs(a_mod, b_mod);
                if (new_sum < sum) {
                    e1 = a_mod;
                    e2 = b_mod;
                    sum = new_sum;
                }
            }
        }
        prod = utils::multiply_uint64_mod(e1, factor1, plain_modulus);
    }

    static bool is_scale_within_bounds(double scale, ContextDataPointer context_data) {
        SchemeType scheme = context_data->parms().scheme();
        int scale_bit_count_bound = -1;
        switch (scheme) {
            case SchemeType::BFV: case SchemeType::BGV: {
                scale_bit_count_bound = static_cast<int>(context_data->parms().plain_modulus_host().bit_count());
                break;
            }
            case SchemeType::CKKS: {
                scale_bit_count_bound = static_cast<int>(context_data->total_coeff_modulus_bit_count());
                break;
            }
        }
        // std::cerr << static_cast<int>(std::log2(scale)) << " " << scale_bit_count_bound << std::endl;
        return !(scale <= 0.0 || static_cast<int>(std::log2(scale)) >= scale_bit_count_bound);
    }

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

    void Evaluator::translate_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, bool subtract) const {
        check_ciphertext("[Evaluator::translate_inplace]", encrypted1);
        check_ciphertext("[Evaluator::translate_inplace]", encrypted2);
        check_same_parms_id("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        check_same_scale("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        check_same_ntt_form("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::translate_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t enc1_size = encrypted1.polynomial_count();
        size_t enc2_size = encrypted2.polynomial_count();
        size_t max_size = std::max(enc1_size, enc2_size);
        size_t min_size = std::min(enc1_size, enc2_size);
        size_t coeff_count = parms.poly_modulus_degree();

        if (encrypted1.correction_factor() != 1 || encrypted2.correction_factor() != 1) {
            // Balance correction factors and multiply by scalars before addition in BGV
            uint64_t f0, f1, f2;
            const Modulus& plain_modulus = parms.plain_modulus_host();
            balance_correction_factors(
                encrypted1.correction_factor(), encrypted2.correction_factor(),
                plain_modulus, f0, f1, f2
            );
            utils::multiply_scalar_inplace_ps(encrypted1.data().reference(), f1, enc1_size, coeff_count, coeff_modulus);
            Ciphertext encrypted2_copy = encrypted2;
            utils::multiply_scalar_inplace_ps(encrypted2_copy.data().reference(), f2, enc2_size, coeff_count, coeff_modulus); 
            // Set new correction factor
            encrypted1.correction_factor() = f0;
            encrypted2_copy.correction_factor() = f0;
            this->translate_inplace(encrypted1, encrypted2_copy, subtract);
        } else {
            // Prepare destination
            encrypted1.resize(this->context(), context_data->parms_id(), max_size);
            if (!subtract) {
                utils::add_inplace_ps(encrypted1.data().reference(), encrypted2.data().const_reference(), min_size, coeff_count, coeff_modulus);
            } else {
                utils::sub_inplace_ps(encrypted1.data().reference(), encrypted2.data().const_reference(), min_size, coeff_count, coeff_modulus);
            }
            // Copy the remainding polys of the array with larger count into encrypted1
            if (enc1_size < enc2_size) {
                if (!subtract) {
                    encrypted1.polys(enc1_size, enc2_size).copy_from_slice(encrypted2.polys(enc1_size, enc2_size));
                } else {
                    utils::negate_ps(encrypted2.polys(enc1_size, enc2_size), enc2_size - enc1_size, coeff_count, coeff_modulus, encrypted1.polys(enc1_size, enc2_size));
                }
            }
        }
    }

    void Evaluator::bfv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
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
        ConstSlice<Modulus> base_Bsk_m_tilde = rns_tool.base_Bsk_m_tilde().base();
        size_t base_Bsk_m_tilde_size = base_Bsk_m_tilde.size();
        
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
        encrypted1.resize(this->context(), context_data->parms_id(), dest_size);
        // Allocate space for a base q output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted1_q(encrypted1_size, base_q_size, coeff_count, device);
        // Allocate space for a base Bsk output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted1_Bsk(encrypted1_size, base_Bsk_size, coeff_count, device);

        // Perform BEHZ steps (1)-(3) for encrypted1
        // Make copy of input polynomial (in base q) and convert to NTT form
        encrypted1_q.copy_from_slice(encrypted1.const_polys(0, encrypted1_size));
        // Lazy reduction
        utils::ntt_negacyclic_harvey_lazy_ps(encrypted1_q.reference(), encrypted1_size, coeff_count, base_q_ntt_tables);
        // Allocate temporary space for a polynomial in the Bsk U {m_tilde} base
        Buffer<uint64_t> temp(base_Bsk_m_tilde_size, coeff_count, device);
        for (size_t i = 0; i < encrypted1_size; i++) {
            // (1) Convert from base q to base Bsk U {m_tilde}
            rns_tool.fast_b_conv_m_tilde(encrypted1.const_poly(i), temp.reference());
            // (2) Reduce q-overflows in with Montgomery reduction, switching base to Bsk
            rns_tool.sm_mrq(temp.const_reference(), encrypted1_Bsk.poly(i));
        }
        // Transform to NTT form in base Bsk
        utils::ntt_negacyclic_harvey_lazy_ps(encrypted1_Bsk.reference(), encrypted1_size, coeff_count, base_Bsk_ntt_tables);

        // Repeat for encrypted2
        Buffer<uint64_t> encrypted2_q(encrypted2_size, base_q_size, coeff_count, device);
        Buffer<uint64_t> encrypted2_Bsk(encrypted2_size, base_Bsk_size, coeff_count, device);
        encrypted2_q.copy_from_slice(encrypted2.polys(0, encrypted2_size));
        utils::ntt_negacyclic_harvey_lazy_ps(encrypted2_q.reference(), encrypted2_size, coeff_count, base_q_ntt_tables);
        for (size_t i = 0; i < encrypted2_size; i++) {
            rns_tool.fast_b_conv_m_tilde(encrypted2.poly(i), temp.reference());
            rns_tool.sm_mrq(temp.const_reference(), encrypted2_Bsk.poly(i));
        }
        utils::ntt_negacyclic_harvey_lazy_ps(encrypted2_Bsk.reference(), encrypted2_size, coeff_count, base_Bsk_ntt_tables);

        // Allocate temporary space for the output of step (4)
        // We allocate space separately for the base q and the base Bsk components
        Buffer<uint64_t> temp_dest_q(dest_size, base_q_size, coeff_count, device);
        Buffer<uint64_t> temp_dest_Bsk(dest_size, base_Bsk_size, coeff_count, device);

        // Perform BEHZ step (4): dyadic multiplication on arbitrary size ciphertexts
        Buffer<uint64_t> temp1(base_q_size, coeff_count, device);
        Buffer<uint64_t> temp2(base_Bsk_size, coeff_count, device);
        for (size_t i = 0; i < dest_size; i++) {
            // We iterate over relevant components of encrypted1 and encrypted2 in increasing order for
            // encrypted1 and reversed (decreasing) order for encrypted2. The bounds for the indices of
            // the relevant terms are obtained as follows.
            size_t curr_encrypted1_last = std::min(i, encrypted1_size - 1);
            size_t curr_encrypted2_first = std::min(i, encrypted2_size - 1);
            size_t curr_encrypted1_first = i - curr_encrypted2_first;
            size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

            // Perform the BEHZ ciphertext product both for base q and base Bsk
            for (size_t j = 0; j < steps; j++) {
                utils::dyadic_product_p(
                    encrypted1_q.const_poly(curr_encrypted1_first + j),
                    encrypted2_q.const_poly(curr_encrypted2_first - j),
                    coeff_count,
                    base_q,
                    temp1.reference()
                );
                utils::add_inplace_p(
                    temp_dest_q.poly(i),
                    temp1.const_reference(),
                    coeff_count,
                    base_q
                );
            }
            for (size_t j = 0; j < steps; j++) {
                utils::dyadic_product_p(
                    encrypted1_Bsk.const_poly(curr_encrypted1_first + j),
                    encrypted2_Bsk.const_poly(curr_encrypted2_first - j),
                    coeff_count,
                    base_Bsk,
                    temp2.reference()
                );
                utils::add_inplace_p(
                    temp_dest_Bsk.poly(i),
                    temp2.const_reference(),
                    coeff_count,
                    base_Bsk
                );
            }
        }

        // Perform BEHZ step (5): transform data from NTT form
        // Lazy reduction here. The following multiplyPolyScalarCoeffmod will correct the value back to [0, p)
        utils::inverse_ntt_negacyclic_harvey_ps(temp_dest_q.reference(), dest_size, coeff_count, base_q_ntt_tables);
        utils::inverse_ntt_negacyclic_harvey_ps(temp_dest_Bsk.reference(), dest_size, coeff_count, base_Bsk_ntt_tables);

        // Perform BEHZ steps (6)-(8)
        Buffer<uint64_t> temp_q_Bsk(base_q_size + base_Bsk_size, coeff_count, device);
        Buffer<uint64_t> temp_Bsk(base_Bsk_size, coeff_count, device);
        uint64_t plain_modulus_value = parms.plain_modulus_host().value();
        for (size_t i = 0; i < dest_size; i++) {
            // Bring together the base q and base Bsk components into a single allocation
            // Step (6): multiply base q components by t (plain_modulus)
            utils::multiply_scalar_p(
                temp_dest_q.const_slice(i*coeff_count*base_q_size, (i+1)*coeff_count*base_q_size),
                plain_modulus_value,
                coeff_count,
                base_q,
                temp_q_Bsk.components(0, base_q_size)
            );
            utils::multiply_scalar_p(
                temp_dest_Bsk.const_slice(i*coeff_count*base_Bsk_size, (i+1)*coeff_count*base_Bsk_size),
                plain_modulus_value,
                coeff_count,
                base_Bsk,
                temp_q_Bsk.components(base_q_size, base_q_size + base_Bsk_size)
            );
            // Step (7): divide by q and floor, producing a result in base Bsk
            rns_tool.fast_floor(temp_q_Bsk.const_reference(), temp_Bsk.reference());
            // Step (8): use Shenoy-Kumaresan method to convert the result to base q and write to encrypted1
            rns_tool.fast_b_conv_sk(temp_Bsk.const_reference(), encrypted1.poly(i));
        }
    }
    
    void Evaluator::ckks_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
        check_is_ntt_form("[Evaluator::ckks_multiply_inplace]", encrypted1);
        check_is_ntt_form("[Evaluator::ckks_multiply_inplace]", encrypted2);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::ckks_multiply_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.polynomial_count();
        size_t encrypted2_size = encrypted2.polynomial_count();
        
        // Determine destination.size()
        size_t dest_size = encrypted1_size + encrypted2_size - 1;

        encrypted1.resize(this->context(), context_data->parms_id(), dest_size);
        bool device = encrypted1.on_device();
        Buffer<uint64_t> temp(dest_size, coeff_modulus_size, coeff_count, device);

        Buffer<uint64_t> prod(coeff_modulus_size, coeff_count, device);
        for (size_t i = 0; i < dest_size; i++) {
            // We iterate over relevant components of encrypted1 and encrypted2 in increasing order for
            // encrypted1 and reversed (decreasing) order for encrypted2. The bounds for the indices of
            // the relevant terms are obtained as follows.
            size_t curr_encrypted1_last = std::min(i, encrypted1_size - 1);
            size_t curr_encrypted2_first = std::min(i, encrypted2_size - 1);
            size_t curr_encrypted1_first = i - curr_encrypted2_first;
            // let curr_encrypted2_last = i - curr_encrypted1_last;
            size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

            for (size_t j = 0; j < steps; j++) {
                utils::dyadic_product_p(
                    encrypted1.const_poly(curr_encrypted1_first + j),
                    encrypted2.const_poly(curr_encrypted2_first - j),
                    coeff_count,
                    coeff_modulus,
                    prod.reference()
                );
                utils::add_inplace_p(
                    temp.poly(i),
                    prod.const_reference(),
                    coeff_count,
                    coeff_modulus
                );
            }
        }

        encrypted1.polys(0, dest_size).copy_from_slice(temp.const_reference());
        encrypted1.scale() = encrypted1.scale() * encrypted2.scale();
        if (!is_scale_within_bounds(encrypted1.scale(), context_data)) {
            throw std::invalid_argument("[Evaluator::ckks_multiply_inplace] Scale out of bounds");
        }
    }
    
    void Evaluator::bgv_multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
        check_is_not_ntt_form("[Evaluator::bgv_multiply_inplace]", encrypted1);
        check_is_not_ntt_form("[Evaluator::bgv_multiply_inplace]", encrypted2);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::bgv_multiply_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.polynomial_count();
        size_t encrypted2_size = encrypted2.polynomial_count();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        
        // Determine destination.size()
        size_t dest_size = encrypted1_size + encrypted2_size - 1;

        encrypted1.resize(this->context(), context_data->parms_id(), dest_size);
        bool device = encrypted1.on_device();

        utils::ntt_negacyclic_harvey_ps(
            encrypted1.polys(0, encrypted1_size), 
            encrypted1_size, coeff_count, ntt_tables
        );
        Ciphertext encrypted2_copy = encrypted2;
        utils::ntt_negacyclic_harvey_ps(
            encrypted2_copy.polys(0, encrypted2_size), 
            encrypted2_size, coeff_count, ntt_tables
        );
        Buffer<uint64_t> temp(dest_size, coeff_modulus_size, coeff_count, device);

        Buffer<uint64_t> prod(coeff_modulus_size, coeff_count, device);
        for (size_t i = 0; i < dest_size; i++) {
            // We iterate over relevant components of encrypted1 and encrypted2 in increasing order for
            // encrypted1 and reversed (decreasing) order for encrypted2. The bounds for the indices of
            // the relevant terms are obtained as follows.
            size_t curr_encrypted1_last = std::min(i, encrypted1_size - 1);
            size_t curr_encrypted2_first = std::min(i, encrypted2_size - 1);
            size_t curr_encrypted1_first = i - curr_encrypted2_first;
            // let curr_encrypted2_last = i - curr_encrypted1_last;
            size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

            for (size_t j = 0; j < steps; j++) {
                utils::dyadic_product_p(
                    encrypted1.const_poly(curr_encrypted1_first + j),
                    encrypted2_copy.const_poly(curr_encrypted2_first - j),
                    coeff_count,
                    coeff_modulus,
                    prod.reference()
                );
                utils::add_inplace_p(
                    temp.poly(i),
                    prod.const_reference(),
                    coeff_count,
                    coeff_modulus
                );
            }
        }
        
        encrypted1.polys(0, dest_size).copy_from_slice(temp.const_reference());
        utils::inverse_ntt_negacyclic_harvey_ps(
            encrypted1.polys(0, dest_size), 
            dest_size, coeff_count, ntt_tables
        );
        encrypted1.correction_factor() = utils::multiply_uint64_mod(
            encrypted1.correction_factor(),
            encrypted2.correction_factor(),
            parms.plain_modulus_host()
        );
    }

    void Evaluator::multiply_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2) const {
        check_no_seed("[Evaluator::multiply_inplace]", encrypted1);
        check_no_seed("[Evaluator::multiply_inplace]", encrypted2);
        check_same_parms_id("[Evaluator::multiply_inplace]", encrypted1, encrypted2);
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                this->bfv_multiply_inplace(encrypted1, encrypted2);
                break;
            }
            case SchemeType::CKKS: {
                this->ckks_multiply_inplace(encrypted1, encrypted2);
                break;
            }
            case SchemeType::BGV: {
                this->bgv_multiply_inplace(encrypted1, encrypted2);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::multiply_inplace] Scheme not implemented.");
            }
        }
    }

    void Evaluator::bfv_square_inplace(Ciphertext& encrypted) const {
        check_is_not_ntt_form("[Evaluator::bfv_square_inplace]", encrypted);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::bfv_square_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> base_q = parms.coeff_modulus();
        size_t base_q_size = base_q.size();
        size_t encrypted_size = encrypted.polynomial_count();

        if (encrypted_size != 2) {
            this->bfv_multiply_inplace(encrypted, encrypted);
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
        encrypted.resize(this->context(), context_data->parms_id(), dest_size);
        // Allocate space for a base q output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted_q(encrypted_size, base_q_size, coeff_count, device);
        // Allocate space for a base Bsk output of behz_extend_base_convertToNtt for encrypted1
        Buffer<uint64_t> encrypted_Bsk(encrypted_size, base_Bsk_size, coeff_count, device);

        // Perform BEHZ steps (1)-(3) for encrypted1
        // Make copy of input polynomial (in base q) and convert to NTT form
        encrypted_q.copy_from_slice(encrypted.const_polys(0, encrypted_size));
        // Lazy reduction
        utils::ntt_negacyclic_harvey_lazy_ps(encrypted_q.reference(), encrypted_size, coeff_count, base_q_ntt_tables);
        // Allocate temporary space for a polynomial in the Bsk U {m_tilde} base
        Buffer<uint64_t> temp(base_Bsk_m_tilde_size, coeff_count, device);
        for (size_t i = 0; i < encrypted_size; i++) {
            // (1) Convert from base q to base Bsk U {m_tilde}
            rns_tool.fast_b_conv_m_tilde(encrypted.const_poly(i), temp.reference());
            // (2) Reduce q-overflows in with Montgomery reduction, switching base to Bsk
            rns_tool.sm_mrq(temp.const_reference(), encrypted_Bsk.poly(i));
        }
        // Transform to NTT form in base Bsk
        utils::ntt_negacyclic_harvey_lazy_ps(encrypted_Bsk.reference(), encrypted_size, coeff_count, base_Bsk_ntt_tables);

        // Allocate temporary space for the output of step (4)
        // We allocate space separately for the base q and the base Bsk components
        Buffer<uint64_t> temp_dest_q(dest_size, base_q_size, coeff_count, device);
        Buffer<uint64_t> temp_dest_Bsk(dest_size, base_Bsk_size, coeff_count, device);

        // Perform the BEHZ ciphertext square both for base q and base Bsk

        // Compute c0^2
        Slice<uint64_t> eq0 = encrypted_q.poly(0);
        Slice<uint64_t> eq1 = encrypted_q.poly(1);
        utils::dyadic_product_p(eq0.as_const(), eq0.as_const(), coeff_count, base_q, temp_dest_q.poly(0));
        // Compute 2*c0*c1
        utils::dyadic_product_p(eq0.as_const(), eq1.as_const(), coeff_count, base_q, temp_dest_q.poly(1));
        utils::add_inplace_p(temp_dest_q.poly(1), temp_dest_q.const_poly(1), coeff_count, base_q);
        // Compute c1^2
        utils::dyadic_product_p(eq1.as_const(), eq1.as_const(), coeff_count, base_q, temp_dest_q.poly(2));

        Slice<uint64_t> eb0 = encrypted_Bsk.poly(0);
        Slice<uint64_t> eb1 = encrypted_Bsk.poly(1);
        utils::dyadic_product_p(eb0.as_const(), eb0.as_const(), coeff_count, base_Bsk, temp_dest_Bsk.poly(0));
        utils::dyadic_product_p(eb0.as_const(), eb1.as_const(), coeff_count, base_Bsk, temp_dest_Bsk.poly(1));
        utils::add_inplace_p(temp_dest_Bsk.poly(1), temp_dest_Bsk.const_poly(1), coeff_count, base_Bsk);
        utils::dyadic_product_p(eb1.as_const(), eb1.as_const(), coeff_count, base_Bsk, temp_dest_Bsk.poly(2));
        
        // Perform BEHZ step (5): transform data from NTT form
        // Lazy reduction here. The following multiplyPolyScalarCoeffmod will correct the value back to [0, p)
        utils::inverse_ntt_negacyclic_harvey_ps(temp_dest_q.reference(), dest_size, coeff_count, base_q_ntt_tables);
        utils::inverse_ntt_negacyclic_harvey_ps(temp_dest_Bsk.reference(), dest_size, coeff_count, base_Bsk_ntt_tables);

        // Perform BEHZ steps (6)-(8)
        Buffer<uint64_t> temp_q_Bsk(base_q_size + base_Bsk_size, coeff_count, device);
        Buffer<uint64_t> temp_Bsk(base_Bsk_size, coeff_count, device);
        uint64_t plain_modulus_value = parms.plain_modulus_host().value();
        for (size_t i = 0; i < dest_size; i++) {
            // Bring together the base q and base Bsk components into a single allocation
            // Step (6): multiply base q components by t (plain_modulus)
            utils::multiply_scalar_p(
                temp_dest_q.const_slice(i*coeff_count*base_q_size, (i+1)*coeff_count*base_q_size),
                plain_modulus_value,
                coeff_count,
                base_q,
                temp_q_Bsk.components(0, base_q_size)
            );
            utils::multiply_scalar_p(
                temp_dest_Bsk.const_slice(i*coeff_count*base_Bsk_size, (i+1)*coeff_count*base_Bsk_size),
                plain_modulus_value,
                coeff_count,
                base_Bsk,
                temp_q_Bsk.components(base_q_size, base_q_size + base_Bsk_size)
            );
            // Step (7): divide by q and floor, producing a result in base Bsk
            rns_tool.fast_floor(temp_q_Bsk.const_reference(), temp_Bsk.reference());
            // Step (8): use Shenoy-Kumaresan method to convert the result to base q and write to encrypted1
            rns_tool.fast_b_conv_sk(temp_Bsk.const_reference(), encrypted.poly(i));
        }
    }

    void Evaluator::ckks_square_inplace(Ciphertext& encrypted) const {
        check_is_ntt_form("[Evaluator::ckks_square_inplace]", encrypted);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::ckks_square_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.polynomial_count();

        if (encrypted_size != 2) {
            this->ckks_multiply_inplace(encrypted, encrypted);
            return;
        }
        
        // Determine destination.size()
        size_t dest_size = 2 * encrypted_size - 1;

        encrypted.resize(this->context(), context_data->parms_id(), dest_size);
        bool device = encrypted.on_device();
        
        Slice<uint64_t> c0 = encrypted.poly(0);
        Slice<uint64_t> c1 = encrypted.poly(1);
        Slice<uint64_t> c2 = encrypted.poly(2);
        
        utils::dyadic_product_p(c1.as_const(), c1.as_const(), coeff_count, coeff_modulus, c2);
        utils::dyadic_product_p(c0.as_const(), c1.as_const(), coeff_count, coeff_modulus, c1);
        utils::add_inplace_p(   c1,            c1.as_const(), coeff_count, coeff_modulus);
        utils::dyadic_product_p(c0.as_const(), c0.as_const(), coeff_count, coeff_modulus, c0);

        encrypted.scale() = encrypted.scale() * encrypted.scale();
        if (!is_scale_within_bounds(encrypted.scale(), context_data)) {
            throw std::invalid_argument("[Evaluator::ckks_multiply_inplace] Scale out of bounds");
        }
    }
    
    void Evaluator::bgv_square_inplace(Ciphertext& encrypted) const {
        check_is_not_ntt_form("[Evaluator::bgv_square_inplace]", encrypted);
        
        // Extract encryption parameters.
        ContextDataPointer context_data = this->get_context_data("[Evaluator::bgv_square_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.polynomial_count();

        if (encrypted_size != 2) {
            this->bgv_multiply_inplace(encrypted, encrypted);
            return;
        }
        
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        
        // Determine destination.size()
        size_t dest_size = 2 * encrypted_size - 1;

        encrypted.resize(this->context(), context_data->parms_id(), dest_size);
        bool device = encrypted.on_device();

        utils::ntt_negacyclic_harvey_ps(
            encrypted.polys(0, encrypted_size), 
            encrypted_size, coeff_count, ntt_tables
        );
        Buffer<uint64_t> temp(dest_size, coeff_modulus_size, coeff_count, device);

        ConstSlice<uint64_t> eq0 = encrypted.const_poly(0);
        ConstSlice<uint64_t> eq1 = encrypted.const_poly(1);
        Slice<uint64_t> tq0 = temp.poly(0);
        Slice<uint64_t> tq1 = temp.poly(1);
        Slice<uint64_t> tq2 = temp.poly(2);
        
        utils::dyadic_product_p(eq0, eq0, coeff_count, coeff_modulus, tq0);
        // Compute 2*c0*c1
        utils::dyadic_product_p(eq0, eq1, coeff_count, coeff_modulus, tq1);
        utils::add_inplace_p(tq1, tq1.as_const(), coeff_count, coeff_modulus);
        // Compute c1^2
        utils::dyadic_product_p(eq1, eq1, coeff_count, coeff_modulus, tq2);

        encrypted.polys(0, dest_size).copy_from_slice(temp.const_reference());
        utils::inverse_ntt_negacyclic_harvey_ps(
            encrypted.polys(0, dest_size), 
            dest_size, coeff_count, ntt_tables
        );
        encrypted.correction_factor() = utils::multiply_uint64_mod(
            encrypted.correction_factor(),
            encrypted.correction_factor(),
            parms.plain_modulus_host()
        );
    }

    void Evaluator::square_inplace(Ciphertext& encrypted) const {
        check_no_seed("[Evaluator::square_inplace]", encrypted);
        SchemeType scheme = this->context()->first_context_data().value()->parms().scheme();
        switch (scheme) {
            case SchemeType::BFV: {
                this->bfv_square_inplace(encrypted);
                break;
            }
            case SchemeType::CKKS: {
                this->ckks_square_inplace(encrypted);
                break;
            }
            case SchemeType::BGV: {
                this->bgv_square_inplace(encrypted);
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
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, true);
        for (size_t k = 0; k < key_component_count; k++) {
            utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
            size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
            Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
            utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
            accumulator_l[0] = key_modulus->reduce_uint128(qword_slice.as_const());
            accumulator_l[1] = 0;
        }
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
            uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, false);
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t k = 0; k < key_component_count; k++) {
                    utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
                    size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
                    Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
                    utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
                    accumulator_l[0] = key_modulus->reduce_uint128(qword_slice.as_const());
                    accumulator_l[1] = 0;
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            kernel_ski_util1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                key_vector_j, key_poly_coeff_size, t_operand, key_index, key_modulus
            );
        }
    }
    
    __global__ static void kernel_ski_util2(
        Slice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        ConstSlice<uint64_t> key_vector_j,
        size_t key_poly_coeff_size,
        ConstSlice<uint64_t> t_operand,
        size_t key_index,
        ConstPointer<Modulus> key_modulus
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, true);
        for (size_t k = 0; k < key_component_count; k++) {
            utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
            size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
            Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
            utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
            accumulator_l[0] = qword_slice[0];
            accumulator_l[1] = qword_slice[1];
        }
    }

    static void ski_util2(
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
            uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, false);
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
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            kernel_ski_util2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                key_vector_j, key_poly_coeff_size, t_operand, key_index, key_modulus
            );
        }
    }

    __global__ static void kernel_ski_util3(
        ConstSlice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        size_t rns_modulus_size,
        Slice<uint64_t> t_poly_prod_iter
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        for (size_t k = 0; k < key_component_count; k++) {
            size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
            t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = t_poly_lazy[accumulator_l_offset];
        }
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
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            kernel_ski_util3<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, rns_modulus_size, t_poly_prod_iter
            );
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
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        for (size_t k = 0; k < key_component_count; k++) {
            size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
            t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = key_modulus->reduce_uint128(
                t_poly_lazy.const_slice(accumulator_l_offset, accumulator_l_offset + 2)
            );
        }
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
                    t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = key_modulus->reduce_uint128(
                        t_poly_lazy.const_slice(accumulator_l_offset, accumulator_l_offset + 2)
                    );
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            kernel_ski_util4<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                rns_modulus_size, t_poly_prod_iter, key_modulus
            );
        }
    }

    __global__ static void kernel_ski_util5(
        ConstSlice<uint64_t> t_last,
        Slice<uint64_t> t_poly_prod_i,
        size_t coeff_count,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        size_t rns_modulus_size,
        uint64_t qk_inv_qp,
        uint64_t qk,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> encrypted_i
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        uint64_t k = utils::barrett_reduce_uint64(t_last[i], *plain_modulus);
        k = utils::negate_uint64_mod(k, *plain_modulus);
        if (qk_inv_qp != 1) 
            k = utils::multiply_uint64_mod(k, qk_inv_qp, *plain_modulus);
        uint64_t delta = 0; uint64_t c_mod_qi = 0;
        for (size_t j = 0; j < decomp_modulus_size; j++) {
            delta = utils::barrett_reduce_uint64(k, key_modulus[j]);
            delta = utils::multiply_uint64_mod(delta, qk, key_modulus[j]);
            c_mod_qi = utils::barrett_reduce_uint64(t_last[i], key_modulus[j]);
            const uint64_t Lqi = key_modulus[j].value() << 1;
            uint64_t& target = t_poly_prod_i[j * coeff_count + i];
            target = target + Lqi - (delta + c_mod_qi);
            target = utils::multiply_uint64operand_mod(target, modswitch_factors[j], key_modulus[j]);
            encrypted_i[j * coeff_count + i] = utils::add_uint64_mod(target, encrypted_i[j * coeff_count + i], key_modulus[j]);
        }
    }

    static void ski_util5(
        ConstSlice<uint64_t> t_last,
        Slice<uint64_t> t_poly_prod_i,
        size_t coeff_count,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        size_t rns_modulus_size,
        uint64_t qk_inv_qp,
        uint64_t qk,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> encrypted_i
    ) {
        bool device = t_last.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_count; i++) {
                uint64_t k = utils::barrett_reduce_uint64(t_last[i], *plain_modulus);
                k = utils::negate_uint64_mod(k, *plain_modulus);
                if (qk_inv_qp != 1) 
                    k = utils::multiply_uint64_mod(k, qk_inv_qp, *plain_modulus);
                uint64_t delta = 0; uint64_t c_mod_qi = 0;
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    delta = utils::barrett_reduce_uint64(k, key_modulus[j]);
                    delta = utils::multiply_uint64_mod(delta, qk, key_modulus[j]);
                    c_mod_qi = utils::barrett_reduce_uint64(t_last[i], key_modulus[j]);
                    const uint64_t Lqi = key_modulus[j].value() << 1;
                    uint64_t& target = t_poly_prod_i[j * coeff_count + i];
                    target = target + Lqi - (delta + c_mod_qi);
                    target = utils::multiply_uint64operand_mod(target, modswitch_factors[j], key_modulus[j]);
                    encrypted_i[j * coeff_count + i] = utils::add_uint64_mod(target, encrypted_i[j * coeff_count + i], key_modulus[j]);
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            kernel_ski_util5<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, t_poly_prod_i, coeff_count, plain_modulus, key_modulus, 
                decomp_modulus_size, rns_modulus_size, qk_inv_qp, qk, modswitch_factors, encrypted_i
            );
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
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
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
            kernel_ski_util6<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, coeff_count, qk, key_modulus, decomp_modulus_size, t_ntt
            );
        }
    }

    __global__ static void kernel_ski_util7(
        Slice<uint64_t> t_poly_prod_i,
        ConstSlice<uint64_t> t_ntt,
        size_t coeff_count, 
        Slice<uint64_t> encrypted_i,
        bool is_ckks,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        for (size_t j = 0; j < decomp_modulus_size; j++) {
            uint64_t& dest = t_poly_prod_i[j*coeff_count + i];
            uint64_t qi = key_modulus[j].value();
            dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - t_ntt[j * coeff_count + i];
            dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
            encrypted_i[j * coeff_count + i] = utils::add_uint64_mod(
                encrypted_i[j * coeff_count + i], dest, key_modulus[j]
            );
        }
    }

    static void ski_util7(
        Slice<uint64_t> t_poly_prod_i,
        ConstSlice<uint64_t> t_ntt,
        size_t coeff_count, 
        Slice<uint64_t> encrypted_i,
        bool is_ckks,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors
    ) {
        bool device = t_poly_prod_i.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    uint64_t& dest = t_poly_prod_i[j*coeff_count + i];
                    uint64_t qi = key_modulus[j].value();
                    dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - t_ntt[j * coeff_count + i];
                    dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
                    encrypted_i[j * coeff_count + i] = utils::add_uint64_mod(
                        encrypted_i[j * coeff_count + i], dest, key_modulus[j]
                    );
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            kernel_ski_util7<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_prod_i, t_ntt, coeff_count, encrypted_i, is_ckks, 
                decomp_modulus_size, key_modulus, modswitch_factors
            );
        }
    }

    void Evaluator::switch_key_inplace_internal(Ciphertext& encrypted, utils::ConstSlice<uint64_t> target, const KSwitchKeys& kswitch_keys, size_t kswitch_keys_index) const {
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

        switch (scheme) {
            case SchemeType::BFV: case SchemeType::BGV: {
                check_is_not_ntt_form("[Evaluator::switch_key_inplace_internal]", encrypted);
                break;
            }
            case SchemeType::CKKS: {
                check_is_ntt_form("[Evaluator::switch_key_inplace_internal]", encrypted);
                break;
            }
            default: {
                throw std::logic_error("[Evaluator::switch_key_inplace_internal] Scheme not implemented.");
            }
        }

        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        ConstSlice<Modulus> key_modulus = key_parms.coeff_modulus();
        Array<Modulus> key_modulus_host = Array<Modulus>::create_and_copy_from_slice(key_modulus);
        key_modulus_host.to_host_inplace();
        size_t key_modulus_size = key_modulus.size();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        ConstSlice<NTTTables> key_ntt_tables = key_context_data->small_ntt_tables();
        ConstSlice<MultiplyUint64Operand> modswitch_factors = key_context_data->rns_tool().inv_q_last_mod_q();

        const std::vector<PublicKey>& key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].as_ciphertext().polynomial_count();
        for (size_t i = 0; i < key_vector.size(); i++) {
            check_no_seed("[Evaluator::switch_key_inplace_internal]", key_vector[i].as_ciphertext());
        }

        if (target.size() != decomp_modulus_size * coeff_count) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Invalid target size.");
        }
        Array<uint64_t> target_copied = Array<uint64_t>::create_and_copy_from_slice(target);

        // In CKKS target is in NTT form; switch back to normal form
        if (scheme == SchemeType::CKKS) {
            utils::inverse_ntt_negacyclic_harvey_p(
                target_copied.reference(), coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size)
            );
        }

        // Temporary result
        bool device = target.on_device();
        Array<uint64_t> poly_prod(key_component_count * coeff_count * rns_modulus_size, device);
        Array<uint64_t> poly_lazy(key_component_count * coeff_count * 2, device);
        Array<uint64_t> temp_ntt(coeff_count, device);
        for (size_t i = 0; i < rns_modulus_size; i++) {
            size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);

            // Product of two numbers is up to 60 + 60 = 120 bits, so we can sum up to 256 of them without reduction.
            size_t lazy_reduction_summand_bound = utils::HE_MULTIPLY_ACCUMULATE_USER_MOD_MAX;
            size_t lazy_reduction_counter = lazy_reduction_summand_bound;

            // Allocate memory for a lazy accumulator (128-bit coefficients)
            poly_lazy.set_zero();

            // Multiply with keys and perform lazy reduction on product's coefficients
            temp_ntt.set_zero();
            for (size_t j = 0; j < decomp_modulus_size; j++) {
                ConstSlice<uint64_t> temp_operand(nullptr, 0, device);
                if ((scheme == SchemeType::CKKS) && (i == j)) {
                    temp_operand = target.const_slice(j * coeff_count, (j + 1) * coeff_count);
                } else {
                    if (key_modulus_host[j].value() <= key_modulus_host[key_index].value()) {
                        temp_ntt.copy_from_slice(target_copied.const_slice(j * coeff_count, (j + 1) * coeff_count));
                    } else {
                        utils::modulo(target_copied.const_slice(j * coeff_count, (j + 1) * coeff_count), key_modulus.at(key_index), temp_ntt.reference());
                    }
                    utils::ntt_negacyclic_harvey_lazy(temp_ntt.reference(), coeff_count, key_ntt_tables.at(key_index));
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
                        temp_operand, key_index, key_modulus.at(key_index)
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
        
        // Accumulated products are now stored in t_poly_prod

        temp_ntt = Array<uint64_t>(decomp_modulus_size * coeff_count, device);
        for (size_t i = 0; i < key_component_count; i++) {
            if (scheme == SchemeType::BGV) {
                // qk is the special prime
                uint64_t qk = key_modulus_host[key_modulus_size - 1].value();
                uint64_t qk_inv_qp = this->context()->key_context_data().value()->rns_tool().inv_q_last_mod_t();

                // Lazy reduction; this needs to be then reduced mod qi
                size_t t_last_offset = coeff_count * rns_modulus_size * i + decomp_modulus_size * coeff_count;
                Slice<uint64_t> t_last = poly_prod.slice(t_last_offset, t_last_offset + coeff_count);
                utils::inverse_ntt_negacyclic_harvey(t_last, coeff_count, key_ntt_tables.at(key_modulus_size - 1));
                utils::inverse_ntt_negacyclic_harvey_p(
                    poly_prod.slice(
                        i * coeff_count * rns_modulus_size, 
                        i * coeff_count * rns_modulus_size + decomp_modulus_size * coeff_count
                    ), 
                    coeff_count, 
                    key_ntt_tables.const_slice(0, decomp_modulus_size)
                );
                ConstPointer<Modulus> plain_modulus = parms.plain_modulus();

                ski_util5(
                    t_last.as_const(), poly_prod.slice(i * coeff_count * rns_modulus_size, poly_prod.size()),
                    coeff_count, plain_modulus, key_modulus,
                    decomp_modulus_size, rns_modulus_size, qk_inv_qp, qk,
                    modswitch_factors, encrypted.poly(i)
                );
            } else {
                // Lazy reduction; this needs to be then reduced mod qi
                size_t t_last_offset = coeff_count * rns_modulus_size * i + decomp_modulus_size * coeff_count;
                Slice<uint64_t> t_last = poly_prod.slice(t_last_offset, t_last_offset + coeff_count);
                temp_ntt.set_zero();
                utils::inverse_ntt_negacyclic_harvey(t_last, coeff_count, key_ntt_tables.at(key_modulus_size - 1));

                ski_util6(
                    t_last, coeff_count, key_modulus.at(key_modulus_size - 1),
                    key_modulus,
                    decomp_modulus_size,
                    temp_ntt.reference()
                );
            
                if (scheme == SchemeType::CKKS) {
                    utils::ntt_negacyclic_harvey_lazy_p(temp_ntt.reference(), coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size));
                } else if (scheme == SchemeType::BFV) {
                    utils::inverse_ntt_negacyclic_harvey_p(
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
                    coeff_count, encrypted.poly(i),
                    scheme==SchemeType::CKKS, decomp_modulus_size, key_modulus,
                    modswitch_factors
                );
            }
            // printf("enc %ld: ", i); printDeviceArray(encrypted.data(i).get(), key_component_count * coeff_count);
        }
    }

    void Evaluator::apply_keyswitching_inplace(Ciphertext& encrypted, const KSwitchKeys& kswitch_keys) const {
        if (kswitch_keys.data().size() != 1) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys size must be 1.");
        }
        if (encrypted.polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Ciphertext polynomial count must be 2.");
        }
        // due to the semantics of `switch_key_inplace_internal`, we should first get the c0 out
        // and then clear the original c0 in the encrypted.
        Array<uint64_t> target = Array<uint64_t>::create_and_copy_from_slice(encrypted.const_poly(1));
        encrypted.poly(1).set_zero();
        this->switch_key_inplace_internal(encrypted, target.const_reference(), kswitch_keys, 0);
    }

}