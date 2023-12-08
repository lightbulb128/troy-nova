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

    template <typename C>
    inline static void check_no_seed(const char* prompt, const C& c) {
        if (c.contains_seed()) {
            throw std::invalid_argument(std::string(prompt) + " Argument contains seed.");
        }
    }

    inline void check_ciphertext(const char* prompt, const Ciphertext& ciphertext) {
        check_no_seed(prompt, ciphertext);
    }

    template <typename C>
    inline static void check_same_parms_id(const char* prompt, const C& a, const C& b) {
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




}