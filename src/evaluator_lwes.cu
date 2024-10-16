#include "batch_utils.h"
#include "ciphertext.h"
#include "context_data.h"
#include "evaluator.h"
#include "evaluator_utils.h"
#include "lwe_ciphertext.h"
#include "utils/constants.h"

namespace troy {
    
    using utils::Slice;
    using utils::ConstSlice;
    using utils::NTTTables;

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

    void Evaluator::field_trace_inplace_batched(const std::vector<Ciphertext*>& encrypted, const GaloisKeys& automorphism_keys, size_t logn, MemoryPoolHandle pool) const {
        if (!this->on_device() || encrypted.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                this->field_trace_inplace(*encrypted[i], automorphism_keys, logn, pool);
            }
            return;
        }
        size_t poly_degree = encrypted[0]->poly_modulus_degree();
        for (size_t i = 0; i < encrypted.size(); i++) {
            if (encrypted[i]->poly_modulus_degree() != poly_degree) {
                throw std::invalid_argument("[Evaluator::field_trace_inplace_batched] Mismatched poly_modulus_degree.");
            }
        }
        std::vector<Ciphertext> temp(encrypted.size());
        auto temp_ptrs = batch_utils::collect_pointer(temp);
        auto temp_const_ptrs = batch_utils::collect_const_pointer(temp);
        auto encrypted_const_ptrs = batch_utils::pcollect_const_pointer(encrypted);
        while (poly_degree > (static_cast<size_t>(1) << logn)) {
            size_t galois_element = poly_degree + 1;
            this->apply_galois_batched(encrypted_const_ptrs, galois_element, automorphism_keys, temp_ptrs, pool);
            this->add_inplace_batched(encrypted, temp_const_ptrs, pool);
            poly_degree >>= 1;
        }
    }
    
    void Evaluator::divide_by_poly_modulus_degree_inplace(Ciphertext& encrypted, uint64_t mul) const {
        ContextDataPointer context_data = this->get_context_data("[Evaluator::divide_by_poly_modulus_degree_inplace]", encrypted.parms_id());
        size_t size = encrypted.polynomial_count();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        size_t n = context_data->parms().poly_modulus_degree();
        size_t logn = static_cast<size_t>(utils::get_power_of_two(n));
        utils::ntt_multiply_inv_degree(
            encrypted.polys(0, size), size, logn, ntt_tables, mul
        );
    }

    void Evaluator::divide_by_poly_modulus_degree_inplace_batched(const std::vector<Ciphertext*>& encrypted, uint64_t mul, MemoryPoolHandle pool) const {
        auto parms_id = get_vec_parms_id(encrypted);
        auto size = get_vec_polynomial_count(encrypted);
        auto context_data = this->get_context_data("[Evaluator::divide_by_poly_modulus_degree_inplace_batched]", parms_id);
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        size_t n = context_data->parms().poly_modulus_degree();
        size_t logn = static_cast<size_t>(utils::get_power_of_two(n));
        utils::ntt_multiply_inv_degree_batched(
            batch_utils::pcollect_reference(encrypted), size, logn, ntt_tables, mul, pool
        );
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

    
    void Evaluator::negacyclic_shift_batched(
        const std::vector<const Ciphertext*>& encrypted, size_t shift, 
        const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool
    ) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::negacyclic_shift_batched] Size mismatch.");
        }
        if (encrypted.size() == 0) return;
        check_no_seed_vec("[Evaluator::negacyclic_shift_batched]", encrypted);
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negacyclic_shift]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();

        for (size_t i = 0; i < encrypted.size(); i++) 
            *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        auto encrypted_poly_count = get_vec_polynomial_count(encrypted);
        auto encrypted_polys = batch_utils::pcollect_const_polys(encrypted, 0, encrypted_poly_count);
        auto destination_polys = batch_utils::pcollect_polys(destination, 0, encrypted_poly_count);

        utils::negacyclic_shift_bps(
            encrypted_polys,
            shift, encrypted_poly_count, coeff_count, coeff_modulus, 
            destination_polys, pool
        );
    }


    Ciphertext Evaluator::pack_lwe_ciphertexts_new(const std::vector<const LWECiphertext*>& lwes, const GaloisKeys& automorphism_keys, MemoryPoolHandle pool, bool apply_field_trace) const {
        size_t lwes_count = lwes.size();
        if (lwes_count == 0) {
            throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must not be empty.");
        }
        ParmsID lwe_parms_id = lwes[0]->parms_id();
        // check all have same parms id
        for (size_t i = 1; i < lwes_count; i++) {
            if (lwes[i]->parms_id() != lwe_parms_id) {
                throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must have same parms id.");
            }
        }

        ContextDataPointer context_data = this->get_context_data("[Evaluator::pack_lwe_ciphertexts_new]", lwe_parms_id);
        size_t poly_modulus_degree = context_data->parms().poly_modulus_degree();
        if (lwes_count > poly_modulus_degree) {
            throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts count must be less than poly_modulus_degree.");
        }

        size_t l = 0;
        while ((static_cast<size_t>(1) << l) < lwes_count) l += 1;

        std::vector<Ciphertext> rlwes = LWECiphertext::assemble_lwe_batched_new(lwes, pool);

        return this->pack_rlwe_ciphertexts_new(
            batch_utils::collect_const_pointer(rlwes), 
            automorphism_keys, 
            0, poly_modulus_degree, poly_modulus_degree / (static_cast<size_t>(1) << l), pool, apply_field_trace);

    }

    void Evaluator::pack_lwe_ciphertexts_batched(const std::vector<std::vector<const LWECiphertext*>>& lwes_groups, 
            const GaloisKeys& automorphism_keys, const std::vector<Ciphertext*>& output, 
            MemoryPoolHandle pool, bool apply_field_trace
    ) const {
        if (lwes_groups.size() != output.size()) {
            throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_batched] Input groups and outputs should have same size.");
        }
        if (lwes_groups.size() == 0) return;
        
        size_t max_lwes_count = 0;
        size_t total_lwes_count = 0;
        ParmsID parms_id;
        bool parms_id_set = false;
        for (const std::vector<const LWECiphertext*>& lwes: lwes_groups) {
            size_t lwes_count = lwes.size();
            max_lwes_count = std::max(max_lwes_count, lwes_count);
            total_lwes_count += lwes_count;
            if (lwes_count == 0) {
                throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must not be empty.");
            }
            ParmsID lwe_parms_id = lwes[0]->parms_id();
            if (!parms_id_set) {
                parms_id = lwe_parms_id;
                parms_id_set = true;
            }
            // check all have same parms id
            for (size_t i = 0; i < lwes_count; i++) {
                if (parms_id_set && lwes[i]->parms_id() != parms_id) {
                    throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts must have same parms id.");
                }
            }

            ContextDataPointer context_data = this->get_context_data("[Evaluator::pack_lwe_ciphertexts_new]", lwe_parms_id);
            size_t poly_modulus_degree = context_data->parms().poly_modulus_degree();
            if (lwes_count > poly_modulus_degree) {
                throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts count must be less than poly_modulus_degree.");
            }
        }

        ContextDataPointer context_data = this->get_context_data("[Evaluator::pack_lwe_ciphertexts_new]", parms_id);
        size_t poly_modulus_degree = context_data->parms().poly_modulus_degree();

        size_t l = 0;
        while ((static_cast<size_t>(1) << l) < max_lwes_count) l += 1;

        std::vector<std::vector<Ciphertext>> rlwes(lwes_groups.size());
        std::vector<const LWECiphertext*> lwes_flattened; lwes_flattened.reserve(total_lwes_count);
        std::vector<Ciphertext*> rlwes_flattened; rlwes_flattened.reserve(total_lwes_count);
        std::vector<std::vector<const Ciphertext*>> rlwes_const_ptrs(lwes_groups.size());
        for (size_t i = 0; i < lwes_groups.size(); i++) {
            rlwes[i] = std::vector<Ciphertext>(lwes_groups[i].size());
            for (size_t j = 0; j < lwes_groups[i].size(); j++) {
                lwes_flattened.push_back(lwes_groups[i][j]);
                rlwes_flattened.push_back(&rlwes[i][j]);
            }
            rlwes_const_ptrs[i] = batch_utils::collect_const_pointer(rlwes[i]);
        }
        LWECiphertext::assemble_lwe_batched(lwes_flattened, rlwes_flattened, pool);

        this->pack_rlwe_ciphertexts_batched(
            rlwes_const_ptrs, 
            automorphism_keys, 
            0, poly_modulus_degree, poly_modulus_degree / (static_cast<size_t>(1) << l), 
            output, pool, apply_field_trace);

    }


    std::pair<size_t, bool> is_power_of_two(uint64_t r) {
        auto p = utils::get_power_of_two(r);
        return std::make_pair(p, r == (static_cast<uint64_t>(1) << p));
    }

    Ciphertext Evaluator::pack_rlwe_ciphertexts_new(
        const std::vector<const Ciphertext*>& ciphers, const GaloisKeys& automorphism_keys, size_t shift, size_t input_interval, size_t output_interval,
        MemoryPoolHandle pool,
        bool apply_field_trace
    ) const {
        ParmsID parms_id = get_vec_parms_id(ciphers);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::pack_rlwe_ciphertexts_new]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        size_t poly_modulus_degree = parms.poly_modulus_degree();
        bool output_ntt_form = scheme == SchemeType::CKKS || scheme == SchemeType::BGV;
        if (scheme == SchemeType::CKKS) {
            get_vec_scale(ciphers); // check all have same scale
        }
        if (scheme == SchemeType::BGV) {
            get_vec_correction_factor(ciphers); // check all have same correction factor
        }
        bool input_ntt_form = get_is_ntt_form_vec(ciphers);
        
        if (input_interval > poly_modulus_degree) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_new] input_interval must be less than poly_modulus_degree.");
        }
        if (output_interval > input_interval) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_new] output_interval must be less than input_interval.");
        }
        if (!is_power_of_two(input_interval).second) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_new] input_interval must be power of two.");
        }
        if (!is_power_of_two(output_interval).second) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_new] output_interval must be power of two.");
        }
        size_t max_cipher_count = input_interval / output_interval; 
        if (ciphers.size() > max_cipher_count) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_new] ciphers count must be less than input_interval / output_interval.");
        }
        size_t layers_required = is_power_of_two(max_cipher_count).first;
        bool device = this->on_device();
        auto coeff_modulus = parms.coeff_modulus();

        std::vector<Ciphertext> rlwes(max_cipher_count);
        std::vector<Slice<uint64_t>> rlwes_slice(ciphers.size(), Slice<uint64_t>(nullptr, 0, false, nullptr));
        std::vector<Ciphertext*> rlwes_ptr; rlwes_ptr.reserve(ciphers.size());
        for (size_t i = 0; i < max_cipher_count; i++) {
            size_t index = static_cast<size_t>(utils::reverse_bits_uint64(static_cast<uint64_t>(i), layers_required));
            if (index < ciphers.size()) {
                rlwes[i] = Ciphertext::like(*ciphers[index], false, pool);
                rlwes_slice[index] = rlwes[i].reference();
                rlwes_ptr.push_back(&rlwes[i]);
            } else {
                rlwes[i] = Ciphertext(); 
                if (device) rlwes[i].to_device_inplace(pool);
            }
        }
        utils::copy_slice_b(batch_utils::pcollect_const_reference(ciphers), rlwes_slice, pool);

        if (input_ntt_form) {
            this->transform_from_ntt_inplace_batched(rlwes_ptr, pool);
        }
        this->divide_by_poly_modulus_degree_inplace_batched(rlwes_ptr, poly_modulus_degree / input_interval, pool);
        if (shift != 0) {
            this->negacyclic_shift_inplace_batched(rlwes_ptr, shift, pool);
        }
        
        if (!device) { // For host, we save some computation when the source even/odd is empty
            Ciphertext temp = Ciphertext::like(rlwes[0], false, pool);
            for (size_t layer = 0; layer < layers_required; layer++) {
                size_t gap = 1 << layer;
                size_t shift = input_interval >> (layer + 1);
                for (size_t offset = 0; offset < max_cipher_count; offset += gap * 2) {
                    Ciphertext& even = rlwes[offset];
                    Ciphertext& odd = rlwes[offset + gap];
                    bool even_empty = even.polynomial_count() == 0;
                    bool odd_empty = odd.polynomial_count() == 0;
                    if (odd_empty && even_empty) {
                        even = Ciphertext(); if (device) even.to_device_inplace(pool); continue;
                    }
                    if (!odd_empty) {
                        utils::negacyclic_shift_ps(
                            odd.const_reference(), shift, odd.polynomial_count(), 
                            poly_modulus_degree, coeff_modulus, temp.reference()
                        );
                    }
                    size_t galois_element = (poly_modulus_degree / input_interval) * (1 << (layer + 1)) + 1;
                    if (!even_empty) {
                        if (!odd_empty) {
                            this->sub(even, temp, odd, pool);
                            this->add_inplace(even, temp, pool);
                            if (output_ntt_form) this->transform_to_ntt_inplace(odd);
                            this->apply_galois_inplace(odd, galois_element, automorphism_keys, pool);
                            if (output_ntt_form) this->transform_from_ntt_inplace(odd);
                            this->add_inplace(even, odd, pool);
                        } else {
                            if (output_ntt_form) {
                                this->transform_to_ntt(even, temp, pool);
                                this->apply_galois_inplace(temp, galois_element, automorphism_keys, pool);
                                this->transform_from_ntt_inplace(temp);
                            } else {
                                this->apply_galois(even, galois_element, automorphism_keys, temp, pool);
                            }
                            this->add_inplace(even, temp, pool);
                        }
                    } else {
                        this->negate(temp, even, pool);
                        if (output_ntt_form) this->transform_to_ntt_inplace(even);
                        this->apply_galois_inplace(even, galois_element, automorphism_keys, pool);
                        if (output_ntt_form) this->transform_from_ntt_inplace(even);
                        this->add_inplace(even, temp, pool);
                    }
                }
            }
        } else { // For device code, it is more convenient to directly set the empties to zero
            {
                std::vector<Slice<uint64_t>> to_set_zeros;
                for (size_t i = 0; i < max_cipher_count; i++) {
                    if (rlwes[i].polynomial_count() == 0) {
                        rlwes[i] = Ciphertext::like(rlwes[0], false, pool);
                        to_set_zeros.push_back(rlwes[i].reference());
                    }
                }
                utils::set_slice_b(0, to_set_zeros, pool);
            }

            std::vector<Ciphertext> temps(max_cipher_count / 2);
            for (size_t i = 0; i < max_cipher_count / 2; i++) {
                temps[i] = Ciphertext::like(rlwes[0], false, pool);
            }

            for (size_t layer = 0; layer < layers_required; layer++) {

                size_t gap = 1 << layer;
                size_t shift = input_interval >> (layer + 1);
                size_t galois_element = (poly_modulus_degree / input_interval) * (1 << (layer + 1)) + 1;
                size_t pair_count = max_cipher_count / (gap * 2);

                std::vector<Ciphertext*> odds(pair_count);
                std::vector<const Ciphertext*> odds_const(pair_count);
                std::vector<Ciphertext*> evens(pair_count);
                std::vector<const Ciphertext*> evens_const(pair_count);
                std::vector<Ciphertext*> temps_ptr(pair_count);
                std::vector<const Ciphertext*> temps_const_ptr(pair_count);
                size_t odd_polynomial_count = 0;
                for (size_t i = 0; i < pair_count; i++) {
                    size_t offset = i * gap * 2;
                    Ciphertext& even = rlwes[offset];
                    Ciphertext& odd = rlwes[offset + gap];
                    if (i == 0) {
                        odd_polynomial_count = odd.polynomial_count();
                    } else {
                        if (odd.polynomial_count() != odd_polynomial_count) {
                            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_new] Mismatched polynomial count.");
                        }
                    }
                    odds[i] = &odd;
                    odds_const[i] = &odd;
                    evens[i] = &even;
                    evens_const[i] = &even;
                    temps_ptr[i] = &temps[i];
                    temps_const_ptr[i] = &temps[i];
                }

                utils::negacyclic_shift_bps(
                    batch_utils::pcollect_const_reference(odds_const),
                    shift, odd_polynomial_count, poly_modulus_degree, coeff_modulus,
                    batch_utils::pcollect_reference(temps_ptr), pool
                );
                this->sub_batched(evens_const, temps_const_ptr, odds, pool);
                this->add_inplace_batched(evens, temps_const_ptr, pool);
                if (output_ntt_form) this->transform_to_ntt_inplace_batched(odds, pool);
                this->apply_galois_inplace_batched(odds, galois_element, automorphism_keys, pool);
                if (output_ntt_form) this->transform_from_ntt_inplace_batched(odds, pool);
                this->add_inplace_batched(evens, odds_const, pool);

            }
        }

        // take the first element
        Ciphertext ret = std::move(rlwes[0]);
        if (output_ntt_form) {
            this->transform_to_ntt_inplace(ret);
        }
        if (output_interval != 1 && apply_field_trace) {
            size_t logn = is_power_of_two(poly_modulus_degree / output_interval).first;
            field_trace_inplace(ret, automorphism_keys, logn, pool);
        }
        return ret;

    }

    void Evaluator::pack_rlwe_ciphertexts_batched(
        const std::vector<std::vector<const Ciphertext*>>& cipher_groups, const GaloisKeys& automorphism_keys, size_t shift, size_t input_interval, size_t output_interval,
        const std::vector<Ciphertext*> outputs, MemoryPoolHandle pool, bool apply_field_trace
    ) const {

        if (cipher_groups.size() != outputs.size()) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input groups and outputs should have same size.");
        }

        size_t groups_count = cipher_groups.size();
        if (!this->on_device() || groups_count < utils::BATCH_OP_THRESHOLD) {
            // simply run singles
            for (size_t i = 0; i < groups_count; i++) {
                *outputs[i] = this->pack_rlwe_ciphertexts_new(
                    cipher_groups[i], automorphism_keys, shift, input_interval, output_interval, pool, apply_field_trace
                );
            }
            return;
        }

        // sanity check
        if (cipher_groups[0].size() == 0) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input group 0 is empty.");
        }
        ParmsID parms_id = cipher_groups[0][0]->parms_id();
        bool input_ntt_form = cipher_groups[0][0]->is_ntt_form();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::pack_rlwe_ciphertexts_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        SchemeType scheme = parms.scheme();
        bool output_ntt_form = scheme == SchemeType::CKKS || scheme == SchemeType::BGV;
        size_t poly_modulus_degree = parms.poly_modulus_degree();
        if (input_interval > poly_modulus_degree) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] input_interval must be less than poly_modulus_degree.");
        }
        if (output_interval > input_interval) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] output_interval must be less than input_interval.");
        }
        if (!is_power_of_two(input_interval).second) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] input_interval must be power of two.");
        }
        if (!is_power_of_two(output_interval).second) {
            throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] output_interval must be power of two.");
        }
        size_t max_cipher_count = input_interval / output_interval; 
        size_t total_input_cipher_count = 0;
        size_t polynomial_count = cipher_groups[0][0]->polynomial_count();
        for (size_t i = 0; i < groups_count; i++) {
            const std::vector<const Ciphertext*>& group = cipher_groups[i];
            if (group.size() == 0) {
                throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input group " + std::to_string(i) + " is empty.");
            }
            if (group.size() > max_cipher_count) {
                throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input group " + std::to_string(i) + " has more than input_interval / output_interval.");
            }
            total_input_cipher_count += group.size();
            // every input should have the same parms_id and ntt_form
            for (size_t j = 0; j < group.size(); j++) {
                if (group[j]->parms_id() != parms_id) {
                    throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input[" + std::to_string(i) + "][" + std::to_string(j) + "] has different parms_id.");
                }
                if (group[j]->is_ntt_form() != input_ntt_form) {
                    throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input[" + std::to_string(i) + "][" + std::to_string(j) + "] has different ntt_form.");
                }
                if (group[j]->polynomial_count() != polynomial_count) {
                    throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input[" + std::to_string(i) + "][" + std::to_string(j) + "] has different polynomial count.");
                }
                // for ckks, inside a single group the scale should be the same
                if (scheme == SchemeType::CKKS) {
                    if (group[j]->scale() != group[0]->scale()) {
                        throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input[" + std::to_string(i) + "][" + std::to_string(j) + "] has different scale.");
                    }
                }
                // for bgv, inside a single group the correction factor should be the same
                if (scheme == SchemeType::BGV) {
                    if (group[j]->correction_factor() != group[0]->correction_factor()) {
                        throw std::invalid_argument("[Evaluator::pack_rlwe_ciphertexts_batched] Input[" + std::to_string(i) + "][" + std::to_string(j) + "] has different correction factor.");
                    }
                }
            }
        }

        size_t layers_required = is_power_of_two(max_cipher_count).first;
        auto coeff_modulus = parms.coeff_modulus();

        std::vector<std::vector<Ciphertext>> rlwes(groups_count, std::vector<Ciphertext>(max_cipher_count));
        std::vector<Slice<uint64_t>> rlwes_slice(total_input_cipher_count, Slice<uint64_t>(nullptr, 0, false, nullptr));
        std::vector<Ciphertext*> rlwes_ptr; rlwes_ptr.reserve(total_input_cipher_count);

        { // create copies or empty ciphertext buffers
            size_t group_offset = 0;
            std::vector<ConstSlice<uint64_t>> copy_froms(total_input_cipher_count, ConstSlice<uint64_t>(nullptr, 0, false, nullptr));
            std::vector<Slice<uint64_t>> to_set_zeros; to_set_zeros.reserve(max_cipher_count * groups_count - total_input_cipher_count);
            for (size_t j = 0; j < groups_count; j++) {
                for (size_t i = 0; i < max_cipher_count; i++) {
                    size_t index = static_cast<size_t>(utils::reverse_bits_uint64(static_cast<uint64_t>(i), layers_required));
                    if (index < cipher_groups[j].size()) {
                        rlwes[j][i] = Ciphertext::like(*cipher_groups[j][index], false, pool);
                        rlwes_slice[group_offset + index] = rlwes[j][i].reference();
                        rlwes_ptr.push_back(&rlwes[j][i]);
                    } else {
                        // rlwes[j][0] must be present because we have at least 1 element in the group
                        rlwes[j][i] = Ciphertext::like(rlwes[j][0], false, pool);
                        rlwes[j][i].is_ntt_form() = false;
                        to_set_zeros.push_back(rlwes[j][i].reference());
                    }
                }
                for (size_t i = 0; i < cipher_groups[j].size(); i++) {
                    copy_froms[group_offset + i] = cipher_groups[j][i]->reference();
                }
                group_offset += cipher_groups[j].size();
            }
            utils::copy_slice_b(copy_froms, rlwes_slice, pool);
            utils::set_slice_b(0, to_set_zeros, pool);
        }

        if (input_ntt_form) {
            this->transform_from_ntt_inplace_batched(rlwes_ptr, pool);
        }
        this->divide_by_poly_modulus_degree_inplace_batched(rlwes_ptr, poly_modulus_degree / input_interval, pool);
        if (shift != 0) {
            this->negacyclic_shift_inplace_batched(rlwes_ptr, shift, pool);
        }

        std::vector<std::vector<Ciphertext>> temps(groups_count, std::vector<Ciphertext>(max_cipher_count / 2));
        for (size_t j = 0; j < groups_count; j++) {
            for (size_t i = 0; i < max_cipher_count / 2; i++) {
                temps[j][i] = Ciphertext::like(rlwes[j][0], false, pool);
            }
        }

        for (size_t layer = 0; layer < layers_required; layer++) {

            size_t gap = 1 << layer;
            size_t shift = input_interval >> (layer + 1);
            size_t galois_element = (poly_modulus_degree / input_interval) * (1 << (layer + 1)) + 1;
            size_t pair_count = max_cipher_count / (gap * 2);

            std::vector<Ciphertext*> odds(groups_count * pair_count);
            std::vector<const Ciphertext*> odds_const(groups_count * pair_count);
            std::vector<Ciphertext*> evens(groups_count * pair_count);
            std::vector<const Ciphertext*> evens_const(groups_count * pair_count);
            std::vector<Ciphertext*> temps_ptr(groups_count * pair_count);
            std::vector<const Ciphertext*> temps_const_ptr(groups_count * pair_count);

            size_t iter = 0;
            for (size_t j = 0; j < groups_count; j++) {
                for (size_t i = 0; i < pair_count; i++) {
                    size_t offset = i * gap * 2;
                    Ciphertext& even = rlwes[j][offset];
                    Ciphertext& odd = rlwes[j][offset + gap];
                    odds[iter]            = &odd;
                    odds_const[iter]      = &odd;
                    evens[iter]           = &even;
                    evens_const[iter]     = &even;
                    temps_ptr[iter]       = &temps[j][i];
                    temps_const_ptr[iter] = &temps[j][i];
                    iter++;
                }
            }

            utils::negacyclic_shift_bps(
                batch_utils::pcollect_const_reference(odds_const),
                shift, polynomial_count, poly_modulus_degree, coeff_modulus,
                batch_utils::pcollect_reference(temps_ptr), pool
            );
            this->sub_batched(evens_const, temps_const_ptr, odds, pool);
            this->add_inplace_batched(evens, temps_const_ptr, pool);
            if (output_ntt_form) this->transform_to_ntt_inplace_batched(odds, pool);
            this->apply_galois_inplace_batched(odds, galois_element, automorphism_keys, pool);
            if (output_ntt_form) this->transform_from_ntt_inplace_batched(odds, pool);
            this->add_inplace_batched(evens, odds_const, pool);

        }

        // take the first element
        for (size_t j = 0; j < groups_count; j++) {
            *outputs[j] = std::move(rlwes[j][0]);
        }
        if (output_ntt_form) {
            this->transform_to_ntt_inplace_batched(outputs, pool);
        }
        if (output_interval != 1 && apply_field_trace) {
            size_t logn = is_power_of_two(poly_modulus_degree / output_interval).first;
            field_trace_inplace_batched(outputs, automorphism_keys, logn, pool);
        }

    }


}