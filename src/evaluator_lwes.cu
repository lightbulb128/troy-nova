#include "batch_utils.h"
#include "ciphertext.h"
#include "evaluator.h"
#include "evaluator_utils.h"
#include "decryptor.h"
#include "batch_encoder.h"

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


    Ciphertext Evaluator::pack_lwe_ciphertexts_new(const std::vector<LWECiphertext>& lwes, const GaloisKeys& automorphism_keys, MemoryPoolHandle pool, bool apply_field_trace) const {
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
        size_t poly_modulus_degree = context_data->parms().poly_modulus_degree();
        if (lwes_count > poly_modulus_degree) {
            throw std::invalid_argument("[Evaluator::pack_lwe_ciphertexts_new] LWE ciphertexts count must be less than poly_modulus_degree.");
        }

        size_t l = 0;
        while ((static_cast<size_t>(1) << l) < lwes_count) l += 1;

        std::vector<Ciphertext> rlwes(lwes_count);
        for (size_t i = 0; i < rlwes.size(); i++) {
            rlwes[i] = this->assemble_lwe_new(lwes[i], pool);
        }

        return this->pack_rlwe_ciphertexts_new(
            batch_utils::collect_const_pointer(rlwes), 
            automorphism_keys, 
            0, poly_modulus_degree, poly_modulus_degree / (static_cast<size_t>(1) << l), apply_field_trace, pool);

    }

    std::pair<size_t, bool> is_power_of_two(uint64_t r) {
        auto p = utils::get_power_of_two(r);
        return std::make_pair(p, r == (static_cast<uint64_t>(1) << p));
    }

    Ciphertext Evaluator::pack_rlwe_ciphertexts_new(
        const std::vector<const Ciphertext*>& ciphers, const GaloisKeys& automorphism_keys, size_t shift, size_t input_interval, size_t output_interval,
        bool apply_field_trace,
        MemoryPoolHandle pool
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
        for (size_t i = 0; i < max_cipher_count; i++) {
            size_t index = static_cast<size_t>(utils::reverse_bits_uint64(static_cast<uint64_t>(i), layers_required));
            if (index < ciphers.size()) {
                rlwes[i] = Ciphertext::like(*ciphers[index], false, pool);
                rlwes_slice[index] = rlwes[i].reference();
            } else {
                rlwes[i] = Ciphertext(); 
                if (device) rlwes[i].to_device_inplace(pool);
            }
        }
        utils::copy_slice_b(batch_utils::pcollect_const_reference(ciphers), rlwes_slice, pool);

        {
            std::vector<Ciphertext*> rlwes_ptr(ciphers.size());
            for (size_t i = 0; i < ciphers.size(); i++) {
                rlwes_ptr[i] = &rlwes[i];
            }
            if (input_ntt_form) this->transform_from_ntt_inplace_batched(rlwes_ptr, pool);
            for (size_t i = 0; i < ciphers.size(); i++) {
                this->divide_by_poly_modulus_degree_inplace(rlwes[i], poly_modulus_degree / input_interval);
            }
            if (shift != 0) {
                this->negacyclic_shift_inplace_batched(rlwes_ptr, shift, pool);
            }
            
        }
        
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


}