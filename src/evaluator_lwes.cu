#include "evaluator.h"
#include "evaluator_utils.h"

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
        bool device = this->on_device();
        while ((static_cast<size_t>(1) << l) < lwes_count) l += 1;
        std::vector<Ciphertext> rlwes(1 << l);
        for (size_t i = 0; i < (static_cast<size_t>(1)<<l); i++) {
            size_t index = static_cast<size_t>(utils::reverse_bits_uint64(static_cast<uint64_t>(i), l));
            if (index < lwes_count) {
                rlwes[i] = this->assemble_lwe_new(lwes[index], pool);
                this->divide_by_poly_modulus_degree_inplace(rlwes[i]);
            } else {
                rlwes[i] = Ciphertext(); 
                if (device) rlwes[i].to_device_inplace(pool);
            }
        }
        Ciphertext temp = Ciphertext::like(rlwes[0], false, pool);
        for (size_t layer = 0; layer < l; layer++) {
            size_t gap = 1 << layer;
            size_t offset = 0;
            size_t shift = poly_modulus_degree >> (layer + 1);
            while (offset < (static_cast<size_t>(1) << l)) {
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
                
                if (!even_empty) {
                    if (!odd_empty) {
                        this->sub(even, temp, odd, pool);
                        this->add_inplace(even, temp, pool);
                        if (ntt_form) this->transform_to_ntt_inplace(odd);
                        this->apply_galois_inplace(odd, (1 << (layer + 1)) + 1, automorphism_keys, pool);
                        if (ntt_form) this->transform_from_ntt_inplace(odd);
                        this->add_inplace(even, odd, pool);
                    } else {
                        if (ntt_form) {
                            this->transform_to_ntt(even, temp, pool);
                            this->apply_galois_inplace(temp, (1 << (layer + 1)) + 1, automorphism_keys, pool);
                            this->transform_from_ntt_inplace(temp);
                        } else {
                            this->apply_galois(even, (1 << (layer + 1)) + 1, automorphism_keys, temp, pool);
                        }
                        this->add_inplace(even, temp, pool);
                    }
                } else {
                    this->negate(temp, even, pool);
                    if (ntt_form) this->transform_to_ntt_inplace(even);
                    this->apply_galois_inplace(even, (1 << (layer + 1)) + 1, automorphism_keys, pool);
                    if (ntt_form) this->transform_from_ntt_inplace(even);
                    this->add_inplace(even, temp, pool);
                }
                offset += (gap << 1);
            }
        }
        // take the first element
        Ciphertext ret = std::move(rlwes[0]);
        if (ntt_form) {
            this->transform_to_ntt_inplace(ret);
        }
        if (apply_field_trace) field_trace_inplace(ret, automorphism_keys, l, pool);
        return ret;
    }

}