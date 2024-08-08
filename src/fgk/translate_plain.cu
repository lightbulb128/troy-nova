#include "translate_plain.h"


namespace troy::utils::fgk::translate_plain {
    
    __global__ static void kernel_multiply_translate_plain(
        ConstSlice<uint64_t> from,
        size_t plain_coeff_count,
        size_t coeff_count,
        ConstSlice<uint64_t> plain_data,
        ConstSlice<MultiplyUint64Operand> coeff_div_plain_modulus,
        uint64_t plain_upper_half_threshold,
        uint64_t q_mod_t,
        uint64_t plain_modulus_value,
        ConstSlice<Modulus> coeff_modulus,
        Slice<uint64_t> destination,
        bool subtract
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * coeff_modulus.size()) {
            return;
        }
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        size_t poly_count = from.size() / coeff_count / coeff_modulus.size();
        uint64_t& destination_target = destination[j * coeff_count + i];
        if (i < plain_coeff_count) {
            uint64_t prod[2]{0, 0};
            uint64_t numerator[2]{0, 0};
            uint64_t fix[2]{0, 0};
            // Compute numerator = (q mod t) * m[i] + (t+1)/2
            utils::multiply_uint64_uint64(plain_data[i], q_mod_t, Slice(prod, 2, true, nullptr));
            uint8_t carry = utils::add_uint64(prod[0], plain_upper_half_threshold, numerator[0]);
            numerator[1] = prod[1] + static_cast<uint64_t>(carry);
            // Compute fix[0] = floor(numerator / t)
            utils::divide_uint128_uint64_inplace(Slice(numerator, 2, true, nullptr), plain_modulus_value, Slice(fix, 2, true, nullptr));
            uint64_t scaled_rounded_coeff = utils::multiply_uint64operand_add_uint64_mod(
                plain_data[i], coeff_div_plain_modulus[j], fix[0], coeff_modulus[j]
            );
            if (from.raw_pointer()) {
                uint64_t original = from[j * coeff_count + i];
                if (!subtract) {
                    destination_target = utils::add_uint64_mod(original, scaled_rounded_coeff, coeff_modulus[j]);
                } else {
                    destination_target = utils::sub_uint64_mod(original, scaled_rounded_coeff, coeff_modulus[j]);
                }
            } else {
                if (!subtract) {
                    destination_target = scaled_rounded_coeff;
                } else {
                    destination_target = utils::negate_uint64_mod(scaled_rounded_coeff, coeff_modulus[j]);
                }
            }
        } else {
            if (from.raw_pointer()) {
                destination_target = from[j * coeff_count + i];
            } else {
                destination_target = 0;
            }
        }
        for (size_t p = 1; p < poly_count; p++) {
            size_t offset = p * coeff_count * coeff_modulus.size() + j * coeff_count + i;
            if (from.raw_pointer()) {
                destination[offset] = from[offset];
            } else {
                destination[offset] = 0;
            }
        }
    }


    void multiply_translate_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, bool subtract) {
        bool device = plain.on_device();
        if (!utils::device_compatible(*context_data, plain, destination)) {
            throw std::invalid_argument("[scaling_variant::scale_up] Arguments are not on the same device.");
        }
        if (from.raw_pointer()) {
            if (!utils::device_compatible(*context_data, from)) {
                throw std::invalid_argument("[scaling_variant::scale_up] Arguments are not on the same device.");
            }
            if (from.size() != destination.size()) {
                throw std::invalid_argument("[scaling_variant::scale_up] add_to_destination has incorrect size.");
            }
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t plain_coeff_count = plain.coeff_count();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        ConstSlice<uint64_t> plain_data = plain.data().const_reference();
        ConstSlice<MultiplyUint64Operand> coeff_div_plain_modulus = context_data->coeff_div_plain_modulus();
        uint64_t plain_upper_half_threshold = context_data->plain_upper_half_threshold();
        uint64_t q_mod_t = context_data->coeff_modulus_mod_plain_modulus();

        // Coefficients of plain m multiplied by coeff_modulus q, divided by plain_modulus t,
        // and rounded to the nearest integer (rounded up in case of a tie). Equivalent to
        // floor((q * m + floor((t+1) / 2)) / t).
        if (!device) {
            Array<uint64_t> prod(2, false, nullptr);
            Array<uint64_t> numerator(2, false, nullptr);
            Array<uint64_t> fix(2, false, nullptr);
            uint64_t plain_modulus_value = parms.plain_modulus_host().value();
            for (size_t i = 0; i < coeff_count; i++) {
                if (i < plain_coeff_count) {
                    // Compute numerator = (q mod t) * m[i] + (t+1)/2
                    utils::multiply_uint64_uint64(plain_data[i], q_mod_t, prod.reference());
                    uint8_t carry = utils::add_uint64(prod[0], plain_upper_half_threshold, numerator[0]);
                    numerator[1] = prod[1] + static_cast<uint64_t>(carry);
                    // Compute fix[0] = floor(numerator / t)
                    utils::divide_uint128_uint64_inplace(numerator.reference(), plain_modulus_value, fix.reference());
                    // Add to ciphertext: floor(q / t) * m + increment
                    for (size_t j = 0; j < coeff_modulus_size; j++) {
                        uint64_t scaled_rounded_coeff = utils::multiply_uint64operand_add_uint64_mod(
                            plain_data[i], coeff_div_plain_modulus[j], fix[0], coeff_modulus[j]
                        );
                        if (from.raw_pointer()) {
                            if (!subtract) {
                                destination[j * coeff_count + i] = utils::add_uint64_mod(from[j * coeff_count + i], scaled_rounded_coeff, coeff_modulus[j]);
                            } else {
                                destination[j * coeff_count + i] = utils::sub_uint64_mod(from[j * coeff_count + i], scaled_rounded_coeff, coeff_modulus[j]);
                            }
                        } else {
                            if (!subtract) {
                                destination[j * coeff_count + i] = scaled_rounded_coeff;
                            } else {
                                destination[j * coeff_count + i] = utils::negate_uint64_mod(scaled_rounded_coeff, coeff_modulus[j]);
                            }
                        }
                    }
                } else {
                    for (size_t j = 0; j < coeff_modulus_size; j++) {
                        if (from.raw_pointer()) {
                            destination[j * coeff_count + i] = from[j * coeff_count + i];
                        } else {
                            destination[j * coeff_count + i] = 0;
                        }
                    }
                }
            }
            if (from.raw_pointer()) {
                destination.slice(coeff_count * coeff_modulus_size, destination.size())
                    .copy_from_slice(from.const_slice(coeff_count * coeff_modulus_size, from.size()));
            } else {
                destination.slice(coeff_count * coeff_modulus_size, destination.size()).set_zero();
            }
        } else {
            size_t total = coeff_count * coeff_modulus_size;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_multiply_translate_plain<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                from,
                plain_coeff_count,
                coeff_count,
                plain_data,
                coeff_div_plain_modulus,
                plain_upper_half_threshold,
                q_mod_t,
                parms.plain_modulus_host().value(),
                coeff_modulus,
                destination,
                subtract
            );
            utils::stream_sync();
        }
    }

    static __global__ void kernel_scatter_translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= moduli.size() * from_degree) return;
        
        size_t j = idx / from_degree;
        size_t k = idx % from_degree;
        const size_t& idx1 = idx;
        size_t idx2 = j * translation_degree + k;
        
        if (k < translation_degree) {
            if (!subtract) {
                destination[idx] = utils::add_uint64_mod(from[idx1], translation[idx2], moduli[j]);
            } else {
                destination[idx] = utils::sub_uint64_mod(from[idx1], translation[idx2], moduli[j]);
            }
        } else {
            destination[idx] = from[idx1];
        }

        size_t poly_count = from.size() / from_degree / moduli.size();
        for (size_t p = 1; p < poly_count; p++) {
            size_t offset = p * from_degree * moduli.size() + idx;
            destination[offset] = from[offset];
        }
    
    }

    void scatter_translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract) {
        bool device = from.on_device();
        if (!utils::device_compatible(from, translation, destination)) {
            throw std::invalid_argument("[scatter_translate_copy] Arguments are not on the same device.");
        }
        if (!device) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < from_degree; k++) {
                    size_t idx = j * from_degree + k;
                    if (k < translation_degree) {
                        size_t idx1 = j * from_degree + k;
                        size_t idx2 = j * translation_degree + k;
                        if (!subtract) {
                            destination[idx] = utils::add_uint64_mod(from[idx1], translation[idx2], moduli[j]);
                        } else {
                            destination[idx] = utils::sub_uint64_mod(from[idx1], translation[idx2], moduli[j]);
                        }
                    } else {
                        destination[idx] = from[idx];
                    }
                }
            }
            destination.slice(moduli.size() * from_degree, destination.size())
                .copy_from_slice(from.const_slice(moduli.size() * from_degree, from.size()));
        } else {
            size_t total = from_degree * moduli.size();
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_scatter_translate_copy<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                from,
                translation,
                from_degree,
                translation_degree,
                moduli,
                destination,
                subtract
            );
            utils::stream_sync();
        }
    }


}