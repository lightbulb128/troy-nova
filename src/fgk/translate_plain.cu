#include "translate_plain.h"
#include "../batch_utils.h"

namespace troy::utils::fgk::translate_plain {
    
    __device__ static void device_multiply_translate_plain(
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
        size_t poly_count = destination.size() / coeff_count / coeff_modulus.size();
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
        device_multiply_translate_plain(
            from,
            plain_coeff_count,
            coeff_count,
            plain_data,
            coeff_div_plain_modulus,
            plain_upper_half_threshold,
            q_mod_t,
            plain_modulus_value,
            coeff_modulus,
            destination,
            subtract
        );
    }
    
    __global__ static void kernel_multiply_translate_plain_batched(
        ConstSliceArrayRef<uint64_t> from,
        size_t plain_coeff_count,
        size_t coeff_count,
        ConstSliceArrayRef<uint64_t> plain_data,
        ConstSlice<MultiplyUint64Operand> coeff_div_plain_modulus,
        uint64_t plain_upper_half_threshold,
        uint64_t q_mod_t,
        uint64_t plain_modulus_value,
        ConstSlice<Modulus> coeff_modulus,
        SliceArrayRef<uint64_t> destination,
        bool subtract
    ) {
        for (size_t i = 0; i < destination.size(); i++) {
            device_multiply_translate_plain(
                from[i],
                plain_coeff_count,
                coeff_count,
                plain_data[i],
                coeff_div_plain_modulus,
                plain_upper_half_threshold,
                q_mod_t,
                plain_modulus_value,
                coeff_modulus,
                destination[i],
                subtract
            );
        }
    }

    void multiply_translate_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count, bool subtract) {
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
        size_t coeff_count = destination_coeff_count;
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
    
    void multiply_translate_plain_batched(
        const utils::ConstSliceVec<uint64_t>& from, 
        const std::vector<const Plaintext*> plain, 
        ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, 
        size_t destination_coeff_count, bool subtract,
        MemoryPoolHandle pool
    ) {
        
        if (plain.size() != destination.size() || plain.size() != from.size()) {
            throw std::invalid_argument("[scaling_variant::scale_up_batched] Arguments have different sizes.");
        }
        if (destination.size() == 0) return;
        bool device = context_data->on_device();
        if (!device || destination.size() < utils::BATCH_OP_THRESHOLD) {
            // directly call n times single multiply_translate_plain
            for (size_t i = 0; i < destination.size(); i++) {
                multiply_translate_plain(from[i], *plain[i], context_data, destination[i], destination_coeff_count, subtract);
            }
        } else {
            const EncryptionParameters& parms = context_data->parms();
            ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
            size_t plain_coeff_count = plain[0]->coeff_count();
            // check all should have same plain_coeff_count
            for (size_t i = 1; i < plain.size(); i++) {
                if (plain[i]->coeff_count() != plain_coeff_count) {
                    throw std::invalid_argument("[scaling_variant::scale_up_batched] All plaintexts should have the same number of coefficients.");
                }
            }
            if (destination_coeff_count < plain_coeff_count) {
                throw std::invalid_argument("[scaling_variant::scale_up] destination_coeff_count should no less than plain_coeff_count.");
            }
            size_t coeff_modulus_size = coeff_modulus.size();
            ConstSlice<MultiplyUint64Operand> coeff_div_plain_modulus = context_data->coeff_div_plain_modulus();
            uint64_t plain_upper_half_threshold = context_data->plain_upper_half_threshold();
            uint64_t q_mod_t = context_data->coeff_modulus_mod_plain_modulus();

            auto plain_slices = batch_utils::pcollect_const_reference(plain);
            auto plain_batched = batch_utils::construct_batch(plain_slices, pool, coeff_modulus);
            auto destination_batched = batch_utils::construct_batch(destination, pool, coeff_modulus);
            ConstSliceArray<uint64_t> from_batched(&*from.begin(), from.size());
            from_batched.to_device_inplace(pool);
            size_t total = destination_coeff_count * coeff_modulus_size;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(coeff_modulus.device_index());

            kernel_multiply_translate_plain_batched<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                from_batched,
                plain_coeff_count,
                destination_coeff_count,
                plain_batched,
                coeff_div_plain_modulus,
                plain_upper_half_threshold,
                q_mod_t,
                parms.plain_modulus_host().value(),
                coeff_modulus,
                destination_batched,
                subtract
            );
        }
    }

    static __device__ void device_scatter_translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= moduli.size() * from_degree) return;
        
        size_t j = idx / from_degree;
        size_t k = idx % from_degree;
        const size_t& idx1 = idx;
        size_t idx2 = j * translation_degree + k;
        
        if (k < translation_degree) {
            if (from.raw_pointer()) {
                if (!subtract) {
                    destination[idx] = utils::add_uint64_mod(from[idx1], translation[idx2], moduli[j]);
                } else {
                    destination[idx] = utils::sub_uint64_mod(from[idx1], translation[idx2], moduli[j]);
                }
            } else {
                if (!subtract) {
                    destination[idx] = translation[idx2];
                } else {
                    destination[idx] = utils::negate_uint64_mod(translation[idx2], moduli[j]);
                }
            }
        } else {
            if (from.raw_pointer()) {
                destination[idx] = from[idx];
            } else {
                destination[idx] = 0;
            }
        }

        size_t poly_count = destination.size() / from_degree / moduli.size();
        for (size_t p = 1; p < poly_count; p++) {
            size_t offset = p * from_degree * moduli.size() + idx;
            if (from.raw_pointer()) {
                destination[offset] = from[offset];
            } else {
                destination[offset] = 0;
            }
        }
    
    }

    static __global__ void kernel_scatter_translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract) {
        device_scatter_translate_copy(from, translation, from_degree, translation_degree, moduli, destination, subtract);
    }

    static __global__ void kernel_scatter_translate_copy_batched(
        ConstSliceArrayRef<uint64_t> from, 
        ConstSliceArrayRef<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, 
        SliceArrayRef<uint64_t> destination, bool subtract
    ) {
        for (size_t i = 0; i < destination.size(); i++) {
            device_scatter_translate_copy(from[i], translation[i], from_degree, translation_degree, moduli, destination[i], subtract);
        }
    }

    void scatter_translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract) {
        bool device = destination.on_device();
        if (!utils::device_compatible(translation, destination)) {
            throw std::invalid_argument("[scatter_translate_copy] Arguments are not on the same device.");
        }
        if (from.raw_pointer()) {
            if (!utils::device_compatible(from, destination)) {
                throw std::invalid_argument("[scatter_translate_copy] Arguments are not on the same device.");
            }
            if (from.size() != destination.size()) {
                throw std::invalid_argument("[scatter_translate_copy] add_to_destination has incorrect size.");
            }
        }
        if (!device) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < from_degree; k++) {
                    size_t idx = j * from_degree + k;
                    if (k < translation_degree) {
                        size_t idx1 = j * from_degree + k;
                        size_t idx2 = j * translation_degree + k;
                        if (from.raw_pointer()) {
                            if (!subtract) {
                                destination[idx] = utils::add_uint64_mod(from[idx1], translation[idx2], moduli[j]);
                            } else {
                                destination[idx] = utils::sub_uint64_mod(from[idx1], translation[idx2], moduli[j]);
                            }
                        } else {
                            if (!subtract) {
                                destination[idx] = translation[idx2];
                            } else {
                                destination[idx] = utils::negate_uint64_mod(translation[idx2], moduli[j]);
                            }
                        }
                    } else {
                        if (from.raw_pointer()) {
                            destination[idx] = from[idx];
                        } else {
                            destination[idx] = 0;
                        }
                    }
                }
            }
            if (from.raw_pointer()) {
                destination.slice(moduli.size() * from_degree, destination.size())
                    .copy_from_slice(from.const_slice(moduli.size() * from_degree, from.size()));
            } else {
                destination.slice(moduli.size() * from_degree, destination.size()).set_zero();
            }
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

    void scatter_translate_copy_batched(
        const ConstSliceVec<uint64_t>& from, 
        const ConstSliceVec<uint64_t>& translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, 
        const SliceVec<uint64_t>& destination, bool subtract,
        MemoryPoolHandle pool
    ) {
        if (from.size() != translation.size() || from.size() != destination.size()) {
            throw std::invalid_argument("[scatter_translate_copy_batched] Arguments have different sizes.");
        }
        if (destination.size() == 0) return;
        bool device = moduli.on_device();
        if (!device || destination.size() < utils::BATCH_OP_THRESHOLD) {
            // directly call n times single scatter_translate_copy
            for (size_t i = 0; i < destination.size(); i++) {
                scatter_translate_copy(from[i], translation[i], from_degree, translation_degree, moduli, destination[i], subtract);
            }
        } else {
            auto destination_batched = batch_utils::construct_batch(destination, pool, moduli);
            utils::ConstSliceArray<uint64_t> from_batched(&*from.begin(), from.size());
            from_batched.to_device_inplace(pool);
            auto translation_batched = batch_utils::construct_batch(translation, pool, moduli);
            size_t total = from_degree * moduli.size();
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(moduli.device_index());
            kernel_scatter_translate_copy_batched<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                from_batched,
                translation_batched,
                from_degree,
                translation_degree,
                moduli,
                destination_batched,
                subtract
            );
        }


    }

}