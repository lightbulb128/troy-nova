#include "scaling_variant.h"

namespace troy {namespace scaling_variant {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::MultiplyUint64Operand;
    using utils::Array;

    __global__ static void kernel_translate_plain(
        size_t plain_coeff_count,
        size_t coeff_count,
        ConstSlice<uint64_t> plain_data,
        ConstSlice<Modulus> coeff_modulus,
        Slice<uint64_t> destination,
        bool subtract
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= plain_coeff_count * coeff_modulus.size()) {
            return;
        }
        size_t i = global_index / plain_coeff_count;
        size_t j = global_index % plain_coeff_count;
        uint64_t m = coeff_modulus[i].reduce(plain_data[j]);
        size_t destination_index = i * coeff_count + j;
        if (!subtract) {
            destination[destination_index] = utils::add_uint64_mod(destination[destination_index], m, coeff_modulus[i]);
        } else {
            destination[destination_index] = utils::sub_uint64_mod(destination[destination_index], m, coeff_modulus[i]);
        }
    }

    static void translate_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, bool subtract) {
        bool device = plain.on_device();
        if (!utils::device_compatible(*context_data, plain, destination)) {
            throw std::invalid_argument("[scaling_variant::translate_plain] Arguments are not on the same device.");
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t plain_coeff_count = plain.coeff_count();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        ConstSlice<uint64_t> plain_data = plain.data().const_reference();
        if (!device) {
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                for (size_t j = 0; j < plain_coeff_count; j++) {
                    uint64_t m = coeff_modulus[i].reduce(plain_data[j]);
                    if (!subtract) {
                        destination[i * coeff_count + j] = utils::add_uint64_mod(destination[i * coeff_count + j], m, coeff_modulus[i]);
                    } else {
                        destination[i * coeff_count + j] = utils::sub_uint64_mod(destination[i * coeff_count + j], m, coeff_modulus[i]);
                    }
                }
            }
        } else {
            size_t total = coeff_modulus_size * plain_coeff_count;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(plain_data.device_index());
            kernel_translate_plain<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count,
                coeff_count,
                plain_data,
                coeff_modulus,
                destination,
                subtract
            );
            utils::stream_sync();
        }
    }

    void add_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination) {
        translate_plain(plain, context_data, destination, false);
    }

    void sub_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination) {
        translate_plain(plain, context_data, destination, true);
    }

    __global__ static void kernel_multiply_translate_plain(
        size_t plain_coeff_count,
        ConstSlice<uint64_t> plain_data,
        ConstSlice<MultiplyUint64Operand> coeff_div_plain_modulus,
        uint64_t plain_upper_half_threshold,
        uint64_t q_mod_t,
        uint64_t plain_modulus_value,
        ConstSlice<Modulus> coeff_modulus,
        Slice<uint64_t> destination,
        size_t destination_coeff_count,
        ConstSlice<uint64_t> add_to_destination,
        bool subtract
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= plain_coeff_count * coeff_modulus.size()) {
            return;
        }
        size_t i = global_index % plain_coeff_count;
        size_t j = global_index / plain_coeff_count;
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
        uint64_t& destination_target = destination[j * destination_coeff_count + i];
        if (add_to_destination.raw_pointer()) {
            uint64_t original = add_to_destination[j * destination_coeff_count + i];
            if (!subtract) {
                destination_target = utils::add_uint64_mod(original, scaled_rounded_coeff, coeff_modulus[j]);
            } else {
                destination_target = utils::sub_uint64_mod(original, scaled_rounded_coeff, coeff_modulus[j]);
            }
        } else {
            destination_target = scaled_rounded_coeff;
        }
    }

    void scale_up(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count, utils::ConstSlice<uint64_t> add_to_destination, bool subtract) {
        bool device = plain.on_device();
        if (!utils::device_compatible(*context_data, plain, destination)) {
            throw std::invalid_argument("[scaling_variant::scale_up] Arguments are not on the same device.");
        }
        if (add_to_destination.raw_pointer()) {
            if (!utils::device_compatible(*context_data, add_to_destination)) {
                throw std::invalid_argument("[scaling_variant::scale_up] Arguments are not on the same device.");
            }
            if (add_to_destination.size() != destination.size()) {
                throw std::invalid_argument("[scaling_variant::scale_up] add_to_destination has incorrect size.");
            }
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t plain_coeff_count = plain.coeff_count();
        if (destination_coeff_count < plain_coeff_count) {
            throw std::invalid_argument("[scaling_variant::scale_up] destination_coeff_count should no less than plain_coeff_count.");
        }
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
            for (size_t i = 0; i < plain_coeff_count; i++) {
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
                    if (add_to_destination.raw_pointer()) {
                        if (!subtract) {
                            destination[j * destination_coeff_count + i] = utils::add_uint64_mod(add_to_destination[j * destination_coeff_count + i], scaled_rounded_coeff, coeff_modulus[j]);
                        } else {
                            destination[j * destination_coeff_count + i] = utils::sub_uint64_mod(add_to_destination[j * destination_coeff_count + i], scaled_rounded_coeff, coeff_modulus[j]);
                        }
                    } else {
                        destination[j * destination_coeff_count + i] = scaled_rounded_coeff;
                    }
                }
            }
        } else {
            size_t total = plain_coeff_count * coeff_modulus_size;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_multiply_translate_plain<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count,
                plain_data,
                coeff_div_plain_modulus,
                plain_upper_half_threshold,
                q_mod_t,
                parms.plain_modulus_host().value(),
                coeff_modulus,
                destination,
                destination_coeff_count,
                add_to_destination,
                subtract
            );
            utils::stream_sync();
        }
    }

    void multiply_add_plain_inplace(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, destination.as_const(), false);
    }

    void multiply_sub_plain_inplace(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, destination.as_const(), true);
    }


    void multiply_add_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, from, false);
    }

    void multiply_sub_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, from, true);
    }

    __global__ static void kernel_multiply_plain_normal_no_fast_plain_lift(
        size_t plain_coeff_count, size_t temp_coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= temp_coeff_count) return;
        if (i < plain_coeff_count) {
            size_t plain_value = plain[i];
            if (plain_value >= plain_upper_half_threshold) {
                utils::add_uint_uint64(plain_upper_half_increment, plain_value, temp.slice(i * coeff_modulus_size, (i + 1) * coeff_modulus_size));
            } else {
                temp[coeff_modulus_size * i] = plain_value;
                for (size_t j = 1; j < coeff_modulus_size; j++) {
                    temp[coeff_modulus_size * i + j] = 0;
                }
            }
        } else {
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                temp[coeff_modulus_size * i + j] = 0;
            }
        }
    }

    void multiply_plain_normal_no_fast_plain_lift(
        size_t plain_coeff_count, size_t temp_coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp,
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        bool device = temp.on_device();
        if (!device) {
            for (size_t i = 0; i < plain_coeff_count; i++) {
                size_t plain_value = plain[i];
                if (plain_value >= plain_upper_half_threshold) {
                    utils::add_uint_uint64(plain_upper_half_increment, plain_value, temp.slice(i * coeff_modulus_size, (i + 1) * coeff_modulus_size));
                } else {
                    temp[coeff_modulus_size * i] = plain_value;
                    for (size_t j = 1; j < coeff_modulus_size; j++) {
                        temp[coeff_modulus_size * i + j] = 0;
                    }
                }
            }
            for (size_t i = plain_coeff_count; i < temp_coeff_count; i++) {
                for (size_t j = 0; j < coeff_modulus_size; j++) {
                    temp[coeff_modulus_size * i + j] = 0;
                }
            }
        } else {
            size_t block_count = utils::ceil_div(temp_coeff_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(temp.device_index());
            kernel_multiply_plain_normal_no_fast_plain_lift<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count, temp_coeff_count, coeff_modulus_size,
                plain, temp, plain_upper_half_threshold, plain_upper_half_increment
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_multiply_plain_normal_fast_plain_lift(
        size_t plain_coeff_count, size_t coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * coeff_modulus_size) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;
        if (j < plain_coeff_count) {
            temp[i * coeff_count + j] = (plain[j] >= plain_upper_half_threshold)
                ? plain[j] + plain_upper_half_increment[i]
                : plain[j];
        } else {
            temp[i * coeff_count + j] = 0;
        }
    }

    static void multiply_plain_normal_fast_plain_lift(
        size_t plain_coeff_count, size_t temp_coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        bool device = temp.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                for (size_t j = 0; j < temp_coeff_count; j++) {
                    if (j < plain_coeff_count) {
                        temp[i * temp_coeff_count + j] = (plain[j] >= plain_upper_half_threshold)
                            ? plain[j] + plain_upper_half_increment[i]
                            : plain[j];
                    } else {
                        temp[i * temp_coeff_count + j] = 0;
                    }
                }
            }
        } else {
            size_t total = temp_coeff_count * coeff_modulus_size;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(temp.device_index());
            kernel_multiply_plain_normal_fast_plain_lift<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count, temp_coeff_count, coeff_modulus_size,
                plain, temp, plain_upper_half_threshold, plain_upper_half_increment
            );
            utils::stream_sync();
        }
    }

    void centralize(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count, MemoryPoolHandle pool) {
        if (!utils::device_compatible(*context_data, plain, destination)) {
            throw std::invalid_argument("[scaling_variant::centralize] Arguments are not on the same device.");
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t plain_coeff_count = plain.coeff_count();
        if (destination_coeff_count < plain_coeff_count) {
            throw std::invalid_argument("[scaling_variant::centralize] destination_coeff_count should no less than plain_coeff_count.");
        }
        size_t coeff_modulus_size = coeff_modulus.size();
        uint64_t plain_upper_half_threshold = context_data->plain_upper_half_threshold();
        ConstSlice<uint64_t> plain_upper_half_increment = context_data->plain_upper_half_increment();
        if (destination.size() < coeff_modulus_size * plain_coeff_count) {
            throw std::invalid_argument("[scaling_variant::centralize] Destination has incorrect size.");
        }
        // destination.set_zero();
        if (!context_data->qualifiers().using_fast_plain_lift) {
            multiply_plain_normal_no_fast_plain_lift(
                plain_coeff_count, destination_coeff_count, coeff_modulus_size,
                plain.poly(), destination, plain_upper_half_threshold, plain_upper_half_increment
            );
            context_data->rns_tool().base_q().decompose_array(destination, pool);
        } else {
            // Note that in this case plain_upper_half_increment holds its value in RNS form modulo the coeff_modulus
            // primes.
            multiply_plain_normal_fast_plain_lift(
                plain_coeff_count, destination_coeff_count, coeff_modulus_size,
                plain.poly(), destination, plain_upper_half_threshold, plain_upper_half_increment
            );
        }
    }

    void scale_down(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, MemoryPoolHandle pool) {
        if (!utils::device_compatible(*context_data, plain, destination)) {
            throw std::invalid_argument("[scaling_variant::scale_down] Arguments are not on the same device.");
        }
        size_t plain_coeff_count = plain.coeff_count();
        if (destination.size() < plain_coeff_count) {
            throw std::invalid_argument("[scaling_variant::scale_down] Destination size should no less than plain_coeff_count.");
        }
        context_data->rns_tool().decrypt_scale_and_round(plain.const_reference(), plain_coeff_count, destination, pool);
    }

    void decentralize(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, uint64_t correction_factor, MemoryPoolHandle pool) {
        if (!utils::device_compatible(*context_data, plain, destination)) {
            throw std::invalid_argument("[scaling_variant::decentralize] Arguments are not on the same device.");
        }
        size_t plain_coeff_count = plain.coeff_count();
        if (destination.size() < plain_coeff_count) {
            throw std::invalid_argument("[scaling_variant::decentralize] Destination size should no less than plain_coeff_count.");
        }
        context_data->rns_tool().decrypt_mod_t(plain.const_reference(), destination, pool);
        if (correction_factor != 1) {
            uint64_t fix = 1;
            if (!utils::try_invert_uint64_mod(correction_factor, context_data->parms().plain_modulus_host(), fix)) {
                throw std::logic_error("[scaling_variant::decentralize] Correction factor is not invertible.");
            }
            utils::multiply_scalar_inplace(destination, fix, context_data->parms().plain_modulus());
        }
    }




}}