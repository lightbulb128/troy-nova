#include "scaling_variant.cuh"

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
        if (!utils::same(device, context_data->on_device(), destination.on_device())) {
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
            kernel_translate_plain<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count,
                coeff_count,
                plain_data,
                coeff_modulus,
                destination,
                subtract
            );
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
        size_t coeff_count,
        ConstSlice<uint64_t> plain_data,
        ConstSlice<MultiplyUint64Operand> coeff_div_plain_modulus,
        uint64_t plain_upper_half_threshold,
        uint64_t q_mod_t,
        uint64_t plain_modulus_value,
        ConstSlice<Modulus> coeff_modulus,
        Slice<uint64_t> destination,
        bool add_to_destination,
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
        utils::multiply_uint64_uint64(plain_data[i], q_mod_t, Slice(prod, 2, true));
        uint8_t carry = utils::add_uint64(prod[0], plain_upper_half_threshold, numerator[0]);
        numerator[1] = prod[1] + static_cast<uint64_t>(carry);
        // Compute fix[0] = floor(numerator / t)
        utils::divide_uint128_uint64_inplace(Slice(numerator, 2, true), plain_modulus_value, Slice(fix, 2, true));
        uint64_t scaled_rounded_coeff = utils::multiply_uint64operand_add_uint64_mod(
            plain_data[i], coeff_div_plain_modulus[j], fix[0], coeff_modulus[j]
        );
        uint64_t& destination_target = destination[j * coeff_count + i];
        if (add_to_destination) {
            if (!subtract) {
                destination_target = utils::add_uint64_mod(destination_target, scaled_rounded_coeff, coeff_modulus[j]);
            } else {
                destination_target = utils::sub_uint64_mod(destination_target, scaled_rounded_coeff, coeff_modulus[j]);
            }
        } else {
            destination_target = scaled_rounded_coeff;
        }
    }

    void scale_up(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, bool add_to_destination, bool subtract) {
        bool device = plain.on_device();
        if (!utils::same(device, context_data->on_device(), destination.on_device())) {
            throw std::invalid_argument("[scaling_variant::scale_up] Arguments are not on the same device.");
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t plain_coeff_count = plain.coeff_count();
        size_t coeff_count = parms.poly_modulus_degree();
        if (plain_coeff_count > coeff_count) {
            throw std::invalid_argument("[scaling_variant::scale_up] Plain coeff count too large.");
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
            Array<uint64_t> prod(2, false);
            Array<uint64_t> numerator(2, false);
            Array<uint64_t> fix(2, false);
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
                    if (add_to_destination) {
                        if (!subtract) {
                            destination[j * coeff_count + i] = utils::add_uint64_mod(destination[j * coeff_count + i], scaled_rounded_coeff, coeff_modulus[j]);
                        } else {
                            destination[j * coeff_count + i] = utils::sub_uint64_mod(destination[j * coeff_count + i], scaled_rounded_coeff, coeff_modulus[j]);
                        }
                    } else {
                        destination[j * coeff_count + i] = scaled_rounded_coeff;
                    }
                }
            }
        } else {
            size_t total = plain_coeff_count * coeff_modulus_size;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            kernel_multiply_translate_plain<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count,
                coeff_count,
                plain_data,
                coeff_div_plain_modulus,
                plain_upper_half_threshold,
                q_mod_t,
                parms.plain_modulus_host().value(),
                coeff_modulus,
                destination,
                add_to_destination,
                subtract
            );
        }
    }

    void multiply_add_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination) {
        scale_up(plain, context_data, destination, true, false);
    }

    void multiply_sub_plain(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination) {
        scale_up(plain, context_data, destination, true, true);
    }

    __global__ static void kernel_multiply_plain_normal_no_fast_plain_lift(
        size_t plain_coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= plain_coeff_count) return;
        size_t plain_value = plain[i];
        if (plain_value >= plain_upper_half_threshold) {
            utils::add_uint_uint64(plain_upper_half_increment, plain_value, temp.slice(i * coeff_modulus_size, (i + 1) * coeff_modulus_size));
        } else {
            temp[coeff_modulus_size * i] = plain_value;
        }
    }

    void multiply_plain_normal_no_fast_plain_lift(
        size_t plain_coeff_count, size_t coeff_modulus_size,
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
                }
            } 
        } else {
            size_t block_count = utils::ceil_div(plain_coeff_count, utils::KERNEL_THREAD_COUNT);
            kernel_multiply_plain_normal_no_fast_plain_lift<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count, coeff_modulus_size,
                plain, temp, plain_upper_half_threshold, plain_upper_half_increment
            );
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
        if (global_index >= plain_coeff_count * coeff_modulus_size) return;
        size_t i = global_index / plain_coeff_count;
        size_t j = global_index % plain_coeff_count;
        temp[i * coeff_count + j] = (plain[j] >= plain_upper_half_threshold)
            ? plain[j] + plain_upper_half_increment[i]
            : plain[j];
    }

    static void multiply_plain_normal_fast_plain_lift(
        size_t plain_coeff_count, size_t coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        bool device = temp.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_modulus_size; i++) {
                for (size_t j = 0; j < plain_coeff_count; j++) {
                    temp[i * coeff_count + j] = (plain[j] >= plain_upper_half_threshold)
                        ? plain[j] + plain_upper_half_increment[i]
                        : plain[j];
                }
            }
        } else {
            size_t total = plain_coeff_count * coeff_modulus_size;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            kernel_multiply_plain_normal_fast_plain_lift<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count, coeff_count, coeff_modulus_size,
                plain, temp, plain_upper_half_threshold, plain_upper_half_increment
            );
        }
    }

    void centralize(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination) {
        bool device = plain.on_device();
        if (!utils::same(device, context_data->on_device(), destination.on_device())) {
            throw std::invalid_argument("[scaling_variant::centralize] Arguments are not on the same device.");
        }
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t plain_coeff_count = plain.coeff_count();
        size_t coeff_count = parms.poly_modulus_degree();
        if (plain_coeff_count > coeff_count) {
            throw std::invalid_argument("[scaling_variant::centralize] Plain coeff count too large.");
        }
        size_t coeff_modulus_size = coeff_modulus.size();
        ConstSlice<uint64_t> plain_data = plain.data().const_reference();
        ConstSlice<MultiplyUint64Operand> coeff_div_plain_modulus = context_data->coeff_div_plain_modulus();
        uint64_t plain_upper_half_threshold = context_data->plain_upper_half_threshold();
        ConstSlice<uint64_t> plain_upper_half_increment = context_data->plain_upper_half_increment();
        if (destination.size() != coeff_modulus_size * coeff_count) {
            throw std::invalid_argument("[scaling_variant::centralize] Destination has incorrect size.");
        }

        if (!context_data->qualifiers().using_fast_plain_lift) {
            multiply_plain_normal_no_fast_plain_lift(
                plain_coeff_count, coeff_modulus_size,
                plain.poly(), destination, plain_upper_half_threshold, plain_upper_half_increment
            );
            context_data->rns_tool().base_q().decompose_array(destination);
        } else {
            // Note that in this case plain_upper_half_increment holds its value in RNS form modulo the coeff_modulus
            // primes.
            multiply_plain_normal_fast_plain_lift(
                plain_coeff_count, coeff_count, coeff_modulus_size,
                plain.poly(), destination, plain_upper_half_threshold, plain_upper_half_increment
            );
        }
    }

}}