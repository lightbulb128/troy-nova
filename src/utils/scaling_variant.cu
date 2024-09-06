#include "constants.h"
#include "scaling_variant.h"
#include "../batch_utils.h"
#include "../fgk/translate_plain.h"

namespace troy {namespace scaling_variant {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::SliceVec;
    using utils::ConstSliceVec;
    using utils::SliceArrayRef;
    using utils::ConstSliceArrayRef;
    using utils::MultiplyUint64Operand;
    using utils::Array;

    void scale_up(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count, utils::ConstSlice<uint64_t> add_to_destination, bool subtract) {
        size_t plain_coeff_count = plain.coeff_count();
        if (destination_coeff_count < plain_coeff_count) {
            throw std::invalid_argument("[scaling_variant::scale_up] destination_coeff_count should no less than plain_coeff_count.");
        }
        utils::fgk::translate_plain::multiply_translate_plain(
            add_to_destination,
            plain,
            context_data,
            destination,
            destination_coeff_count,
            subtract
        );
    }

    void scale_up_batched(
        const std::vector<const Plaintext*>& plain, ContextDataPointer context_data,
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count,
        const utils::ConstSliceVec<uint64_t>& add_to_destination, bool subtract,
        MemoryPoolHandle pool
    ) {
        if (plain.size() != destination.size()) {
            throw std::invalid_argument("[scaling_variant::scale_up_batched] plain and destination size mismatch.");
        }
        for (size_t i = 0; i < plain.size(); i++) {
            if (plain[i]->coeff_count() > destination_coeff_count) {
                throw std::invalid_argument("[scaling_variant::scale_up_batched] destination_coeff_count should no less than plain_coeff_count.");
            }
        }
        utils::fgk::translate_plain::multiply_translate_plain_batched(
            add_to_destination,
            plain,
            context_data,
            destination,
            destination_coeff_count,
            subtract,
            pool
        );
    }

    void multiply_add_plain_inplace(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, destination.as_const(), false);
    }

    void multiply_sub_plain_inplace(const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, destination.as_const(), true);
    }


    void multiply_add_plain_inplace_batched(
        const std::vector<const Plaintext*>& plain, ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count,
        MemoryPoolHandle pool
    ) {
        scale_up_batched(plain, context_data, destination, destination_coeff_count, batch_utils::rcollect_as_const(destination), false, pool);
    }
    void multiply_sub_plain_inplace_batched(
        const std::vector<const Plaintext*>& plain, ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count,
        MemoryPoolHandle pool
    ) {
        scale_up_batched(plain, context_data, destination, destination_coeff_count, batch_utils::rcollect_as_const(destination), true, pool);
    }

    void multiply_add_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, from, false);
    }

    void multiply_sub_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, size_t destination_coeff_count) {
        scale_up(plain, context_data, destination, destination_coeff_count, from, true);
    }

    __device__ static void device_multiply_plain_normal_no_fast_plain_lift(
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
    __global__ static void kernel_multiply_plain_normal_no_fast_plain_lift(
        size_t plain_coeff_count, size_t temp_coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        device_multiply_plain_normal_no_fast_plain_lift(
            plain_coeff_count, temp_coeff_count, coeff_modulus_size,
            plain, temp, plain_upper_half_threshold, plain_upper_half_increment
        );
    }
    __global__ static void kernel_multiply_plain_normal_no_fast_plain_lift_batched(
        size_t plain_coeff_count, size_t temp_coeff_count, size_t coeff_modulus_size,
        ConstSliceArrayRef<uint64_t> plain, 
        SliceArrayRef<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        for (size_t i = 0; i < plain.size(); i++) {
            device_multiply_plain_normal_no_fast_plain_lift(
                plain_coeff_count, temp_coeff_count, coeff_modulus_size,
                plain[i], temp[i], plain_upper_half_threshold, plain_upper_half_increment
            );
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

    void multiply_plain_normal_no_fast_plain_lift_batched(
        size_t plain_coeff_count, size_t temp_coeff_count, size_t coeff_modulus_size,
        const ConstSliceVec<uint64_t>& plain, 
        const SliceVec<uint64_t>& temp,
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment,
        MemoryPoolHandle pool
    ) {
        if (plain.size() != temp.size()) {
            throw std::invalid_argument("[scaling_variant::multiply_plain_normal_no_fast_plain_lift_batched] plain and temp size mismatch.");
        }
        if (plain.size() == 0) return;
        const auto& comp_ref = plain_upper_half_increment;
        bool device = comp_ref.on_device();
        if (!device || plain.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < plain.size(); i++) {
                multiply_plain_normal_no_fast_plain_lift(
                    plain_coeff_count, temp_coeff_count, coeff_modulus_size,
                    plain[i], temp[i], plain_upper_half_threshold, plain_upper_half_increment
                );
            }
        } else {
            size_t block_count = utils::ceil_div(temp_coeff_count, utils::KERNEL_THREAD_COUNT);
            auto plain_batched = batch_utils::construct_batch(plain, pool, comp_ref);
            auto temp_batched = batch_utils::construct_batch(temp, pool, comp_ref);
            utils::set_device(comp_ref.device_index());
            kernel_multiply_plain_normal_no_fast_plain_lift_batched<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count, temp_coeff_count, coeff_modulus_size,
                plain_batched, temp_batched, plain_upper_half_threshold, plain_upper_half_increment
            );
            utils::stream_sync();
        }
    }

    __device__ static void device_multiply_plain_normal_fast_plain_lift(
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

    __global__ static void kernel_multiply_plain_normal_fast_plain_lift(
        size_t plain_coeff_count, size_t coeff_count, size_t coeff_modulus_size,
        ConstSlice<uint64_t> plain, 
        Slice<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        device_multiply_plain_normal_fast_plain_lift(
            plain_coeff_count, coeff_count, coeff_modulus_size,
            plain, temp, plain_upper_half_threshold, plain_upper_half_increment
        );
    }

    __global__ static void kernel_multiply_plain_normal_fast_plain_lift_batched(
        size_t plain_coeff_count, size_t coeff_count, size_t coeff_modulus_size,
        ConstSliceArrayRef<uint64_t> plain, 
        SliceArrayRef<uint64_t> temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment
    ) {
        for (size_t i = 0; i < plain.size(); i++) {
            device_multiply_plain_normal_fast_plain_lift(
                plain_coeff_count, coeff_count, coeff_modulus_size,
                plain[i], temp[i], plain_upper_half_threshold, plain_upper_half_increment
            );
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

    static void multiply_plain_normal_fast_plain_lift_batched(
        size_t plain_coeff_count, size_t temp_coeff_count, size_t coeff_modulus_size,
        const ConstSliceVec<uint64_t>& plain, 
        const SliceVec<uint64_t>& temp, 
        uint64_t plain_upper_half_threshold,
        ConstSlice<uint64_t> plain_upper_half_increment,
        MemoryPoolHandle pool
    ) {
        if (plain.size() != temp.size()) {
            throw std::invalid_argument("[scaling_variant::multiply_plain_normal_no_fast_plain_lift_batched] plain and temp size mismatch.");
        }
        if (plain.size() == 0) return;
        const auto& comp_ref = plain_upper_half_increment;
        bool device = comp_ref.on_device();
        if (!device || plain.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < plain.size(); i++) {
                multiply_plain_normal_fast_plain_lift(
                    plain_coeff_count, temp_coeff_count, coeff_modulus_size,
                    plain[i], temp[i], plain_upper_half_threshold, plain_upper_half_increment
                );
            }
        } else {
            size_t total = temp_coeff_count * coeff_modulus_size;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            auto plain_batched = batch_utils::construct_batch(plain, pool, comp_ref);
            auto temp_batched = batch_utils::construct_batch(temp, pool, comp_ref);
            utils::set_device(comp_ref.device_index());
            kernel_multiply_plain_normal_fast_plain_lift_batched<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                plain_coeff_count, temp_coeff_count, coeff_modulus_size,
                plain_batched, temp_batched, plain_upper_half_threshold, plain_upper_half_increment
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

    void centralize_batched(
        const std::vector<const Plaintext*> plain, ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, size_t destination_coeff_count, 
        MemoryPoolHandle pool
    ) {
        if (plain.size() != destination.size()) {
            throw std::invalid_argument("[scaling_variant::centralize_batched] plain and destination size mismatch.");
        }
        if (plain.size() == 0) return;
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t plain_coeff_count = plain[0]->coeff_count();
        for (size_t i = 1; i < plain.size(); i++) {
            if (plain[i]->coeff_count() != plain_coeff_count) {
                throw std::invalid_argument("[scaling_variant::centralize_batched] plain coeff_count mismatch.");
            }
        }
        if (destination_coeff_count < plain_coeff_count) {
            throw std::invalid_argument("[scaling_variant::centralize_batched] destination_coeff_count should no less than plain_coeff_count.");
        }
        size_t coeff_modulus_size = coeff_modulus.size();
        uint64_t plain_upper_half_threshold = context_data->plain_upper_half_threshold();
        ConstSlice<uint64_t> plain_upper_half_increment = context_data->plain_upper_half_increment();
        for (size_t i = 0; i < destination.size(); i++) {
            if (destination[i].size() < coeff_modulus_size * plain_coeff_count) {
                throw std::invalid_argument("[scaling_variant::centralize_batched] Destination has incorrect size.");
            }
        }
        // destination.set_zero();
        if (!context_data->qualifiers().using_fast_plain_lift) {
            multiply_plain_normal_no_fast_plain_lift_batched(
                plain_coeff_count, destination_coeff_count, coeff_modulus_size,
                batch_utils::pcollect_const_poly(plain), destination, plain_upper_half_threshold, plain_upper_half_increment, pool
            );
            context_data->rns_tool().base_q().decompose_array_batched(destination, pool);
        } else {
            // Note that in this case plain_upper_half_increment holds its value in RNS form modulo the coeff_modulus
            // primes.
            multiply_plain_normal_fast_plain_lift_batched(
                plain_coeff_count, destination_coeff_count, coeff_modulus_size,
                batch_utils::pcollect_const_poly(plain), destination, plain_upper_half_threshold, plain_upper_half_increment, pool
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