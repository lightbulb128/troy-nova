#include "evaluator.h"
#include "evaluator_utils.h"
#include "batch_utils.h"
#include "utils/constants.h"
#include "utils/polynomial_buffer.h"
#include "fgk/switch_key.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;
    using utils::NTTTables;
    using utils::ConstPointer;
    using utils::RNSTool;
    using utils::Buffer;
    using utils::MultiplyUint64Operand;
    using utils::GaloisTool;

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
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, true, nullptr);
        utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
        utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
        accumulator_l[0] = key_modulus->reduce_uint128_limbs(qword_slice.as_const());
        accumulator_l[1] = 0;
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
            uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, false, nullptr);
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t k = 0; k < key_component_count; k++) {
                    utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
                    size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
                    Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
                    utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
                    accumulator_l[0] = key_modulus->reduce_uint128_limbs(qword_slice.as_const());
                    accumulator_l[1] = 0;
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                key_vector_j, key_poly_coeff_size, t_operand, key_index, key_modulus
            );
            utils::stream_sync();
        }
    }
    
    __global__ static void kernel_ski_util2(
        Slice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        ConstSlice<uint64_t> key_vector_j,
        size_t key_poly_coeff_size,
        ConstSlice<uint64_t> t_operand,
        size_t key_index
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, true, nullptr);
        utils::multiply_uint64_uint64(t_operand[i], key_vector_j[k * key_poly_coeff_size + key_index * coeff_count + i], qword_slice);
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        Slice<uint64_t> accumulator_l = t_poly_lazy.slice(accumulator_l_offset, accumulator_l_offset + 2);
        utils::add_uint128_inplace(qword_slice, accumulator_l.as_const());
        accumulator_l[0] = qword_slice[0];
        accumulator_l[1] = qword_slice[1];
    }

    static void ski_util2(
        Slice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        ConstSlice<uint64_t> key_vector_j,
        size_t key_poly_coeff_size,
        ConstSlice<uint64_t> t_operand,
        size_t key_index
    ) {
        bool device = t_poly_lazy.on_device();
        if (!device) {
            uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, false, nullptr);
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
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                key_vector_j, key_poly_coeff_size, t_operand, key_index
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_ski_util3(
        ConstSlice<uint64_t> t_poly_lazy,
        size_t coeff_count,
        size_t key_component_count,
        size_t rns_modulus_size,
        Slice<uint64_t> t_poly_prod_iter
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = t_poly_lazy[accumulator_l_offset];
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
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util3<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, rns_modulus_size, t_poly_prod_iter
            );
            utils::stream_sync();
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
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        size_t accumulator_l_offset = k * coeff_count * 2 + 2 * i;
        t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = key_modulus->reduce_uint128_limbs(
            t_poly_lazy.const_slice(accumulator_l_offset, accumulator_l_offset + 2)
        );
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
                    t_poly_prod_iter[k * coeff_count * rns_modulus_size + i] = key_modulus->reduce_uint128_limbs(
                        t_poly_lazy.const_slice(accumulator_l_offset, accumulator_l_offset + 2)
                    );
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * key_component_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_lazy.device_index());
            kernel_ski_util4<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_lazy, coeff_count, key_component_count, 
                rns_modulus_size, t_poly_prod_iter, key_modulus
            );
            utils::stream_sync();
        }
    }

    __global__ static void kernel_ski_util5_step1(
        ConstSlice<uint64_t> t_last,
        size_t coeff_count,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        uint64_t qk_inv_qp,
        uint64_t qk,
        Slice<uint64_t> delta_array
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        uint64_t k = utils::barrett_reduce_uint64(t_last[i], *plain_modulus);
        k = utils::negate_uint64_mod(k, *plain_modulus);
        if (qk_inv_qp != 1) {
            k = utils::multiply_uint64_mod(k, qk_inv_qp, *plain_modulus);
        }
        uint64_t delta = utils::barrett_reduce_uint64(k, key_modulus[j]);
        delta = utils::multiply_uint64_mod(delta, qk, key_modulus[j]);
        uint64_t c_mod_qi = utils::barrett_reduce_uint64(t_last[i], key_modulus[j]);
        delta = utils::add_uint64_mod(delta, c_mod_qi, key_modulus[j]);
        delta_array[global_index] = delta;
    }


    __global__ static void kernel_ski_util5_step2(
        Slice<uint64_t> t_poly_prod_i,
        size_t coeff_count,
        ConstSlice<Modulus> key_modulus,
        size_t decomp_modulus_size,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> encrypted_i,
        ConstSlice<uint64_t> delta_array,
        bool add_inplace
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        uint64_t delta = delta_array[global_index];
        uint64_t& target = t_poly_prod_i[j * coeff_count + i];
        target = utils::sub_uint64_mod(target, delta, key_modulus[j]);
        target = utils::multiply_uint64operand_mod(target, modswitch_factors[j], key_modulus[j]);
        if (add_inplace) {
            encrypted_i[global_index] = utils::add_uint64_mod(target, encrypted_i[global_index], key_modulus[j]);
        } else {
            encrypted_i[global_index] = target;
        }
    }


    static void ski_util5(
        ConstSlice<uint64_t> t_last,
        Slice<uint64_t> t_poly_prod_i,
        size_t coeff_count,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<NTTTables> key_ntt_tables,
        size_t decomp_modulus_size,
        uint64_t qk_inv_qp,
        uint64_t qk,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> destination_i,
        bool add_inplace,
        MemoryPoolHandle pool
    ) {
        bool device = t_last.on_device();
        if (!device) {

            Buffer<uint64_t> buffer(coeff_count * 3, false, nullptr);
            Slice<uint64_t> delta = buffer.slice(0, coeff_count);
            Slice<uint64_t> c_mod_qi = buffer.slice(coeff_count, 2 * coeff_count);
            Slice<uint64_t> k = buffer.slice(2 * coeff_count, 3 * coeff_count);

            utils::modulo(t_last, plain_modulus, k);
            utils::negate_inplace(k, plain_modulus);
            if (qk_inv_qp != 1) {
                utils::multiply_scalar_inplace(k, qk_inv_qp, plain_modulus);
            }

            for (size_t j = 0; j < decomp_modulus_size; j++) {
                // delta = k mod q_i
                utils::modulo(k.as_const(), key_modulus.at(j), delta);
                // delta = k * q_k mod q_i
                utils::multiply_scalar_inplace(delta, qk, key_modulus.at(j));
                // c mod q_i
                utils::modulo(t_last, key_modulus.at(j), c_mod_qi);
                // delta = c + k * q_k mod q_i
                // c_{i} = c_{i} - delta mod q_i
                utils::add_inplace(delta, c_mod_qi.as_const(), key_modulus.at(j));
                utils::ntt_inplace(delta, coeff_count, key_ntt_tables.at(j));
                Slice<uint64_t> t_poly_prod_i_comp_j = t_poly_prod_i.slice(j * coeff_count, (j + 1) * coeff_count);
                utils::sub_inplace(t_poly_prod_i_comp_j, delta.as_const(), key_modulus.at(j));
                utils::multiply_uint64operand_inplace(t_poly_prod_i_comp_j, modswitch_factors.at(j), key_modulus.at(j));
                if (add_inplace) {
                    utils::add_inplace(destination_i.slice(j * coeff_count, (j + 1) * coeff_count), t_poly_prod_i_comp_j.as_const(), key_modulus.at(j));
                } else {
                    destination_i.slice(j * coeff_count, (j + 1) * coeff_count).copy_from_slice(t_poly_prod_i_comp_j.as_const());
                }
            }

        } else {
            Buffer<uint64_t> delta(coeff_count * decomp_modulus_size, true, pool);
            size_t block_count = utils::ceil_div(coeff_count * decomp_modulus_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_last.device_index());
            kernel_ski_util5_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, coeff_count, plain_modulus, key_modulus, 
                decomp_modulus_size, qk_inv_qp, qk,
                delta.reference()
            );
            utils::stream_sync();
            utils::ntt_inplace_p(delta.reference(), coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size));
            utils::set_device(t_poly_prod_i.device_index());
            kernel_ski_util5_step2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_prod_i, coeff_count, key_modulus, 
                decomp_modulus_size, modswitch_factors, destination_i, delta.const_reference(), add_inplace
            );
            utils::stream_sync();
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
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count) return;
        size_t i = global_index % coeff_count;
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
            utils::set_device(t_last.device_index());
            kernel_ski_util6<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, coeff_count, qk, key_modulus, decomp_modulus_size, t_ntt
            );
            utils::stream_sync();
        }
    }

    
    void kernel_ski_util6_merged(
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
            utils::set_device(t_last.device_index());
            kernel_ski_util6<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_last, coeff_count, qk, key_modulus, decomp_modulus_size, t_ntt
            );
            utils::stream_sync();
        }
    }

    __device__ static void device_ski_util5_merged_step1(
        ConstSlice<uint64_t> poly_prod_intt,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        uint64_t qk_inv_qp,
        Slice<uint64_t> delta
    ) {

        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= key_component_count * coeff_count) return;
        size_t k = global_index / coeff_count;
        size_t i = global_index % coeff_count;
        size_t t_last_offset = coeff_count * (decomp_modulus_size + 1) * k + decomp_modulus_size * coeff_count;
        
        uint64_t kt = plain_modulus->reduce(poly_prod_intt[i + t_last_offset]);
        kt = utils::negate_uint64_mod(kt, *plain_modulus);
        if (qk_inv_qp != 1) {
            kt = utils::multiply_uint64_mod(kt, qk_inv_qp, *plain_modulus);
        }

        uint64_t qk = key_modulus[key_modulus.size() - 1].value();

        for (size_t j = 0; j < decomp_modulus_size; j++) {
            uint64_t delta_result = 0;
            const Modulus& qi = key_modulus[j];
            // delta = k mod q_i
            delta_result = qi.reduce(kt);
            // delta = k * q_k mod q_i
            delta_result = utils::multiply_uint64_mod(delta_result, qk, qi);
            // c mod q_i
            uint64_t c_mod_qi = qi.reduce(poly_prod_intt[i + t_last_offset]);
            // delta = c + k * q_k mod q_i
            // c_{i} = c_{i} - delta mod q_i
            delta_result = utils::add_uint64_mod(delta_result, c_mod_qi, qi);
            delta[(k * decomp_modulus_size + j) * coeff_count + i] = delta_result;
        }

    }
    
    __global__ static void kernel_ski_util5_merged_step1(
        ConstSlice<uint64_t> poly_prod_intt,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        uint64_t qk_inv_qp,
        Slice<uint64_t> delta
    ) {
        device_ski_util5_merged_step1(poly_prod_intt, plain_modulus, key_modulus, key_component_count, decomp_modulus_size, coeff_count, qk_inv_qp, delta);
    }

    __global__ static void kernel_ski_util5_merged_step1_batched(
        utils::ConstSliceArrayRef<uint64_t> poly_prod_intt,
        ConstPointer<Modulus> plain_modulus,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        uint64_t qk_inv_qp,
        utils::SliceArrayRef<uint64_t> delta
    ) {
        for (size_t i = 0; i < poly_prod_intt.size(); i++) {
            device_ski_util5_merged_step1(poly_prod_intt[i], plain_modulus, key_modulus, key_component_count, decomp_modulus_size, coeff_count, qk_inv_qp, delta[i]);
        }
    }
    
    __device__ static void device_ski_util5_merged_step2(
        ConstSlice<uint64_t> delta_array,
        Slice<uint64_t> poly_prod,
        size_t coeff_count,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> destination,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        
        bool add_inplace = (assign_method == Evaluator::SwitchKeyDestinationAssignMethod::AddInplace) || (
            k == 0 && assign_method == Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst
        );

        for (size_t j = 0; j < decomp_modulus_size; j++) {
            size_t index = (k * decomp_modulus_size + j) * coeff_count + i;
            uint64_t delta = delta_array[index];
            uint64_t& target = poly_prod[(k * (decomp_modulus_size + 1) + j) * coeff_count + i];
            target = utils::sub_uint64_mod(target, delta, key_modulus[j]);
            target = utils::multiply_uint64operand_mod(target, modswitch_factors[j], key_modulus[j]);
            if (add_inplace) {
                destination[index] = utils::add_uint64_mod(target, destination[index], key_modulus[j]);
            } else {
                destination[index] = target;
            }
        }
    }
    
    __global__ static void kernel_ski_util5_merged_step2(
        ConstSlice<uint64_t> delta_array,
        Slice<uint64_t> poly_prod,
        size_t coeff_count,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Slice<uint64_t> destination,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        device_ski_util5_merged_step2(delta_array, poly_prod, coeff_count, key_modulus, key_component_count, decomp_modulus_size, modswitch_factors, destination, assign_method);
    }

    __global__ static void kernel_ski_util5_merged_step2_batched(
        utils::ConstSliceArrayRef<uint64_t> delta_array,
        utils::SliceArrayRef<uint64_t> poly_prod,
        size_t coeff_count,
        ConstSlice<Modulus> key_modulus,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        utils::SliceArrayRef<uint64_t> destination,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        for (size_t i = 0; i < delta_array.size(); i++) {
            device_ski_util5_merged_step2(delta_array[i], poly_prod[i], coeff_count, key_modulus, key_component_count, decomp_modulus_size, modswitch_factors, destination[i], assign_method);
        }
    }

    __device__ static void device_ski_util6_merged(
        Slice<uint64_t> poly_prod_intt,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        ConstPointer<Modulus> qk,
        ConstSlice<Modulus> key_modulus,
        Slice<uint64_t> temp_last
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= key_component_count * coeff_count) return;
        size_t i = global_index % coeff_count;
        size_t k = global_index / coeff_count;
        uint64_t qk_half = qk->value() >> 1;
        size_t offset = coeff_count * (decomp_modulus_size + 1) * k + decomp_modulus_size * coeff_count;
        size_t poly_prod_intt_i = utils::barrett_reduce_uint64(poly_prod_intt[offset + i] + qk_half, *qk);
        for (size_t j = 0; j < decomp_modulus_size; j++) {
            uint64_t result;
            const Modulus& qi = key_modulus[j];
            if (qk->value() > qi.value()) {
                result = utils::barrett_reduce_uint64(poly_prod_intt_i, qi);
            } else {
                result = poly_prod_intt_i;
            }
            uint64_t fix = qi.value() - utils::barrett_reduce_uint64(qk_half, key_modulus[j]);
            result += fix;
            temp_last[(k * decomp_modulus_size + j) * coeff_count + i] = result;
        }
    }

    __global__ static void kernel_ski_util6_merged(
        Slice<uint64_t> poly_prod_intt,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        ConstPointer<Modulus> qk,
        ConstSlice<Modulus> key_modulus,
        Slice<uint64_t> temp_last
    ) {
        device_ski_util6_merged(poly_prod_intt, key_component_count, decomp_modulus_size, coeff_count, qk, key_modulus, temp_last);
    }

    __global__ static void kernel_ski_util6_merged_batched(
        utils::SliceArrayRef<uint64_t> poly_prod_intt,
        size_t key_component_count,
        size_t decomp_modulus_size,
        size_t coeff_count,
        ConstPointer<Modulus> qk,
        ConstSlice<Modulus> key_modulus,
        utils::SliceArrayRef<uint64_t> temp_last
    ) {
        size_t i = blockIdx.y;
        device_ski_util6_merged(poly_prod_intt[i], key_component_count, decomp_modulus_size, coeff_count, qk, key_modulus, temp_last[i]);
    }

    __device__ static void device_ski_util7_merged(
        Slice<uint64_t> poly_prod,
        ConstSlice<uint64_t> temp_last,
        size_t coeff_count, 
        Slice<uint64_t> destination,
        bool is_ckks,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        for (size_t k = 0; k < key_component_count; k++) {
            size_t offset = k * decomp_modulus_size * coeff_count;
            uint64_t& dest = poly_prod[k * (decomp_modulus_size + 1) * coeff_count + j * coeff_count + i];
            uint64_t qi = key_modulus[j].value();
            dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - temp_last[offset + j * coeff_count + i];
            dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
            bool add_inplace = (assign_method == Evaluator::SwitchKeyDestinationAssignMethod::AddInplace) || (
                k == 0 && assign_method == Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst
            );
            if (add_inplace) {
                destination[offset + j * coeff_count + i] = utils::add_uint64_mod(
                    destination[offset + j * coeff_count + i], dest, key_modulus[j]
                );
            } else {
                destination[offset + j * coeff_count + i] = dest;
            }
        }
    }

    __global__ static void kernel_ski_util7_merged(
        Slice<uint64_t> poly_prod,
        ConstSlice<uint64_t> temp_last,
        size_t coeff_count, 
        Slice<uint64_t> destination,
        bool is_ckks,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        device_ski_util7_merged(poly_prod, temp_last, coeff_count, destination, is_ckks, key_component_count, decomp_modulus_size, key_modulus, modswitch_factors, assign_method);
    }
    __global__ static void kernel_ski_util7_merged_batched(
        utils::SliceArrayRef<uint64_t> poly_prod,
        utils::ConstSliceArrayRef<uint64_t> temp_last,
        size_t coeff_count, 
        utils::SliceArrayRef<uint64_t> destination,
        bool is_ckks,
        size_t key_component_count,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        Evaluator::SwitchKeyDestinationAssignMethod assign_method
    ) {
        size_t i = blockIdx.y;
        device_ski_util7_merged(poly_prod[i], temp_last[i], coeff_count, destination[i], is_ckks, key_component_count, decomp_modulus_size, key_modulus, modswitch_factors, assign_method);
    }

    __global__ static void kernel_ski_util7(
        Slice<uint64_t> t_poly_prod_i,
        ConstSlice<uint64_t> t_ntt,
        size_t coeff_count, 
        Slice<uint64_t> destination_i,
        bool is_ckks,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        bool add_inplace
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * decomp_modulus_size) return;
        size_t i = global_index % coeff_count;
        size_t j = global_index / coeff_count;
        uint64_t& dest = t_poly_prod_i[j*coeff_count + i];
        uint64_t qi = key_modulus[j].value();
        dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - t_ntt[j * coeff_count + i];
        dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
        if (add_inplace) {
            destination_i[j * coeff_count + i] = utils::add_uint64_mod(
                destination_i[j * coeff_count + i], dest, key_modulus[j]
            );
        } else {
            destination_i[j * coeff_count + i] = dest;
        }
    }

    static void ski_util7(
        Slice<uint64_t> t_poly_prod_i,
        ConstSlice<uint64_t> t_ntt,
        size_t coeff_count, 
        Slice<uint64_t> destination_i,
        bool is_ckks,
        size_t decomp_modulus_size,
        ConstSlice<Modulus> key_modulus,
        ConstSlice<MultiplyUint64Operand> modswitch_factors,
        bool add_inplace
    ) {
        bool device = t_poly_prod_i.on_device();
        if (!device) {
            for (size_t i = 0; i < coeff_count; i++) {
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    uint64_t& dest = t_poly_prod_i[j*coeff_count + i];
                    uint64_t qi = key_modulus[j].value();
                    dest += ((is_ckks) ? (qi << 2) : (qi << 1)) - t_ntt[j * coeff_count + i];
                    dest = utils::multiply_uint64operand_mod(dest, modswitch_factors[j], key_modulus[j]);
                    if (add_inplace) {
                        destination_i[j * coeff_count + i] = utils::add_uint64_mod(
                            destination_i[j * coeff_count + i], dest, key_modulus[j]
                        );
                    } else {
                        destination_i[j * coeff_count + i] = dest;
                    }
                }
            }
        } else {
            size_t block_count = utils::ceil_div(coeff_count * decomp_modulus_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(t_poly_prod_i.device_index());
            kernel_ski_util7<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                t_poly_prod_i, t_ntt, coeff_count, destination_i, is_ckks, 
                decomp_modulus_size, key_modulus, modswitch_factors, add_inplace
            );
            utils::stream_sync();
        }
    }

    void Evaluator::switch_key_internal(
        const Ciphertext& encrypted, utils::ConstSlice<uint64_t> target, 
        const KSwitchKeys& kswitch_keys, size_t kswitch_keys_index, 
        SwitchKeyDestinationAssignMethod assign_method, 
        Ciphertext& destination, MemoryPoolHandle pool
    ) const {
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
        bool is_ntt_form = encrypted.is_ntt_form();

        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        ConstSlice<Modulus> key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        ConstSlice<NTTTables> key_ntt_tables = key_context_data->small_ntt_tables();
        ConstSlice<MultiplyUint64Operand> modswitch_factors = key_context_data->rns_tool().inv_q_last_mod_q();

        const std::vector<PublicKey>& key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].as_ciphertext().polynomial_count();

        if (destination.polynomial_count() < key_component_count) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Destination should have at least same amount of polys as the key switching key.");
        }
        if (destination.polynomial_count() > key_component_count && assign_method != SwitchKeyDestinationAssignMethod::AddInplace) {
            destination.polys(key_component_count, destination.polynomial_count()).set_zero();
        }
        if (destination.parms_id() != parms_id) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Destination parms_id should match the input parms_id.");
        }
        if (!utils::device_compatible(encrypted, destination, target)) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Incompatible encryption parameters.");
        }

        for (size_t i = 0; i < key_vector.size(); i++) {
            check_no_seed("[Evaluator::switch_key_inplace_internal]", key_vector[i].as_ciphertext());
        }

        if (target.size() != decomp_modulus_size * coeff_count) {
            throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Invalid target size.");
        }
        Buffer<uint64_t> target_copied_underlying(target.size(), target.on_device(), pool);
        ConstSlice<uint64_t> target_copied(nullptr, 0, false, nullptr);

        // If target is in NTT form; switch back to normal form
        if (is_ntt_form) {
            utils::intt_p(
                target, coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size), target_copied_underlying.reference()
            );
            target_copied = target_copied_underlying.const_reference();
        } else {
            target_copied = target;
        }

        // Temporary result
        bool device = target.on_device();

        // We fuse some kernels for device only. This is not necessary for host execution.

        Buffer<uint64_t> poly_prod(key_component_count, rns_modulus_size, coeff_count, device, pool);

        if (!device) { // host

            Buffer<uint64_t> poly_lazy(key_component_count * coeff_count * 2, device, pool);
            Buffer<uint64_t> temp_ntt(coeff_count, device, pool);

            for (size_t i = 0; i < rns_modulus_size; i++) {
                size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);

                // Product of two numbers is up to 60 + 60 = 120 bits, so we can sum up to 256 of them without reduction.
                size_t lazy_reduction_summand_bound = utils::HE_MULTIPLY_ACCUMULATE_USER_MOD_MAX;
                size_t lazy_reduction_counter = lazy_reduction_summand_bound;

                // Allocate memory for a lazy accumulator (128-bit coefficients)
                poly_lazy.set_zero();

                // Multiply with keys and perform lazy reduction on product's coefficients
                for (size_t j = 0; j < decomp_modulus_size; j++) {
                    ConstSlice<uint64_t> temp_operand(nullptr, 0, device, nullptr);
                    if (is_ntt_form && (i == j)) {
                        temp_operand = target.const_slice(j * coeff_count, (j + 1) * coeff_count);
                    } else {
                        if (key_modulus[j].value() <= key_modulus[key_index].value()) {
                            temp_ntt.copy_from_slice(target_copied.const_slice(j * coeff_count, (j + 1) * coeff_count));
                        } else {
                            utils::modulo(target_copied.const_slice(j * coeff_count, (j + 1) * coeff_count), key_modulus.at(key_index), temp_ntt.reference());
                        }
                        utils::ntt_inplace(temp_ntt.reference(), coeff_count, key_ntt_tables.at(key_index));
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
                            temp_operand, key_index
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

        } else { // device

            Buffer<uint64_t> poly_lazy(key_component_count * coeff_count * 2, device, pool);
            Buffer<uint64_t> temp_ntt(rns_modulus_size, decomp_modulus_size, coeff_count, device, pool);

            utils::fgk::switch_key::set_accumulate(decomp_modulus_size, coeff_count, target_copied, temp_ntt, key_modulus);

            utils::ntt_inplace_ps(temp_ntt.reference(), rns_modulus_size, decomp_modulus_size, coeff_count, 
                utils::NTTTableIndexer::key_switching_set_products(key_ntt_tables, decomp_modulus_size));

            utils::fgk::switch_key::accumulate_products(
                decomp_modulus_size, key_component_count, coeff_count, temp_ntt.const_reference(), 
                key_modulus, kswitch_keys.get_data_ptrs(kswitch_keys_index), poly_prod.reference()
            );

        }
        
        // Accumulated products are now stored in t_poly_prod

        if (!device) { // host
            Buffer<uint64_t> temp_ntt = Buffer<uint64_t>(decomp_modulus_size, coeff_count, device, pool);
            for (size_t i = 0; i < key_component_count; i++) {
                
                bool add_inplace = (assign_method == SwitchKeyDestinationAssignMethod::AddInplace) || 
                    (i == 0 && assign_method == SwitchKeyDestinationAssignMethod::OverwriteExceptFirst);

                if (scheme == SchemeType::BGV) {
                    // qk is the special prime
                    uint64_t qk = key_modulus[key_modulus_size - 1].value();
                    uint64_t qk_inv_qp = this->context()->key_context_data().value()->rns_tool().inv_q_last_mod_t();

                    // Lazy reduction; this needs to be then reduced mod qi
                    size_t t_last_offset = coeff_count * rns_modulus_size * i + decomp_modulus_size * coeff_count;
                    Slice<uint64_t> t_last = poly_prod.slice(t_last_offset, t_last_offset + coeff_count);
                    utils::intt_inplace(t_last, coeff_count, key_ntt_tables.at(key_modulus_size - 1));
                    ConstPointer<Modulus> plain_modulus = parms.plain_modulus();

                    ski_util5(
                        t_last.as_const(), poly_prod.slice(i * coeff_count * rns_modulus_size, poly_prod.size()),
                        coeff_count, plain_modulus, key_modulus, key_ntt_tables,
                        decomp_modulus_size, qk_inv_qp, qk,
                        modswitch_factors, destination.poly(i),
                        add_inplace,
                        pool
                    );
                } else {
                    // Lazy reduction; this needs to be then reduced mod qi
                    size_t t_last_offset = coeff_count * rns_modulus_size * i + decomp_modulus_size * coeff_count;
                    Slice<uint64_t> t_last = poly_prod.slice(t_last_offset, t_last_offset + coeff_count);
                    // temp_ntt.set_zero();
                    utils::intt_inplace(t_last, coeff_count, key_ntt_tables.at(key_modulus_size - 1));

                    ski_util6(
                        t_last, coeff_count, key_modulus.at(key_modulus_size - 1),
                        key_modulus,
                        decomp_modulus_size,
                        temp_ntt.reference()
                    );
                    
                    if (is_ntt_form) {
                        utils::ntt_inplace_p(temp_ntt.reference(), coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size));
                    } else {
                        utils::intt_inplace_p(
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
                        coeff_count, destination.poly(i),
                        scheme==SchemeType::CKKS, decomp_modulus_size, key_modulus,
                        modswitch_factors,
                        add_inplace
                    );
                }
            }

        } else { // device

            Buffer<uint64_t> poly_prod_intt(key_component_count, rns_modulus_size, coeff_count, device, pool);
            
            // When is_ntt_form is true, we actually only need the INTT of the every polynomial's last component. 
            // but iteration over the polynomial and then conduct INTT will be inefficient because of multiple kernel calls.
            utils::intt_ps(
                poly_prod.const_reference(), key_component_count, rns_modulus_size, coeff_count, 
                poly_prod_intt.reference(), utils::NTTTableIndexer::key_switching_skip_finals(key_ntt_tables, decomp_modulus_size)
            );
            
            if (scheme == SchemeType::BGV) {

                Buffer<uint64_t> delta(key_component_count, decomp_modulus_size, coeff_count, device, pool);
                
                size_t total = key_component_count * coeff_count;
                size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
                utils::set_device(poly_prod_intt.device_index());
                kernel_ski_util5_merged_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    poly_prod_intt.const_reference(),
                    parms.plain_modulus(), key_modulus, key_component_count, decomp_modulus_size, coeff_count,
                    this->context()->key_context_data().value()->rns_tool().inv_q_last_mod_t(),
                    delta.reference()
                );
                utils::stream_sync();
                utils::ntt_inplace_ps(delta.reference(), key_component_count, coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size));
                utils::set_device(poly_prod.device_index());
                kernel_ski_util5_merged_step2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    delta.const_reference(), poly_prod.reference(), coeff_count, key_modulus, key_component_count, decomp_modulus_size, modswitch_factors, 
                    destination.data().reference(), assign_method
                );
                utils::stream_sync();

            } else {

                Buffer<uint64_t> temp_last(key_component_count, decomp_modulus_size, coeff_count, device, pool);
                
                size_t block_count = utils::ceil_div(key_component_count * coeff_count, utils::KERNEL_THREAD_COUNT);
                kernel_ski_util6_merged<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    poly_prod_intt.reference(),
                    key_component_count, decomp_modulus_size, coeff_count, key_modulus.at(key_modulus_size - 1),
                    key_modulus,
                    temp_last.reference()
                );
                if (is_ntt_form) {
                    utils::ntt_inplace_ps(temp_last.reference(), key_component_count, coeff_count, 
                        key_ntt_tables.const_slice(0, decomp_modulus_size));
                }
                if (!is_ntt_form) {
                    poly_prod = std::move(poly_prod_intt);
                }

                block_count = utils::ceil_div(coeff_count * decomp_modulus_size, utils::KERNEL_THREAD_COUNT);
                kernel_ski_util7_merged<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                    poly_prod.reference(),
                    temp_last.const_reference(),
                    coeff_count, destination.data().reference(),
                    scheme==SchemeType::CKKS, key_component_count, decomp_modulus_size,
                    key_modulus, modswitch_factors, assign_method
                );
                utils::stream_sync();

            }

        }
    }


    void Evaluator::switch_key_internal_batched(
        const std::vector<const Ciphertext*>& encrypted, const utils::ConstSliceVec<uint64_t>& target, 
        const KSwitchKeys& kswitch_keys, size_t kswitch_keys_index, SwitchKeyDestinationAssignMethod assign_method, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool
    ) const {
        using std::vector;
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::switch_key_internal_batched] Input and output vectors must have same size.");
        }
        if (encrypted.size() == 0) return;
        if (!this->on_device() || encrypted.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < encrypted.size(); i++) {
                this->switch_key_internal(*encrypted[i], target[i], kswitch_keys, kswitch_keys_index, assign_method, *destination[i], pool);
            }
            return;
        }
        check_no_seed_vec("[Evaluator::switch_key_internal_batched]", encrypted);
        if (!this->context()->using_keyswitching()) {
            throw std::invalid_argument("[Evaluator::switch_key_internal_batched] Keyswitching is not supported.");
        }
        if (kswitch_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::switch_key_internal_batched] Keyswitching key has incorrect parms id.");
        }
        if (kswitch_keys_index >= kswitch_keys.data().size()) {
            throw std::out_of_range("[Evaluator::switch_key_internal_batched] Key switch keys index out of range.");
        }

        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::switch_key_inplace_internal]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const EncryptionParameters& key_parms = key_context_data->parms();
        SchemeType scheme = parms.scheme();
        bool is_ntt_form = get_is_ntt_form_vec(encrypted);

        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        ConstSlice<Modulus> key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        ConstSlice<NTTTables> key_ntt_tables = key_context_data->small_ntt_tables();
        ConstSlice<MultiplyUint64Operand> modswitch_factors = key_context_data->rns_tool().inv_q_last_mod_q();

        const std::vector<PublicKey>& key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].as_ciphertext().polynomial_count();

        {
            std::vector<Slice<uint64_t>> set_zeros;
            size_t i = 0;
            for (Ciphertext* d: destination) {
                if (d->polynomial_count() < key_component_count) {
                    throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Destination should have at least same amount of polys as the key switching key.");
                }
                if (d->polynomial_count() > key_component_count && assign_method != SwitchKeyDestinationAssignMethod::AddInplace) {
                    if (d->polynomial_count() > key_component_count) {
                        set_zeros.push_back(d->polys(key_component_count, d->polynomial_count()));
                    }
                }
                if (d->parms_id() != parms_id) {
                    throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Destination parms_id should match the input parms_id.");
                }
                if (!utils::device_compatible(*encrypted[i], *d, target[i])) {
                    throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Incompatible encryption parameters.");
                }
                i++;
            }
            if (set_zeros.size() > 0) {
                utils::set_slice_b(0, set_zeros, pool);
            }
        }

        for (size_t i = 0; i < key_vector.size(); i++) {
            check_no_seed("[Evaluator::switch_key_inplace_internal]", key_vector[i].as_ciphertext());
        }

        for (const ConstSlice<uint64_t>& t: target) {
            if (t.size() != decomp_modulus_size * coeff_count) {
                throw std::invalid_argument("[Evaluator::switch_key_inplace_internal] Invalid target size.");
            }
        }
        vector<Buffer<uint64_t>> target_copied_underlying(target.size());
        vector<ConstSlice<uint64_t>> target_copied;

        // If target is in NTT form; switch back to normal form
        if (is_ntt_form) {
            for (size_t i = 0; i < target.size(); i++) {
                target_copied_underlying[i] = Buffer<uint64_t>(target[i].size(), target[i].on_device(), pool);
            }
            utils::intt_bp(
                target, coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size), batch_utils::rcollect_reference(target_copied_underlying), pool
            );
            target_copied = batch_utils::rcollect_const_reference(target_copied_underlying);
        } else {
            target_copied = target;
        }

        // We fuse some kernels for device only. This is not necessary for host execution.
        constexpr bool device = true; 
        size_t n = encrypted.size();
        vector<Buffer<uint64_t>> poly_prod(n);
        vector<Buffer<uint64_t>> poly_lazy(n);
        vector<Buffer<uint64_t>> temp_ntt(n);
        
        for (size_t i = 0; i < n; i++) {
            poly_prod[i] = Buffer<uint64_t>(key_component_count, rns_modulus_size, coeff_count, device, pool);
            poly_lazy[i] = Buffer<uint64_t>(key_component_count * coeff_count * 2, device, pool);
            temp_ntt[i] = Buffer<uint64_t>(rns_modulus_size, decomp_modulus_size, coeff_count, device, pool);
        }

        utils::fgk::switch_key::set_accumulate_batched(decomp_modulus_size, coeff_count, target_copied, temp_ntt, key_modulus, pool);

        utils::ntt_inplace_bps(
            batch_utils::rcollect_reference(temp_ntt), 
            rns_modulus_size, decomp_modulus_size, coeff_count, 
            utils::NTTTableIndexer::key_switching_set_products(key_ntt_tables, decomp_modulus_size),
            pool
        );

        utils::fgk::switch_key::accumulate_products_batched(
            decomp_modulus_size, key_component_count, coeff_count, batch_utils::rcollect_const_reference(temp_ntt), 
            key_modulus, kswitch_keys.get_data_ptrs(kswitch_keys_index), batch_utils::rcollect_reference(poly_prod),
            pool
        );

        
        // Accumulated products are now stored in t_poly_prod

        vector<Buffer<uint64_t>> poly_prod_intt(n);
        for (size_t i = 0; i < n; i++) {
            poly_prod_intt[i] = Buffer<uint64_t>(key_component_count, rns_modulus_size, coeff_count, device, pool);
        }
        
        // When is_ntt_form is true, we actually only need the INTT of the every polynomial's last component. 
        // but iteration over the polynomial and then conduct INTT will be inefficient because of multiple kernel calls.
        utils::intt_bps(
            batch_utils::rcollect_const_reference(poly_prod), key_component_count, rns_modulus_size, coeff_count, 
            batch_utils::rcollect_reference(poly_prod_intt), utils::NTTTableIndexer::key_switching_skip_finals(key_ntt_tables, decomp_modulus_size),
            pool
        );
        
        if (scheme == SchemeType::BGV) {

            vector<Buffer<uint64_t>> delta(n);
            for (size_t i = 0; i < n; i++) {
                delta[i] = Buffer<uint64_t>(key_component_count, decomp_modulus_size, coeff_count, device, pool);
            }
            
            size_t total = key_component_count * coeff_count;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(key_modulus.device_index());
            auto poly_prod_intt_batched = batch_utils::construct_batch(batch_utils::rcollect_const_reference(poly_prod_intt), pool, key_modulus);
            auto delta_batched = batch_utils::construct_batch(batch_utils::rcollect_reference(delta), pool, key_modulus);
            kernel_ski_util5_merged_step1_batched<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                poly_prod_intt_batched,
                parms.plain_modulus(), key_modulus, key_component_count, decomp_modulus_size, coeff_count,
                this->context()->key_context_data().value()->rns_tool().inv_q_last_mod_t(),
                delta_batched
            );
            utils::stream_sync();
            utils::ntt_inplace_bps(batch_utils::rcollect_reference(delta), key_component_count, coeff_count, key_ntt_tables.const_slice(0, decomp_modulus_size), pool);

            auto delta_const_batched = batch_utils::construct_batch(batch_utils::rcollect_const_reference(delta), pool, key_modulus);
            auto poly_prod_batched = batch_utils::construct_batch(batch_utils::rcollect_reference(poly_prod), pool, key_modulus);
            auto destination_batched = batch_utils::construct_batch(batch_utils::pcollect_reference(destination), pool, key_modulus);
            utils::set_device(key_modulus.device_index());
            kernel_ski_util5_merged_step2_batched<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                delta_const_batched, poly_prod_batched, coeff_count, key_modulus, key_component_count, decomp_modulus_size, modswitch_factors, 
                destination_batched, assign_method
            );
            utils::stream_sync();

        } else {

            vector<Buffer<uint64_t>> temp_last(n);
            for (size_t i = 0; i < n; i++) {
                temp_last[i] = Buffer<uint64_t>(key_component_count, decomp_modulus_size, coeff_count, device, pool);
            }
            
            size_t block_count = utils::ceil_div(key_component_count * coeff_count, utils::KERNEL_THREAD_COUNT);
            auto poly_prod_intt_batched = batch_utils::construct_batch(batch_utils::rcollect_reference(poly_prod_intt), pool, key_modulus);
            auto temp_last_reference = batch_utils::rcollect_reference(temp_last);
            auto temp_last_batched = batch_utils::construct_batch(temp_last_reference, pool, key_modulus);
            dim3 block_dims(block_count, n);
            kernel_ski_util6_merged_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                poly_prod_intt_batched,
                key_component_count, decomp_modulus_size, coeff_count, key_modulus.at(key_modulus_size - 1),
                key_modulus,
                temp_last_batched
            );
            if (is_ntt_form) {
                utils::ntt_inplace_bps(temp_last_reference, key_component_count, coeff_count, 
                    key_ntt_tables.const_slice(0, decomp_modulus_size), pool);
            }
            if (!is_ntt_form) {
                poly_prod = std::move(poly_prod_intt);
            }

            block_count = utils::ceil_div(coeff_count * decomp_modulus_size, utils::KERNEL_THREAD_COUNT);
            auto poly_prod_batched = batch_utils::construct_batch(batch_utils::rcollect_reference(poly_prod), pool, key_modulus);
            auto temp_last_const_batched = batch_utils::construct_batch(batch_utils::rcollect_const_reference(temp_last), pool, key_modulus);
            auto destination_batched = batch_utils::construct_batch(batch_utils::pcollect_reference(destination), pool, key_modulus);
            block_dims = dim3(block_count, n);
            kernel_ski_util7_merged_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                poly_prod_batched,
                temp_last_const_batched,
                coeff_count, destination_batched,
                scheme==SchemeType::CKKS, key_component_count, decomp_modulus_size,
                key_modulus, modswitch_factors, assign_method
            );
            utils::stream_sync();

        }

    }

}