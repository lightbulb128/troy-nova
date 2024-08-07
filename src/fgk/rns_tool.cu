#include "rns_tool.h"
#include "../utils/polynomial_buffer.h"
#include <cassert>

namespace troy::utils::fgk::rns_tool {

    __global__ static void kernel_fast_b_conv_m_tilde_sm_mrq(
        ConstSlice<uint64_t> input,
        size_t count,
        uint64_t m_tilde_value,
        ConstSlice<Modulus> base_q,

        ConstSlice<MultiplyUint64Operand> base_q_inv_punc,
        ConstSlice<Modulus> c1_base_out,
        ConstSlice<uint64_t> c1_base_change_matrix,
        ConstSlice<Modulus> c2_base_out,
        ConstSlice<uint64_t> c2_base_change_matrix,

        MultiplyUint64Operand neg_inv_prod_q_mod_m_tilde,
        ConstSlice<uint64_t> prod_q_mod_Bsk,
        ConstSlice<MultiplyUint64Operand> inv_m_tilde_mod_Bsk,

        Slice<uint64_t> temp_base_q,
        Slice<uint64_t> temp_base_q_mul_punc,
        Slice<uint64_t> temp_base_c1c2,
        Slice<uint64_t> output
    ) {

        size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= count) return;

        // first multiply scalar to base q
        size_t ibase_size = base_q.size();
        for (size_t i = 0; i < ibase_size; i++) {
            temp_base_q[i * count + j] = multiply_uint64_mod(
                input[i * count + j], m_tilde_value, base_q[i]
            );
        }

        // convert
        for (size_t i = 0; i < ibase_size; i++) {
            const MultiplyUint64Operand& op = base_q_inv_punc[i];
            const Modulus& base = base_q[i];
            if (op.operand == 1) {
                temp_base_q_mul_punc[j * ibase_size + i] = utils::barrett_reduce_uint64(temp_base_q[i * count + j], base);
            } else {
                temp_base_q_mul_punc[j * ibase_size + i] = utils::multiply_uint64operand_mod(temp_base_q[i * count + j], op, base);
            }
        }

        // output to c1
        size_t obase_size = c1_base_out.size();
        for (size_t i = 0; i < obase_size; i++) {
            temp_base_c1c2[i * count + j] = utils::dot_product_mod(
                temp_base_q_mul_punc.const_slice(j * ibase_size, (j + 1) * ibase_size),
                c1_base_change_matrix.const_slice(i * ibase_size, (i + 1) * ibase_size),
                c1_base_out[i]
            );
        }

        // output to c2
        obase_size = c2_base_out.size();
        size_t offset = c1_base_out.size() * count;
        for (size_t i = 0; i < obase_size; i++) {
            temp_base_c1c2[offset + i * count + j] = utils::dot_product_mod(
                temp_base_q_mul_punc.const_slice(j * ibase_size, (j + 1) * ibase_size),
                c2_base_change_matrix.const_slice(i * ibase_size, (i + 1) * ibase_size),
                c2_base_out[i]
            );
        }
        
        // sm_mrq
        const Modulus& m_tilde = c2_base_out[0];
        uint64_t m_tilde_div_2 = m_tilde.value() >> 1;
        uint64_t base_Bsk_size = c1_base_out.size();
        for (size_t i = 0; i < base_Bsk_size; i++) {
            const Modulus& modulus = *c1_base_out.at(i);
            uint64_t r_m_tilde = utils::multiply_uint64operand_mod(
                temp_base_c1c2[base_Bsk_size * count + j], 
                neg_inv_prod_q_mod_m_tilde,
                m_tilde
            );
            uint64_t temp = r_m_tilde;
            if (temp >= m_tilde_div_2) {
                temp += modulus.value() - m_tilde.value();
            }
            uint64_t& dest = output[i * count + j];
            dest = utils::multiply_uint64operand_mod(
                utils::multiply_uint64operand_add_uint64_mod(
                    temp,
                    MultiplyUint64Operand(prod_q_mod_Bsk[i], modulus),
                    temp_base_c1c2[i * count + j],
                    modulus
                ),
                inv_m_tilde_mod_Bsk[i],
                modulus
            );
        }

    }

    void fast_b_conv_m_tilde_sm_mrq(
        ConstSlice<uint64_t> input,
        size_t coeff_count,
        uint64_t m_tilde_value,
        ConstSlice<Modulus> base_q,
        const BaseConverter& base_q_to_Bsk_conv,
        const BaseConverter& base_q_to_m_tilde_conv,

        MultiplyUint64Operand neg_inv_prod_q_mod_m_tilde,
        ConstSlice<uint64_t> prod_q_mod_Bsk,
        ConstSlice<MultiplyUint64Operand> inv_m_tilde_mod_Bsk,

        Slice<uint64_t> output,
        MemoryPoolHandle pool
    ) {
        if (!input.on_device()) {
            throw std::invalid_argument("[fgk::fast_b_conv_m_tilde] input must be on device");
        }
        Buffer<uint64_t> temp_base_q(base_q.size(), coeff_count, true, pool);
        Buffer<uint64_t> temp_base_q_mul_punc(base_q.size(), coeff_count, true, pool);
        Buffer<uint64_t> temp_base_c1c2(base_q_to_Bsk_conv.output_base().size() + base_q_to_m_tilde_conv.output_base().size(), coeff_count, true, pool);

        size_t block_count = ceil_div(coeff_count, KERNEL_THREAD_COUNT);
        utils::set_device(output.device_index());
        kernel_fast_b_conv_m_tilde_sm_mrq<<<block_count, KERNEL_THREAD_COUNT>>>(
            input, coeff_count, m_tilde_value, base_q,
            base_q_to_Bsk_conv.input_base().inv_punctured_product_mod_base(), 
            
            base_q_to_Bsk_conv.output_base().base(),
            base_q_to_Bsk_conv.base_change_matrix(),
            base_q_to_m_tilde_conv.output_base().base(),
            base_q_to_m_tilde_conv.base_change_matrix(),

            neg_inv_prod_q_mod_m_tilde,
            prod_q_mod_Bsk,
            inv_m_tilde_mod_Bsk,
            
            temp_base_q.reference(),
            temp_base_q_mul_punc.reference(),
            temp_base_c1c2.reference(),
            output
        );
        utils::stream_sync();
    }

    __global__ static void kernel_fast_floor_fast_b_conv_sk(
        ConstSlice<uint64_t> input_q, 
        ConstSlice<uint64_t> input_Bsk, 

        size_t count,
        size_t dest_size,

        ConstSlice<Modulus> base_q,
        ConstSlice<Modulus> base_Bsk,
        ConstSlice<Modulus> base_B,
        ConstSlice<Modulus> base_m_sk,
        ConstPointer<Modulus> plain_modulus,

        ConstSlice<MultiplyUint64Operand> q_inv_punc,
        ConstSlice<MultiplyUint64Operand> B_inv_punc,
        ConstSlice<uint64_t> q_to_Bsk_base_change_matrix,
        ConstSlice<uint64_t> B_to_q_base_change_matrix,
        ConstSlice<uint64_t> B_to_m_sk_base_change_matrix,

        ConstSlice<MultiplyUint64Operand> inv_prod_q_mod_Bsk,
        MultiplyUint64Operand inv_prod_B_mod_m_sk,
        ConstSlice<uint64_t> prod_B_mod_q,
        
        Slice<uint64_t> temp_q_Bsk,
        Slice<uint64_t> temp_q_conv,
        Slice<uint64_t> temp_Bsk,
        Slice<uint64_t> temp_B_conv,
        Slice<uint64_t> temp_m_sk,
        Slice<uint64_t> destination
    ) {
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= count) return;

        size_t base_q_size = base_q.size();
        size_t base_Bsk_size = base_Bsk.size();
        size_t base_B_size = base_B.size();
        size_t base_m_sk_size = base_m_sk.size();
        uint64_t plain_modulus_value = plain_modulus->value();
        const Modulus& m_sk = base_m_sk[0];
        uint64_t m_sk_value = m_sk.value();
        uint64_t m_sk_div_2 = m_sk_value >> 1;

        for (size_t k = 0; k < dest_size; k++) {
            
            // multiply scalar
            for (size_t i = 0; i < base_q_size; i++) {
                temp_q_Bsk[i * count + j] = utils::multiply_uint64_mod(
                    input_q[(i + k * base_q_size) * count + j], 
                    plain_modulus_value, base_q[i]
                );
            }
            for (size_t i = 0; i < base_Bsk_size; i++) {
                temp_q_Bsk[(i + base_q_size) * count + j] = utils::multiply_uint64_mod(
                    input_Bsk[(i + k * base_Bsk_size) * count + j],
                    plain_modulus_value, base_Bsk[i]
                );
            }

            // fast floor - fast convert array
            for (size_t i = 0; i < base_q_size; i++) {
                const MultiplyUint64Operand& op = q_inv_punc[i];
                const Modulus& base = base_q[i];
                if (op.operand == 1) {
                    temp_q_conv[j * base_q_size + i] = utils::barrett_reduce_uint64(temp_q_Bsk[i * count + j], base);
                } else {
                    temp_q_conv[j * base_q_size + i] = utils::multiply_uint64operand_mod(temp_q_Bsk[i * count + j], op, base);
                }
            }
            for (size_t i = 0; i < base_Bsk_size; i++) {
                temp_Bsk[i * count + j] = utils::dot_product_mod(
                    temp_q_conv.const_slice(j * base_q_size, (j + 1) * base_q_size),
                    q_to_Bsk_base_change_matrix.const_slice(i * base_q_size, (i + 1) * base_q_size),
                    base_Bsk[i]
                );
            }

            // fast floor
            for (size_t i = 0; i < base_Bsk_size; i++) {
                size_t global_index = i * count + j;
                uint64_t& dest = temp_Bsk[global_index];
                dest = utils::multiply_uint64operand_mod(
                    temp_q_Bsk[global_index + base_q_size * count] + base_Bsk[i].value() - dest,
                    inv_prod_q_mod_Bsk[i],
                    base_Bsk[i]
                );
            }

            // fast_b_conv_sk
            Slice<uint64_t> dest_i = destination.slice(k * count * base_q_size, (k + 1) * count * base_q_size);

            // base b to q
            for (size_t i = 0; i < base_B_size; i++) {
                const MultiplyUint64Operand& op = B_inv_punc[i];
                const Modulus& base = base_B[i];
                if (op.operand == 1) {
                    temp_B_conv[j * base_B_size + i] = utils::barrett_reduce_uint64(temp_Bsk[i * count + j], base);
                } else {
                    temp_B_conv[j * base_B_size + i] = utils::multiply_uint64operand_mod(temp_Bsk[i * count + j], op, base);
                }
            }
            for (size_t i = 0; i < base_q_size; i++) {
                dest_i[i * count + j] = utils::dot_product_mod(
                    temp_B_conv.const_slice(j * base_B_size, (j + 1) * base_B_size),
                    B_to_q_base_change_matrix.const_slice(i * base_B_size, (i + 1) * base_B_size),
                    base_q[i]
                );
            }

            // base b to m_sk
            for (size_t i = 0; i < base_m_sk_size; i++) {
                temp_m_sk[i * count + j] = utils::dot_product_mod(
                    temp_B_conv.const_slice(j * base_B_size, (j + 1) * base_B_size),
                    B_to_m_sk_base_change_matrix.const_slice(i * base_B_size, (i + 1) * base_B_size),
                    base_m_sk[i]
                );
            }

            // b_conv_sk
            for (size_t i = 0; i < base_q_size; i++) {
                uint64_t alpha_sk = multiply_uint64operand_mod(
                    temp_m_sk[j] + (m_sk_value - temp_Bsk[base_B_size * count + j]),
                    inv_prod_B_mod_m_sk,
                    m_sk
                );
                const Modulus& modulus = *base_q.at(i);
                MultiplyUint64Operand prod_B_mod_q_elt(prod_B_mod_q[i], modulus);
                MultiplyUint64Operand neg_prod_B_mod_q_elt(modulus.value() - prod_B_mod_q[i], modulus);
                uint64_t& dest = dest_i[i * count + j];
                if (alpha_sk > m_sk_div_2) {
                    dest = utils::multiply_uint64operand_add_uint64_mod(
                        utils::negate_uint64_mod(alpha_sk, m_sk), prod_B_mod_q_elt, dest, modulus
                    );
                } else {
                    dest = utils::multiply_uint64operand_add_uint64_mod(
                        alpha_sk, neg_prod_B_mod_q_elt, dest, modulus
                    );
                }
            }
        }
    }
    
    void fast_floor_fast_b_conv_sk(

        ConstSlice<uint64_t> input_q, 
        ConstSlice<uint64_t> input_Bsk, 

        const RNSTool& rns_tool,
        size_t dest_size,
        
        Slice<uint64_t> destination, 
        MemoryPoolHandle pool

    ) {
        if (!input_q.on_device()) {
            throw std::invalid_argument("[fgk::fast_b_conv_m_tilde] input must be on device");
        }

        size_t count = rns_tool.coeff_count();
        size_t base_q_size = rns_tool.base_q().size();
        size_t base_Bsk_size = rns_tool.base_Bsk().size();
        Buffer<uint64_t> temp_q_Bsk(base_q_size + base_Bsk_size, count, true, pool);
        Buffer<uint64_t> temp_q_conv(base_q_size, count, true, pool);
        Buffer<uint64_t> temp_Bsk(base_Bsk_size, count, true, pool);
        Buffer<uint64_t> temp_B_conv(rns_tool.base_B().size(), count, true, pool);
        Buffer<uint64_t> temp_m_sk(1, count, true, pool);

        size_t block_count = ceil_div(count, KERNEL_THREAD_COUNT);
        const BaseConverter& q_to_Bsk_conv = rns_tool.base_q_to_Bsk_conv();
        const BaseConverter& B_to_q_conv = rns_tool.base_B_to_q_conv();
        const BaseConverter& B_to_m_sk_conv = rns_tool.base_B_to_m_sk_conv();

        utils::set_device(destination.device_index());
        kernel_fast_floor_fast_b_conv_sk<<<block_count, KERNEL_THREAD_COUNT>>>(
            input_q, input_Bsk, count, dest_size,
            q_to_Bsk_conv.input_base().base(),
            q_to_Bsk_conv.output_base().base(),
            B_to_q_conv.input_base().base(),
            B_to_m_sk_conv.output_base().base(),
            rns_tool.t(),
            q_to_Bsk_conv.input_base().inv_punctured_product_mod_base(),
            B_to_q_conv.input_base().inv_punctured_product_mod_base(),
            q_to_Bsk_conv.base_change_matrix(),
            B_to_q_conv.base_change_matrix(),
            B_to_m_sk_conv.base_change_matrix(),
            rns_tool.inv_prod_q_mod_Bsk(),
            rns_tool.inv_prod_B_mod_m_sk(),
            rns_tool.prod_B_mod_q(),
            temp_q_Bsk.reference(),
            temp_q_conv.reference(),
            temp_Bsk.reference(),
            temp_B_conv.reference(),
            temp_m_sk.reference(),
            destination
        );
        utils::stream_sync();
        
    }

}
