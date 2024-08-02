#include "dyadic_convolute.h"
#include "../utils/polynomial_buffer.h"

namespace troy::utils::fgk::dyadic_convolute {

    __global__ static void kernel_dyadic_convolute(
        ConstSlice<uint64_t> op1,
        ConstSlice<uint64_t> op2,
        size_t op1_pcount,
        size_t op2_pcount,
        ConstSlice<Modulus> moduli,
        size_t degree,
        Slice<uint64_t> result
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t result_pcount = op1_pcount + op2_pcount - 1;
        if (global_index >= moduli.size() * degree * result_pcount) return;
        size_t coeff_id = global_index % degree;
        size_t modulus_id = (global_index / degree) % moduli.size();
        size_t poly_id = global_index / (degree * moduli.size());
        size_t poly_offset = modulus_id * degree + coeff_id;
        const Modulus& modulus = moduli[modulus_id];

        size_t pc = moduli.size() * degree;

        size_t i_start = (poly_id + 1 > op2_pcount) ? (poly_id + 1 - op2_pcount) : 0;
        size_t i_end = (op1_pcount - 1 < poly_id) ? op1_pcount - 1 : poly_id;

        uint64_t accumulated = 0;

        for (size_t i = i_start; i <= i_end; i++) {
            size_t j = poly_id - i;
            uint64_t a = op1[poly_offset + i * pc];
            uint64_t b = op2[poly_offset + j * pc];
            uint64_t c = multiply_uint64_mod(a, b, modulus);
            accumulated = add_uint64_mod(accumulated, c, modulus);
        }

        result[poly_offset + poly_id * pc] = accumulated;
         
    }

    void dyadic_convolute(
        ConstSlice<uint64_t> op1,
        ConstSlice<uint64_t> op2,
        size_t op1_pcount,
        size_t op2_pcount,
        ConstSlice<Modulus> moduli,
        size_t degree,
        Slice<uint64_t> result,
        MemoryPoolHandle pool
    ) {
        if (!device_compatible(op1, op2, moduli, result)) {
            throw std::invalid_argument("[fgk::dyadic_convolute] Device incompatible");
        }
        if (result.size() != (op1_pcount + op2_pcount - 1) * degree * moduli.size()) {
            throw std::invalid_argument("[fgk::dyadic_convolute] Result size mismatch");
        }
        size_t m = moduli.size();
        size_t pc = m * degree;
        if (!op1.on_device()) {
            Buffer<uint64_t> temp(m, degree, op1.on_device(), pool);
            result.set_zero();
            for (size_t i = 0; i < op1_pcount; i++) {
                for (size_t j = 0; j < op2_pcount; j++) {
                    size_t k = i + j;
                    // std::cout << "i: " << i << ", j: " << j << ", k: " << k << std::endl;
                    dyadic_product_p(
                        op1.const_slice(pc * i, pc * (i + 1)),
                        op2.const_slice(pc * j, pc * (j + 1)),
                        degree, moduli, temp.reference()
                    );
                    add_inplace_p(
                        result.slice(pc * k, pc * (k + 1)),
                        temp.const_reference(),
                        degree,
                        moduli
                    );
                }
            }
        } else {
            size_t total_count = m * degree * (op1_pcount + op2_pcount - 1);
            size_t block_count = ceil_div(total_count, KERNEL_THREAD_COUNT);
            kernel_dyadic_convolute<<<block_count, KERNEL_THREAD_COUNT>>>(
                op1, op2, op1_pcount, op2_pcount, moduli, degree, result
            );
        }
    }

    __global__ static void kernel_dyadic_square(
        ConstSlice<uint64_t> op,
        ConstSlice<Modulus> moduli,
        size_t degree,
        Slice<uint64_t> result
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= moduli.size() * degree) return;
        size_t coeff_id = global_index % degree;
        size_t modulus_id = global_index / degree;
        size_t poly_offset = modulus_id * degree + coeff_id;
        const Modulus& modulus = moduli[modulus_id];

        size_t pc = moduli.size() * degree;

        uint64_t c0 = op[poly_offset];
        uint64_t c1 = op[poly_offset + pc];
        result[poly_offset] = multiply_uint64_mod(c0, c0, modulus);
        uint64_t cross = multiply_uint64_mod(c0, c1, modulus);
        cross = add_uint64_mod(cross, cross, modulus);
        result[poly_offset + pc] = cross;
        result[poly_offset + 2 * pc] = multiply_uint64_mod(c1, c1, modulus);
    }

    void dyadic_square(
        ConstSlice<uint64_t> op,
        ConstSlice<Modulus> moduli,
        size_t degree,
        Slice<uint64_t> result
    ) {
        if (!device_compatible(op, moduli, result)) {
            throw std::invalid_argument("[fgk::dyadic_convolute] Device incompatible");
        }
        if (result.size() != 3 * degree * moduli.size()) {
            throw std::invalid_argument("[fgk::dyadic_convolute] Result size mismatch");
        }
        size_t m = moduli.size();
        size_t pc = m * degree;
        if (!op.on_device()) {
            ConstSlice<uint64_t> c0 = op.const_slice(0 * pc, 1 * pc);
            ConstSlice<uint64_t> c1 = op.const_slice(1 * pc, 2 * pc);
            Slice<uint64_t> r0 = result.slice(0 * pc, 1 * pc);
            Slice<uint64_t> r1 = result.slice(1 * pc, 2 * pc);
            Slice<uint64_t> r2 = result.slice(2 * pc, 3 * pc);
            result.set_zero();
            dyadic_product_p(c0, c0, degree, moduli, r0);
            dyadic_product_p(c0, c1, degree, moduli, r1);
            add_inplace_p(r1, r1.as_const(), degree, moduli);
            dyadic_product_p(c1, c1, degree, moduli, r2);
        } else {
            size_t total_count = m * degree;
            size_t block_count = ceil_div(total_count, KERNEL_THREAD_COUNT);
            kernel_dyadic_square<<<block_count, KERNEL_THREAD_COUNT>>>(
                op, moduli, degree, result
            );
        }
    }

}