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
            utils::set_device(op1.device_index());
            kernel_dyadic_convolute<<<block_count, KERNEL_THREAD_COUNT>>>(
                op1, op2, op1_pcount, op2_pcount, moduli, degree, result
            );
            utils::stream_sync();
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
            utils::set_device(op.device_index());
            kernel_dyadic_square<<<block_count, KERNEL_THREAD_COUNT>>>(
                op, moduli, degree, result
            );
            utils::stream_sync();
        }
    }
    
    __device__ static void device_dyadic_broadcast_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> poly2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result, bool accumulate) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            size_t poly2_idx = idx % (moduli.size() * degree);
            uint64_t r = multiply_uint64_mod(polys1[idx], poly2[poly2_idx], moduli[j]);
            if (accumulate) result[idx] = add_uint64_mod(result[idx], r, moduli[j]);
            else result[idx] = r;
        }
    }

    __global__ static void kernel_dyadic_broadcast_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> poly2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result, bool accumulate) {
        device_dyadic_broadcast_product_ps(polys1, poly2, pcount, degree, moduli, result, accumulate);
    }

    __global__ static void kernel_dyadic_broadcast_product_bps(ConstSliceArrayRef<uint64_t> polys1, ConstSliceArrayRef<uint64_t> poly2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result, bool accumulate) {
        for (size_t i = 0; i < polys1.size(); i++) {
            device_dyadic_broadcast_product_ps(polys1[i], poly2[i], pcount, degree, moduli, result[i], accumulate);
        }
    }

    void dyadic_broadcast_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> poly2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys1, poly2, moduli, result)) {
            throw std::runtime_error("[dyadic_product_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_dyadic_broadcast_product_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, poly2, pcount, degree, moduli, result, false);
            utils::stream_sync();
        } else {
            size_t c = moduli.size() * degree;
            for (size_t i = 0; i < pcount; i++) {
                dyadic_product_p(
                    polys1.const_slice(c * i, c * (i + 1)),
                    poly2,
                    degree,
                    moduli,
                    result.slice(c * i, c * (i + 1))
                );
            }
        }
    }

    void dyadic_broadcast_product_accumulate_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> poly2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys1, poly2, moduli, result)) {
            throw std::runtime_error("[dyadic_product_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_dyadic_broadcast_product_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, poly2, pcount, degree, moduli, result, true);
            utils::stream_sync();
        } else {
            size_t c = moduli.size() * degree;
            Array<uint64_t> temp = Array<uint64_t>::create_uninitialized(c, false);
            for (size_t i = 0; i < pcount; i++) {
                dyadic_product_p(
                    polys1.const_slice(c * i, c * (i + 1)),
                    poly2,
                    degree,
                    moduli,
                    temp.reference()
                );
                add_inplace_p(
                    result.slice(c * i, c * (i + 1)),
                    temp.const_reference(),
                    degree,
                    moduli
                );
            }
        }
    }

    void dyadic_broadcast_product_bps(
        const ConstSliceVec<uint64_t>& op1,
        const ConstSliceVec<uint64_t>& op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli,
        const SliceVec<uint64_t>& result,
        MemoryPoolHandle pool
    ) {
        if (op1.size() != op2.size() || op1.size() != result.size()) {
            throw std::invalid_argument("[fgk::dyadic_broadcast_product_bps] Size mismatch");
        }
        if (result.size() == 0) return;
        bool device = moduli.on_device();
        if (!device || result.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < result.size(); i++) {
                dyadic_broadcast_product_ps(
                    op1[i], op2[i], op1_pcount, degree, moduli, result[i]
                );
            }
        } else {
            size_t block_count = ceil_div<size_t>(op1_pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            auto op1_batch = batch_utils::construct_batch(op1, pool, moduli);
            auto op2_batch = batch_utils::construct_batch(op2, pool, moduli);
            auto result_batch = batch_utils::construct_batch(result, pool, moduli);
            utils::set_device(moduli.device_index());
            kernel_dyadic_broadcast_product_bps<<<block_count, KERNEL_THREAD_COUNT>>>(
                op1_batch, op2_batch, op1_pcount, degree, moduli, result_batch, false
            );
            utils::stream_sync();
        }
    }
    
    void dyadic_broadcast_product_accumulate_bps(
        const ConstSliceVec<uint64_t>& op1,
        const ConstSliceVec<uint64_t>& op2,
        size_t op1_pcount,
        size_t degree,
        ConstSlice<Modulus> moduli,
        const SliceVec<uint64_t>& result,
        MemoryPoolHandle pool
    ) {
        if (op1.size() != op2.size() || op1.size() != result.size()) {
            throw std::invalid_argument("[fgk::dyadic_broadcast_product_bps] Size mismatch");
        }
        if (result.size() == 0) return;
        bool device = moduli.on_device();
        if (!device || result.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < result.size(); i++) {
                dyadic_broadcast_product_accumulate_ps(
                    op1[i], op2[i], op1_pcount, degree, moduli, result[i]
                );
            }
        } else {
            size_t block_count = ceil_div<size_t>(op1_pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            auto op1_batch = batch_utils::construct_batch(op1, pool, moduli);
            auto op2_batch = batch_utils::construct_batch(op2, pool, moduli);
            auto result_batch = batch_utils::construct_batch(result, pool, moduli);
            utils::set_device(moduli.device_index());
            kernel_dyadic_broadcast_product_bps<<<block_count, KERNEL_THREAD_COUNT>>>(
                op1_batch, op2_batch, op1_pcount, degree, moduli, result_batch, true
            );
            utils::stream_sync();
        }
    }
    

}