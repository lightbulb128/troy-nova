#include "constants.h"
#include "memory_pool.h"
#include "poly_small_mod.h"
#include "../batch_utils.h"
#include <cstdint>

namespace troy {namespace utils {

    using batch_utils::construct_batch;

    static __global__ void kernel_copy_slice_b(ConstSliceArrayRef<uint64_t> from, SliceArrayRef<uint64_t> to) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i = blockIdx.y;
        ConstSlice<uint64_t> from_i = from[i];
        Slice<uint64_t> to_i = to[i];
        if (idx < from_i.size()) {
            to_i[idx] = from_i[idx];
        }
    }
    
    static __global__ void kernel_copy_slice_b(ConstSliceArrayRef<uint8_t> from, SliceArrayRef<uint8_t> to) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i = blockIdx.y;
        ConstSlice<uint8_t> from_i = from[i];
        Slice<uint8_t> to_i = to[i];
        if (idx < from_i.size()) {
            to_i[idx] = from_i[idx];
        }
    }

    static __global__ void kernel_set_slice(uint64_t value, Slice<uint64_t> to) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < to.size()) {
            to[idx] = value;
        }
    }

    static __global__ void kernel_set_slice_b(uint64_t value, SliceArrayRef<uint64_t> to) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i = blockIdx.y;
        if (idx < to[i].size()) {
            to[i][idx] = value;
        }
    }

    void copy_slice_b(const ConstSliceVec<uint64_t>& from, const SliceVec<uint64_t>& to, MemoryPoolHandle pool) {
        if (from.size() != to.size()) {
            throw std::runtime_error("[copy_slice_b] from and to must have the same size");
        }
        if (from.size() == 0) return;
        auto device_reference = from[0];
        bool device = device_reference.on_device();
        if (!device || from.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < from.size(); i++) {
                to[i].copy_from_slice(from[i]);
            }
        } else {
            ConstSliceArray<uint64_t> from_arr = construct_batch(from, pool, device_reference);
            SliceArray<uint64_t> to_arr = construct_batch(to, pool, device_reference);
            size_t block_count = ceil_div<size_t>(device_reference.size(), KERNEL_THREAD_COUNT);
            set_device(device_reference.device_index());
            dim3 block_dims(block_count, from.size());
            kernel_copy_slice_b<<<block_dims, KERNEL_THREAD_COUNT>>>(from_arr, to_arr);
            utils::stream_sync();
        }
    }

    
    void copy_slice_b(const ConstSliceVec<uint8_t>& from, const SliceVec<uint8_t>& to, MemoryPoolHandle pool) {
        if (from.size() != to.size()) {
            throw std::runtime_error("[copy_slice_b] from and to must have the same size");
        }
        if (from.size() == 0) return;
        auto device_reference = from[0];
        bool device = device_reference.on_device();
        if (!device || from.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < from.size(); i++) {
                to[i].copy_from_slice(from[i]);
            }
        } else {
            ConstSliceArray<uint8_t> from_arr = construct_batch(from, pool, device_reference);
            SliceArray<uint8_t> to_arr = construct_batch(to, pool, device_reference);
            size_t block_count = ceil_div<size_t>(device_reference.size(), KERNEL_THREAD_COUNT);
            set_device(device_reference.device_index());
            dim3 block_dims(block_count, from.size());
            kernel_copy_slice_b<<<block_dims, KERNEL_THREAD_COUNT>>>(from_arr, to_arr);
            utils::stream_sync();
        }
    }

    void set_slice_b(const uint64_t value, const SliceVec<uint64_t>& to, MemoryPoolHandle pool) {
        if (to.size() == 0) return;
        auto device_reference = to[0];
        bool device = device_reference.on_device();
        if (!device || to.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < to.size(); i++) {
                Slice<uint64_t> to_i = to[i];
                if (!device) {
                    for (size_t j = 0; j < to[i].size(); j++) {
                        to_i[j] = value;
                    }
                } else {
                    size_t block_count = ceil_div<size_t>(to[i].size(), KERNEL_THREAD_COUNT);
                    set_device(device_reference.device_index());
                    kernel_set_slice<<<block_count, KERNEL_THREAD_COUNT>>>(value, to_i);
                    utils::stream_sync();
                }
            }
        } else {
            SliceArray<uint64_t> to_arr = construct_batch(to, pool, device_reference);
            size_t block_count = ceil_div<size_t>(device_reference.size(), KERNEL_THREAD_COUNT);
            set_device(device_reference.device_index());
            dim3 block_dims(block_count, to.size());
            kernel_set_slice_b<<<block_dims, KERNEL_THREAD_COUNT>>>(value, to_arr);
            utils::stream_sync();
        }
    }

    void host_modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    result[idx] = moduli[j].reduce(polys[idx]);
                }
            }
        }
    }

    static __device__ void device_modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            // size_t i = idx / (moduli.size() * degree);
            size_t j = (idx / degree) % moduli.size();
            // size_t k = idx % degree;
            result[idx] = moduli[j].reduce(polys[idx]);
        }
    }
    static __global__ void kernel_modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_modulo_ps(polys, pcount, degree, moduli, result);
    }
    static __global__ void kernel_modulo_bps(ConstSliceArrayRef<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        for (size_t i = 0; i < polys.size(); i++) {
            device_modulo_ps(polys[i], pcount, degree, moduli, result[i]);
        }
    }

    void modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, moduli, result)) {
            throw std::runtime_error("[modulo_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            set_device(result.device_index());
            kernel_modulo_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_modulo_ps(polys, pcount, degree, moduli, result);
        }
    }

    void modulo_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        bool device = moduli.on_device();
        if (polys.size() != result.size()) {
            throw std::runtime_error("[modulo_bps] polys and result must have the same size");
        }
        if (!device || polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys.size(); i++) {
                modulo_ps(polys[i], pcount, degree, moduli, result[i]);
            }
        } else {
            ConstSliceArray<uint64_t> polys_arr = construct_batch(polys, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            set_device(moduli.device_index());
            kernel_modulo_bps<<<block_count, KERNEL_THREAD_COUNT>>>(polys_arr, pcount, degree, moduli, result_arr);
            utils::stream_sync();
        }
    }

    void host_negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    result[idx] = polys[idx] == 0 ? 0 : moduli[j].value() - polys[idx];
                }
            }
        }
    }

    static __device__ void device_negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            result[idx] = polys[idx] == 0 ? 0 : moduli[(idx / degree) % moduli.size()].value() - polys[idx];
        }
    }
    static __global__ void kernel_negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_negate_ps(polys, pcount, degree, moduli, result);
    }
    static __global__ void kernel_negate_bps(ConstSliceArrayRef<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        for (size_t i = 0; i < polys.size(); i++) {
            device_negate_ps(polys[i], pcount, degree, moduli, result[i]);
        }
    }

    void negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, moduli, result)) {
            throw std::runtime_error("[negate_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t total = pcount * moduli.size() * degree;
            size_t thread_count = min(total, KERNEL_THREAD_COUNT);
            size_t block_count = ceil_div<size_t>(total, thread_count);
            utils::set_device(result.device_index());
            kernel_negate_ps<<<block_count, thread_count>>>(polys, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_negate_ps(polys, pcount, degree, moduli, result);
        }
    }
    void negate_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        bool device = moduli.on_device();
        if (polys.size() != result.size()) {
            throw std::runtime_error("[negate_bps] polys and result must have the same size");
        }
        if (!device || polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys.size(); i++) {
                negate_ps(polys[i], pcount, degree, moduli, result[i]);
            }
        } else {
            ConstSliceArray<uint64_t> polys_arr = construct_batch(polys, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            set_device(moduli.device_index());
            kernel_negate_bps<<<block_count, KERNEL_THREAD_COUNT>>>(polys_arr, pcount, degree, moduli, result_arr);
            utils::stream_sync();
        }
    }

    void host_add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    size_t p = polys1[idx] + polys2[idx];
                    result[idx] = p >= moduli[j].value() ? p - moduli[j].value() : p;
                }
            }
        }
    }

    static __device__ void device_add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            size_t p = polys1[idx] + polys2[idx];
            result[idx] = p >= moduli[j].value() ? p - moduli[j].value() : p;
        }
    }
    static __global__ void kernel_add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_add_ps(polys1, polys2, pcount, degree, moduli, result);
    }
    static __global__ void kernel_add_bps(ConstSliceArrayRef<uint64_t> polys1, ConstSliceArrayRef<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        size_t i = blockIdx.y;
        device_add_ps(polys1[i], polys2[i], pcount, degree, moduli, result[i]);
    }

    void add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys1.on_device(), polys2.on_device(), moduli.on_device())) {
            throw std::runtime_error("[add_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_add_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_add_ps(polys1, polys2, pcount, degree, moduli, result);
        }
    }
    void add_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        bool device = moduli.on_device();
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) {
            throw std::runtime_error("[add_bps] polys1, polys2, and result must have the same size");
        }
        if (!device || polys1.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys1.size(); i++) {
                add_ps(polys1[i], polys2[i], pcount, degree, moduli, result[i]);
            }
        } else {
            ConstSliceArray<uint64_t> polys1_arr = construct_batch(polys1, pool, moduli);
            ConstSliceArray<uint64_t> polys2_arr = construct_batch(polys2, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            set_device(moduli.device_index());
            dim3 block_dims(block_count, polys1.size());
            kernel_add_bps<<<block_dims, KERNEL_THREAD_COUNT>>>(polys1_arr, polys2_arr, pcount, degree, moduli, result_arr);
            utils::stream_sync();
        }
    }

    void host_sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    result[idx] = polys1[idx] >= polys2[idx] ? polys1[idx] - polys2[idx] : moduli[j].value() - polys2[idx] + polys1[idx];
                }
            }
        }
    }

    static __device__ void device_sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = polys1[idx] >= polys2[idx] ? polys1[idx] - polys2[idx] : moduli[j].value() - polys2[idx] + polys1[idx];
        }
    }
    static __global__ void kernel_sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_sub_ps(polys1, polys2, pcount, degree, moduli, result);
    }
    static __global__ void kernel_sub_bps(ConstSliceArrayRef<uint64_t> polys1, ConstSliceArrayRef<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        size_t i = blockIdx.y;
        device_sub_ps(polys1[i], polys2[i], pcount, degree, moduli, result[i]);
    }

    void sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys1, polys2, moduli, result)) {
            throw std::runtime_error("[sub_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_sub_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_sub_ps(polys1, polys2, pcount, degree, moduli, result);
        }
    }
    void sub_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        bool device = moduli.on_device();
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) {
            throw std::runtime_error("[sub_bps] polys1, polys2, and result must have the same size");
        }
        if (!device || polys1.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys1.size(); i++) {
                sub_ps(polys1[i], polys2[i], pcount, degree, moduli, result[i]);
            }
        } else {
            ConstSliceArray<uint64_t> polys1_arr = construct_batch(polys1, pool, moduli);
            ConstSliceArray<uint64_t> polys2_arr = construct_batch(polys2, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            set_device(moduli.device_index());
            dim3 block_dims(block_count, polys1.size());
            kernel_sub_bps<<<block_dims, KERNEL_THREAD_COUNT>>>(polys1_arr, polys2_arr, pcount, degree, moduli, result_arr);
            utils::stream_sync();
        }
    }

    static void host_add_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree_result; k++) {
                    size_t idx1 = i * moduli.size() * degree1 + j * degree1 + k;
                    size_t idx2 = i * moduli.size() * degree2 + j * degree2 + k;
                    size_t idx = i * moduli.size() * degree_result + j * degree_result + k;
                    uint64_t c1 = k < degree1 ? polys1[idx1] : 0;
                    uint64_t c2 = k < degree2 ? polys2[idx2] : 0;
                    uint64_t p = c1 + c2;
                    result[idx] = p >= moduli[j].value() ? p - moduli[j].value() : p;
                }
            }
        }
    }

    static __device__ void device_add_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree_result) {
            size_t i = idx / (moduli.size() * degree_result);
            size_t j = (idx / degree_result) % moduli.size();
            size_t k = idx % degree_result;
            size_t idx1 = i * moduli.size() * degree1 + j * degree1 + k;
            size_t idx2 = i * moduli.size() * degree2 + j * degree2 + k;
            uint64_t c1 = k < degree1 ? polys1[idx1] : 0;
            uint64_t c2 = k < degree2 ? polys2[idx2] : 0;
            uint64_t p = c1 + c2;
            result[idx] = p >= moduli[j].value() ? p - moduli[j].value() : p;
        }
    }
    static __global__ void kernel_add_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        device_add_partial_ps(polys1, polys2, pcount, degree1, degree2, moduli, result, degree_result);
    }
    static __global__ void kernel_add_partial_bps(ConstSliceArrayRef<uint64_t> polys1, ConstSliceArrayRef<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result, size_t degree_result) {
        for (size_t i = 0; i < polys1.size(); i++) {
            device_add_partial_ps(polys1[i], polys2[i], pcount, degree1, degree2, moduli, result[i], degree_result);
        }
    }

    void add_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        if (!device_compatible(polys1, polys2, moduli, result)) {
            throw std::runtime_error("[add_partial_ps] All inputs must be on the same device");
        }
        if (degree_result < degree1 || degree_result < degree2) {
            throw std::runtime_error("[add_partial_ps] degree_result must be at least degree1 and degree2");
        }
        if (result.on_device()) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree_result, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_add_partial_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree1, degree2, moduli, result, degree_result);
            utils::stream_sync();
        } else {
            host_add_partial_ps(polys1, polys2, pcount, degree1, degree2, moduli, result, degree_result);
        }
    }
    void add_partial_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool) {
        bool device = moduli.on_device();
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) {
            throw std::runtime_error("[add_partial_bps] polys1, polys2, and result must have the same size");
        }
        if (!device || polys1.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys1.size(); i++) {
                add_partial_ps(polys1[i], polys2[i], pcount, degree1, degree2, moduli, result[i], degree_result);
            }
        } else {
            ConstSliceArray<uint64_t> polys1_arr = construct_batch(polys1, pool, moduli);
            ConstSliceArray<uint64_t> polys2_arr = construct_batch(polys2, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree_result, KERNEL_THREAD_COUNT);
            set_device(moduli.device_index());
            kernel_add_partial_bps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1_arr, polys2_arr, pcount, degree1, degree2, moduli, result_arr, degree_result);
            utils::stream_sync();
        }
    }

    static void host_sub_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree_result; k++) {
                    size_t idx1 = i * moduli.size() * degree1 + j * degree1 + k;
                    size_t idx2 = i * moduli.size() * degree2 + j * degree2 + k;
                    size_t idx = i * moduli.size() * degree_result + j * degree_result + k;
                    uint64_t c1 = k < degree1 ? polys1[idx1] : 0;
                    uint64_t c2 = k < degree2 ? polys2[idx2] : 0;
                    result[idx] = c1 >= c2 ? c1 - c2 : moduli[j].value() - c2 + c1;
                }
            }
        }
    }

    static __device__ void device_sub_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree_result) {
            size_t i = idx / (moduli.size() * degree_result);
            size_t j = (idx / degree_result) % moduli.size();
            size_t k = idx % degree_result;
            size_t idx1 = i * moduli.size() * degree1 + j * degree1 + k;
            size_t idx2 = i * moduli.size() * degree2 + j * degree2 + k;
            uint64_t c1 = k < degree1 ? polys1[idx1] : 0;
            uint64_t c2 = k < degree2 ? polys2[idx2] : 0;
            result[idx] = c1 >= c2 ? c1 - c2 : moduli[j].value() - c2 + c1;
        }
    }
    static __global__ void kernel_sub_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        device_sub_partial_ps(polys1, polys2, pcount, degree1, degree2, moduli, result, degree_result);
    }
    static __global__ void kernel_sub_partial_bps(ConstSliceArrayRef<uint64_t> polys1, ConstSliceArrayRef<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result, size_t degree_result) {
        for (size_t i = 0; i < polys1.size(); i++) {
            device_sub_partial_ps(polys1[i], polys2[i], pcount, degree1, degree2, moduli, result[i], degree_result);
        }
    }

    void sub_partial_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, Slice<uint64_t> result, size_t degree_result) {
        if (!device_compatible(polys1, polys2, moduli, result)) {
            throw std::runtime_error("[sub_partial_ps] All inputs must be on the same device");
        }
        if (degree_result < degree1 || degree_result < degree2) {
            throw std::runtime_error("[sub_partial_ps] degree_result must be at least degree1 and degree2");
        }
        if (result.on_device()) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree_result, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_sub_partial_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree1, degree2, moduli, result, degree_result);
            utils::stream_sync();
        } else {
            host_sub_partial_ps(polys1, polys2, pcount, degree1, degree2, moduli, result, degree_result);
        }
    }
    void sub_partial_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree1, size_t degree2, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, size_t degree_result, MemoryPoolHandle pool) {
        bool device = moduli.on_device();
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) {
            throw std::runtime_error("[sub_partial_bps] polys1, polys2, and result must have the same size");
        }
        if (!device || polys1.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys1.size(); i++) {
                sub_partial_ps(polys1[i], polys2[i], pcount, degree1, degree2, moduli, result[i], degree_result);
            }
        } else {
            ConstSliceArray<uint64_t> polys1_arr = construct_batch(polys1, pool, moduli);
            ConstSliceArray<uint64_t> polys2_arr = construct_batch(polys2, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree_result, KERNEL_THREAD_COUNT);
            set_device(moduli.device_index());
            kernel_sub_partial_bps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1_arr, polys2_arr, pcount, degree1, degree2, moduli, result_arr, degree_result);
            utils::stream_sync();
        }
    }

    static void host_scatter_partial_ps(ConstSlice<uint64_t> source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli_size; j++) {
                for (size_t k = 0; k < destination_degree; k++) {
                    size_t idx = i * moduli_size * destination_degree + j * destination_degree + k;
                    size_t source_idx = i * moduli_size * source_degree + j * source_degree + k;
                    destination[idx] = k < source_degree ? source_polys[source_idx] : 0;
                }
            }
        }
    }

    static __device__ void device_scatter_partial_ps(ConstSlice<uint64_t> source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli_size * destination_degree) {
            size_t i = idx / (moduli_size * destination_degree);
            size_t j = (idx / destination_degree) % moduli_size;
            size_t k = idx % destination_degree;
            size_t source_idx = i * moduli_size * source_degree + j * source_degree + k;
            destination[idx] = k < source_degree ? source_polys[source_idx] : 0;
        }
    }
    static __global__ void kernel_scatter_partial_ps(ConstSlice<uint64_t> source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination) {
        device_scatter_partial_ps(source_polys, pcount, source_degree, destination_degree, moduli_size, destination);
    }
    static __global__ void kernel_scatter_partial_bps(ConstSliceArrayRef<uint64_t> source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, SliceArrayRef<uint64_t> destination) {
        for (size_t i = 0; i < source_polys.size(); i++) {
            device_scatter_partial_ps(source_polys[i], pcount, source_degree, destination_degree, moduli_size, destination[i]);
        }
    }

    void scatter_partial_ps(ConstSlice<uint64_t> source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, Slice<uint64_t> destination) {
        if (!device_compatible(source_polys, destination)) {
            throw std::runtime_error("[scatter_partial_ps] All inputs must be on the same device");
        }
        if (destination_degree < source_degree) {
            throw std::runtime_error("[scatter_partial_ps] destination_degree must be at least source_degree");
        }
        if (destination.on_device()) {
            size_t block_count = ceil_div<size_t>(pcount * moduli_size * destination_degree, KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_scatter_partial_ps<<<block_count, KERNEL_THREAD_COUNT>>>(source_polys, pcount, source_degree, destination_degree, moduli_size, destination);
            utils::stream_sync();
        } else {
            host_scatter_partial_ps(source_polys, pcount, source_degree, destination_degree, moduli_size, destination);
        }
    }
    void scatter_partial_bps(const ConstSliceVec<uint64_t>& source_polys, size_t pcount, size_t source_degree, size_t destination_degree, size_t moduli_size, const SliceVec<uint64_t>& destination, MemoryPoolHandle pool) {
        if (source_polys.size() != destination.size()) {
            throw std::runtime_error("[scatter_partial_bps] source_polys and destination must have the same size");
        }
        if (source_polys.size() == 0) return;
        auto device_reference = source_polys[0];
        bool device = device_reference.on_device();
        if (!device || source_polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < source_polys.size(); i++) {
                scatter_partial_ps(source_polys[i], pcount, source_degree, destination_degree, moduli_size, destination[i]);
            }
        } else {
            ConstSliceArray<uint64_t> source_polys_arr = construct_batch(source_polys, pool, device_reference);
            SliceArray<uint64_t> destination_arr = construct_batch(destination, pool, device_reference);
            size_t block_count = ceil_div<size_t>(pcount * moduli_size * destination_degree, KERNEL_THREAD_COUNT);
            set_device(device_reference.device_index());
            kernel_scatter_partial_bps<<<block_count, KERNEL_THREAD_COUNT>>>(source_polys_arr, pcount, source_degree, destination_degree, moduli_size, destination_arr);
            utils::stream_sync();
        }
    }

    void host_add_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    size_t p = polys[idx] + scalar;
                    result[idx] = p >= moduli[j].value() ? p - moduli[j].value() : p;
                }
            }
        }
    }

    __global__ void kernel_add_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            size_t p = polys[idx] + scalar;
            result[idx] = p >= moduli[j].value() ? p - moduli[j].value() : p;
        }
    }

    void add_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, moduli, result)) {
            throw std::runtime_error("[add_scalar_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_add_scalar_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalar, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_add_scalar_ps(polys, scalar, pcount, degree, moduli, result);
        }
    }

    void host_sub_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    result[idx] = polys[idx] >= scalar ? polys[idx] - scalar : moduli[j].value() - scalar + polys[idx];
                }
            }
        }
    }

    __global__ void kernel_sub_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = polys[idx] >= scalar ? polys[idx] - scalar : moduli[j].value() - scalar + polys[idx];
        }
    }

    void sub_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, moduli, result)) {
            throw std::runtime_error("[sub_scalar_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_sub_scalar_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalar, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_sub_scalar_ps(polys, scalar, pcount, degree, moduli, result);
        }
    }

    void host_multiply_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    result[idx] = multiply_uint64_mod(polys[idx], scalar, moduli[j]);
                }
            }
        }
    }

    __device__ void device_multiply_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = multiply_uint64_mod(polys[idx], scalar, moduli[j]);
        }
    }
    __global__ void kernel_multiply_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_multiply_scalar_ps(polys, scalar, pcount, degree, moduli, result);
    }
    __global__ void kernel_multiply_scalar_bps(ConstSliceArrayRef<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        for (size_t i = 0; i < polys.size(); i++) {
            device_multiply_scalar_ps(polys[i], scalar, pcount, degree, moduli, result[i]);
        }
    }

    void multiply_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, moduli, result)) {
            throw std::runtime_error("[multiply_scalar_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_multiply_scalar_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalar, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_multiply_scalar_ps(polys, scalar, pcount, degree, moduli, result);
        }
    }
    void multiply_scalar_bps(const ConstSliceVec<uint64_t>& polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        bool device = moduli.on_device();
        if (polys.size() != result.size()) {
            throw std::runtime_error("[multiply_scalar_bps] polys and result must have the same size");
        }
        if (!device || polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys.size(); i++) {
                multiply_scalar_ps(polys[i], scalar, pcount, degree, moduli, result[i]);
            }
        } else {
            ConstSliceArray<uint64_t> polys_arr = construct_batch(polys, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            set_device(moduli.device_index());
            kernel_multiply_scalar_bps<<<block_count, KERNEL_THREAD_COUNT>>>(polys_arr, scalar, pcount, degree, moduli, result_arr);
            utils::stream_sync();
        }
    }




    void host_multiply_scalars_ps(ConstSlice<uint64_t> polys, ConstSlice<uint64_t> scalars, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    result[idx] = multiply_uint64_mod(polys[idx], scalars[j], moduli[j]);
                }
            }
        }
    }

    __global__ void kernel_multiply_scalars_ps(ConstSlice<uint64_t> polys, ConstSlice<uint64_t> scalars, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = multiply_uint64_mod(polys[idx], scalars[j], moduli[j]);
        }
    }

    void multiply_scalars_ps(ConstSlice<uint64_t> polys, ConstSlice<uint64_t> scalars, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, scalars, moduli, result)) {
            throw std::runtime_error("[multiply_scalars_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_multiply_scalars_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalars, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_multiply_scalars_ps(polys, scalars, pcount, degree, moduli, result);
        }
    }



    void host_multiply_uint64operand_ps(ConstSlice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    result[idx] = multiply_uint64operand_mod(polys[idx], operand[j], moduli[j]);
                }
            }
        }
    }

    __device__ void device_multiply_uint64operand_ps(ConstSlice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = multiply_uint64operand_mod(polys[idx], operand[j], moduli[j]);
        }
    }

    __global__ void kernel_multiply_uint64operand_ps(ConstSlice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_multiply_uint64operand_ps(polys, operand, pcount, degree, moduli, result);
    }

    __global__ void kernel_multiply_uint64operand_bps(ConstSliceArrayRef<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        size_t i = blockIdx.y;
        device_multiply_uint64operand_ps(polys[i], operand, pcount, degree, moduli, result[i]);
    }

    void multiply_uint64operand_ps(ConstSlice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, operand, moduli, result)) {
            throw std::runtime_error("[multiply_uint64operand_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_multiply_uint64operand_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, operand, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_multiply_uint64operand_ps(polys, operand, pcount, degree, moduli, result);
        }
    }
    

    void multiply_uint64operand_bps(const ConstSliceVec<uint64_t>& polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        if (polys.size() != result.size()) {
            throw std::runtime_error("[multiply_uint64operand_bps] polys and result must have the same size");
        }
        if (polys.size() == 0) return;
        if (!moduli.on_device() || polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys.size(); i++) {
                multiply_uint64operand_ps(polys[i], operand, pcount, degree, moduli, result[i]);
            }
        } else {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            dim3 block_dims(block_count, polys.size());
            auto polys_batched = construct_batch(polys, pool, moduli);
            auto result_batched = construct_batch(result, pool, moduli);
            set_device(moduli.device_index());
            kernel_multiply_uint64operand_bps<<<dim3(block_count, polys.size()), KERNEL_THREAD_COUNT>>>(polys_batched, operand, pcount, degree, moduli, result_batched);
            utils::stream_sync();
        }
    }

    void host_dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                uint64_t modulus_value = moduli[j].value();
                uint64_t cr0 = moduli[j].const_ratio()[0];
                uint64_t cr1 = moduli[j].const_ratio()[1];
                uint64_t z[2]{0, 0}; Slice<uint64_t> z_slice(z, 2, false, nullptr);
                uint64_t tmp1 = 0;
                uint64_t tmp2[2]{0, 0}; Slice<uint64_t> tmp2_slice(tmp2, 2, false, nullptr);
                uint64_t tmp3;
                uint64_t carry = 0;
                for (size_t k = 0; k < degree; k++) {
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    // Reduces z using base 2^64 Barrett reduction
                    multiply_uint64_uint64(polys1[idx], polys2[idx], z_slice);
                    // Multiply input and const_ratio
                    // Round 1
                    multiply_uint64_high_word(z[0], cr0, carry);
                    multiply_uint64_uint64(z[0], cr1, tmp2_slice);
                    tmp3 = tmp2[1] + static_cast<uint64_t>(add_uint64(tmp2[0], carry, tmp1));
                    // Round 2
                    multiply_uint64_uint64(z[1], cr0, tmp2_slice);
                    carry = tmp2[1] + static_cast<uint64_t>(add_uint64(tmp1, tmp2[0], tmp1));
                    // This is all we care about
                    tmp1 = z[1] * cr1 + tmp3 + carry;
                    // Barrett subtraction
                    tmp3 = z[0] - tmp1 * modulus_value;
                    // Claim: One more subtraction is enough
                    result[idx] = (tmp3 >= modulus_value) ? (tmp3 - modulus_value) : tmp3;
                }
            }
        }
    }

    __device__ void device_dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = multiply_uint64_mod(polys1[idx], polys2[idx], moduli[j]);
        }
    }
    __global__ void kernel_dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_dyadic_product_ps(polys1, polys2, pcount, degree, moduli, result);
    }
    __global__ void kernel_dyadic_product_bps(ConstSliceArrayRef<uint64_t> polys1, ConstSliceArrayRef<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        size_t i = blockIdx.y;
        device_dyadic_product_ps(polys1[i], polys2[i], pcount, degree, moduli, result[i]);
    }

    void dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys1, polys2, moduli, result)) {
            throw std::runtime_error("[dyadic_product_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_dyadic_product_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_dyadic_product_ps(polys1, polys2, pcount, degree, moduli, result);
        }
    }

    void dyadic_product_bps(const ConstSliceVec<uint64_t>& polys1, const ConstSliceVec<uint64_t>& polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        if (polys1.size() != polys2.size() || polys1.size() != result.size()) {
            throw std::runtime_error("[dyadic_product_bps] polys1, polys2, and result must have the same size");
        }
        if (polys1.size() == 0) return;
        bool device = moduli.on_device();
        if (!device || polys1.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys1.size(); i++) {
                dyadic_product_ps(polys1[i], polys2[i], pcount, degree, moduli, result[i]);
            }
        } else {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            auto polys1_batched = construct_batch(polys1, pool, moduli);
            auto polys2_batched = construct_batch(polys2, pool, moduli);
            auto result_batched = construct_batch(result, pool, moduli);
            utils::set_device(moduli.device_index());
            dim3 block_dims(block_count, polys1.size());
            kernel_dyadic_product_bps<<<block_dims, KERNEL_THREAD_COUNT>>>(polys1_batched, polys2_batched, pcount, degree, moduli, result_batched);
            utils::stream_sync();
        }
    }

    void host_negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        if (shift == 0) {
            set_uint(polys, pcount * moduli.size() * degree, result);
            return;
        }
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                size_t index_raw = shift;
                size_t mask = degree - 1;
                uint64_t modulus_value = moduli[j].value();
                for (size_t k = 0; k < degree; k++) {
                    size_t index = index_raw & mask;
                    size_t idx = i * moduli.size() * degree + j * degree + k;
                    size_t result_index = i * moduli.size() * degree + j * degree + index;
                    if (polys[idx] == 0 || (index_raw & degree) == 0) {
                        result[result_index] = polys[idx];
                    } else {
                        result[result_index] = modulus_value - polys[idx];
                    }
                    index_raw += 1;
                }
            }
        }
    }

    __device__ void device_negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t i = idx / (moduli.size() * degree);
            size_t j = (idx / degree) % moduli.size();
            size_t k = idx % degree;
            size_t mask = degree - 1;
            uint64_t modulus_value = moduli[j].value();

            size_t index = (shift + k) & mask;
            size_t result_index = i * moduli.size() * degree + j * degree + index;
            if (polys[idx] == 0 || ((shift + k) & degree) == 0) {
                result[result_index] = polys[idx];
            } else {
                result[result_index] = modulus_value - polys[idx];
            }
        }
    }

    __global__ void kernel_negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_negacyclic_shift_ps(polys, shift, pcount, degree, moduli, result);
    }
    __global__ void kernel_negacyclic_shift_bps(ConstSliceArrayRef<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        size_t i = blockIdx.y;
        device_negacyclic_shift_ps(polys[i], shift, pcount, degree, moduli, result[i]);
    }
    

    void negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!device_compatible(polys, moduli, result)) {
            throw std::runtime_error("[negacyclic_shift_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_negacyclic_shift_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, shift, pcount, degree, moduli, result);
            utils::stream_sync();
        } else {
            host_negacyclic_shift_ps(polys, shift, pcount, degree, moduli, result);
        }
    }
    void negacyclic_shift_bps(const ConstSliceVec<uint64_t>& polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) {
        if (polys.size() != result.size()) {
            throw std::runtime_error("[negacyclic_shift_bps] polys and result must have the same size");
        }
        if (polys.size() == 0) return;
        bool device = moduli.on_device();
        if (!device || polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys.size(); i++) {
                negacyclic_shift_ps(polys[i], shift, pcount, degree, moduli, result[i]);
            }
        } else {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            ConstSliceArray<uint64_t> polys_arr = construct_batch(polys, pool, moduli);
            SliceArray<uint64_t> result_arr = construct_batch(result, pool, moduli);
            set_device(moduli.device_index());
            dim3 block_dims(block_count, polys.size());
            kernel_negacyclic_shift_bps<<<block_dims, KERNEL_THREAD_COUNT>>>(polys_arr, shift, pcount, degree, moduli, result_arr);
            utils::stream_sync();
        }
    }

}}