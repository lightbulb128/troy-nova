#include "poly_small_mod.cuh"

namespace troy {namespace utils {

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

    __global__ void kernel_modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            // size_t i = idx / (moduli.size() * degree);
            size_t j = (idx / degree) % moduli.size();
            // size_t k = idx % degree;
            result[idx] = moduli[j].reduce(polys[idx]);
        }
    }

    void modulo_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys.on_device(), moduli.on_device())) {
            throw std::runtime_error("[modulo_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_modulo_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, pcount, degree, moduli, result);
        } else {
            host_modulo_ps(polys, pcount, degree, moduli, result);
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

    __global__ void kernel_negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        result[idx] = polys[idx] == 0 ? 0 : moduli[(idx / degree) % moduli.size()].value() - polys[idx];
    }

    void negate_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys.on_device(), moduli.on_device())) {
            throw std::runtime_error("[negate_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t total = pcount * moduli.size() * degree;
            size_t thread_count = min(total, KERNEL_THREAD_COUNT);
            size_t block_count = ceil_div<size_t>(total, thread_count);
            kernel_negate_ps<<<block_count, thread_count>>>(polys, pcount, degree, moduli, result);
        } else {
            host_negate_ps(polys, pcount, degree, moduli, result);
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

    __global__ void kernel_add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            size_t p = polys1[idx] + polys2[idx];
            result[idx] = p >= moduli[j].value() ? p - moduli[j].value() : p;
        }
    }

    void add_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys1.on_device(), polys2.on_device(), moduli.on_device())) {
            throw std::runtime_error("[add_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_add_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree, moduli, result);
        } else {
            host_add_ps(polys1, polys2, pcount, degree, moduli, result);
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

    __global__ void kernel_sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = polys1[idx] >= polys2[idx] ? polys1[idx] - polys2[idx] : moduli[j].value() - polys2[idx] + polys1[idx];
        }
    }

    void sub_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys1.on_device(), polys2.on_device(), moduli.on_device())) {
            throw std::runtime_error("[sub_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_sub_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree, moduli, result);
        } else {
            host_sub_ps(polys1, polys2, pcount, degree, moduli, result);
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
        if (!same(device, polys.on_device(), moduli.on_device())) {
            throw std::runtime_error("[add_scalar_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_add_scalar_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalar, pcount, degree, moduli, result);
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
        if (!same(device, polys.on_device(), moduli.on_device())) {
            throw std::runtime_error("[sub_scalar_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_sub_scalar_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalar, pcount, degree, moduli, result);
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

    __global__ void kernel_multiply_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = multiply_uint64_mod(polys[idx], scalar, moduli[j]);
        }
    }

    void multiply_scalar_ps(ConstSlice<uint64_t> polys, uint64_t scalar, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys.on_device(), moduli.on_device())) {
            throw std::runtime_error("[multiply_scalar_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_multiply_scalar_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalar, pcount, degree, moduli, result);
        } else {
            host_multiply_scalar_ps(polys, scalar, pcount, degree, moduli, result);
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
        if (!same(device, polys.on_device(), moduli.on_device())) {
            throw std::runtime_error("[multiply_scalars_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_multiply_scalars_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, scalars, pcount, degree, moduli, result);
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

    __global__ void kernel_multiply_uint64operand_ps(ConstSlice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = multiply_uint64operand_mod(polys[idx], operand[j], moduli[j]);
        }
    }

    void multiply_uint64operand_ps(ConstSlice<uint64_t> polys, ConstSlice<MultiplyUint64Operand> operand, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys.on_device(), operand.on_device(), moduli.on_device())) {
            throw std::runtime_error("[multiply_uint64operand_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_multiply_uint64operand_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, operand, pcount, degree, moduli, result);
        } else {
            host_multiply_uint64operand_ps(polys, operand, pcount, degree, moduli, result);
        }
    }

    void host_dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        for (size_t i = 0; i < pcount; i++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                uint64_t modulus_value = moduli[j].value();
                uint64_t cr0 = moduli[j].const_ratio()[0];
                uint64_t cr1 = moduli[j].const_ratio()[1];
                uint64_t z[2]{0, 0}; Slice<uint64_t> z_slice(z, 2, false);
                uint64_t tmp1 = 0;
                uint64_t tmp2[2]{0, 0}; Slice<uint64_t> tmp2_slice(tmp2, 2, false);
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

    __global__ void kernel_dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < pcount * moduli.size() * degree) {
            size_t j = (idx / degree) % moduli.size();
            result[idx] = multiply_uint64_mod(polys1[idx], polys2[idx], moduli[j]);
        }
    }

    void dyadic_product_ps(ConstSlice<uint64_t> polys1, ConstSlice<uint64_t> polys2, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys1.on_device(), polys2.on_device(), moduli.on_device())) {
            throw std::runtime_error("[dyadic_product_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_dyadic_product_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys1, polys2, pcount, degree, moduli, result);
        } else {
            host_dyadic_product_ps(polys1, polys2, pcount, degree, moduli, result);
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

    __global__ void kernel_negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
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

    void negacyclic_shift_ps(ConstSlice<uint64_t> polys, size_t shift, size_t pcount, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        bool device = result.on_device();
        if (!same(device, polys.on_device(), moduli.on_device())) {
            throw std::runtime_error("[negacyclic_shift_ps] All inputs must be on the same device");
        }
        if (device) {
            size_t block_count = ceil_div<size_t>(pcount * moduli.size() * degree, KERNEL_THREAD_COUNT);
            kernel_negacyclic_shift_ps<<<block_count, KERNEL_THREAD_COUNT>>>(polys, shift, pcount, degree, moduli, result);
        } else {
            host_negacyclic_shift_ps(polys, shift, pcount, degree, moduli, result);
        }
    }




}}