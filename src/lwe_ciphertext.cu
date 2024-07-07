#include "lwe_ciphertext.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;

    __global__ static void kernel_assemble_lwe_scatter_c0(size_t poly_modulus_degree, Slice<uint64_t> rlwec0, ConstSlice<uint64_t> c0) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < c0.size()) {
            rlwec0[i * poly_modulus_degree] = c0[i];
        }
    }

    static void assemble_lwe_scatter_c0(size_t poly_modulus_degree, Slice<uint64_t> rlwec0, ConstSlice<uint64_t> c0) {
        bool device = rlwec0.on_device();
        if (!device) {
            for (size_t i = 0; i < c0.size(); i++) {
                rlwec0[i * poly_modulus_degree] = c0[i];
            }
        } else {
            if (c0.size() >= utils::KERNEL_THREAD_COUNT) {
                size_t block_size = utils::ceil_div(c0.size(), utils::KERNEL_THREAD_COUNT);
                cudaSetDevice(c0.device_index());
                kernel_assemble_lwe_scatter_c0<<<block_size, utils::KERNEL_THREAD_COUNT>>>(poly_modulus_degree, rlwec0, c0);
            } else {
                cudaSetDevice(c0.device_index());
                kernel_assemble_lwe_scatter_c0<<<1, c0.size()>>>(poly_modulus_degree, rlwec0, c0);
            }
            cudaStreamSynchronize(0);
        }
    }

    Ciphertext LWECiphertext::assemble_lwe(MemoryPoolHandle pool) const {
        size_t poly_len = this->coeff_modulus_size() * this->poly_modulus_degree();
        utils::DynamicArray<uint64_t> data(poly_len * 2, this->on_device(), pool);
        data.slice(poly_len, 2 * poly_len).copy_from_slice(this->const_c1());
        assemble_lwe_scatter_c0(this->poly_modulus_degree(), data.slice(0, poly_len), this->const_c0());
        return Ciphertext::from_members(
            2,
            this->coeff_modulus_size(),
            this->poly_modulus_degree(),
            this->parms_id(),
            this->scale(),
            false,
            this->correction_factor(),
            0,
            std::move(data)
        );
    }

}