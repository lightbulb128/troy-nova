#include "lwe_ciphertext.h"
#include "utils/dynamic_array.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;

    __global__ static void kernel_assemble_lwe_set(size_t poly_modulus_degree, Slice<uint64_t> rlwec0, Slice<uint64_t> rlwec1, ConstSlice<uint64_t> c0, ConstSlice<uint64_t> c1) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < rlwec1.size()) {
            rlwec1[i] = c1[i];
            if (i % poly_modulus_degree == 0) {
                rlwec0[i] = c0[i / poly_modulus_degree];
            } else {
                rlwec0[i] = 0;
            }
        }
    }

    static void assemble_lwe_set(size_t poly_modulus_degree, Slice<uint64_t> rlwec0, Slice<uint64_t> rlwec1, ConstSlice<uint64_t> c0, ConstSlice<uint64_t> c1) {
        bool device = rlwec0.on_device();
        if (!device) {
            rlwec1.copy_from_slice(c1);
            for (size_t i = 0; i < c0.size(); i++) {
                rlwec0[i * poly_modulus_degree] = c0[i];
                for (size_t j = 1; j < poly_modulus_degree; j++) {
                    rlwec0[i * poly_modulus_degree + j] = 0;
                }
            }
        } else {
            size_t block_size = utils::ceil_div(rlwec1.size(), utils::KERNEL_THREAD_COUNT);
            utils::set_device(rlwec1.device_index());
            kernel_assemble_lwe_set<<<block_size, utils::KERNEL_THREAD_COUNT>>>(poly_modulus_degree, rlwec0, rlwec1, c0, c1);
            utils::stream_sync();
        }
    }

    Ciphertext LWECiphertext::assemble_lwe(MemoryPoolHandle pool) const {
        size_t poly_len = this->coeff_modulus_size() * this->poly_modulus_degree();
        utils::DynamicArray<uint64_t> data = utils::DynamicArray<uint64_t>::create_uninitialized(poly_len * 2, this->on_device(), pool);
        assemble_lwe_set(this->poly_modulus_degree(), data.slice(0, poly_len), data.slice(poly_len, 2 * poly_len), this->const_c0(), this->const_c1());
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