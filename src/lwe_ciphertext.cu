#include "lwe_ciphertext.h"
#include "utils/constants.h"
#include "utils/dynamic_array.h"
#include "batch_utils.h"
#include "utils/memory_pool.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;

    __device__ static void device_assemble_lwe_set(size_t poly_modulus_degree, Slice<uint64_t> rlwec0, Slice<uint64_t> rlwec1, ConstSlice<uint64_t> c0, ConstSlice<uint64_t> c1) {
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

    __global__ static void kernel_assemble_lwe_set(size_t poly_modulus_degree, Slice<uint64_t> rlwec0, Slice<uint64_t> rlwec1, ConstSlice<uint64_t> c0, ConstSlice<uint64_t> c1) {
        device_assemble_lwe_set(poly_modulus_degree, rlwec0, rlwec1, c0, c1);
    }

    __global__ static void kernel_assemble_lwe_set_batched(size_t poly_modulus_degree, utils::SliceArrayRef<uint64_t> rlwec0, utils::SliceArrayRef<uint64_t> rlwec1, utils::ConstSliceArrayRef<uint64_t> c0, utils::ConstSliceArrayRef<uint64_t> c1) {
        for (size_t i = 0; i < rlwec0.size(); i++) {
            device_assemble_lwe_set(poly_modulus_degree, rlwec0[i], rlwec1[i], c0[i], c1[i]);
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

    static void assemble_lwe_set_batched(size_t poly_modulus_degree, const utils::SliceVec<uint64_t>& rlwec0, const utils::SliceVec<uint64_t>& rlwec1, const utils::ConstSliceVec<uint64_t>& c0, const utils::ConstSliceVec<uint64_t>& c1, MemoryPoolHandle pool) {
        if (rlwec0.size() != rlwec1.size() || rlwec0.size() != c0.size() || rlwec0.size() != c1.size()) {
            throw std::invalid_argument("[assemble_lwe_set_batched]: invalid input sizes");
        }
        bool device = rlwec0[0].on_device();
        if (!device || rlwec0.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < rlwec0.size(); i++) {
                assemble_lwe_set(poly_modulus_degree, rlwec0[i], rlwec1[i], c0[i], c1[i]);
            }
        } else {
            size_t block_size = utils::ceil_div(rlwec1[0].size(), utils::KERNEL_THREAD_COUNT);
            auto reference = rlwec1[0];
            utils::set_device(rlwec1[0].device_index());
            auto rlwec0_arr = batch_utils::construct_batch(rlwec0, pool, reference);
            auto rlwec1_arr = batch_utils::construct_batch(rlwec1, pool, reference);
            auto c0_arr = batch_utils::construct_batch(c0, pool, reference);
            auto c1_arr = batch_utils::construct_batch(c1, pool, reference);
            kernel_assemble_lwe_set_batched<<<block_size, utils::KERNEL_THREAD_COUNT>>>(poly_modulus_degree, rlwec0_arr, rlwec1_arr, c0_arr, c1_arr);
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

    void LWECiphertext::assemble_lwe_batched(const std::vector<const LWECiphertext*>& source, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) {
        if (source.size() != destination.size()) {
            throw std::invalid_argument("[assemble_lwe_batched]: invalid input sizes");
        }
        if (source.size() == 0) {
            return;
        }
        ParmsID parms_id = source[0]->parms_id();
        size_t coeff_modulus_size = source[0]->coeff_modulus_size();
        size_t poly_modulus_degree = source[0]->poly_modulus_degree();
        size_t poly_len = coeff_modulus_size * poly_modulus_degree;
        for (size_t i = 1; i < source.size(); i++) {
            if (source[i]->coeff_modulus_size() != coeff_modulus_size || source[i]->poly_modulus_degree() != poly_modulus_degree) {
                throw std::invalid_argument("[assemble_lwe_batched]: invalid input sizes, coeff_modulus_size and poly_modulus_degree must be the same");
            }
            if (source[i]->parms_id() != parms_id) {
                throw std::invalid_argument("[assemble_lwe_batched]: invalid input sizes, parms_id must be the same");
            }
        }
        std::vector<utils::DynamicArray<uint64_t>> data(source.size());
        for (size_t i = 0; i < source.size(); i++) {
            data[i] = utils::DynamicArray<uint64_t>::create_uninitialized(poly_len * 2, source[i]->on_device(), pool);
        }
        utils::SliceVec<uint64_t> rlwec0s; rlwec0s.reserve(destination.size());
        utils::SliceVec<uint64_t> rlwec1s; rlwec1s.reserve(destination.size());
        utils::ConstSliceVec<uint64_t> c0s; c0s.reserve(source.size());
        utils::ConstSliceVec<uint64_t> c1s; c1s.reserve(source.size());
        for (size_t i = 0; i < source.size(); i++) {
            rlwec0s.push_back(data[i].slice(0, poly_len));
            rlwec1s.push_back(data[i].slice(poly_len, 2 * poly_len));
            c0s.push_back(source[i]->const_c0());
            c1s.push_back(source[i]->const_c1());
        }
        assemble_lwe_set_batched(poly_modulus_degree, rlwec0s, rlwec1s, c0s, c1s, pool);
        for (size_t i = 0; i < source.size(); i++) {
            *destination[i] = Ciphertext::from_members(
                2,
                coeff_modulus_size,
                poly_modulus_degree,
                source[i]->parms_id(),
                source[i]->scale(),
                false,
                source[i]->correction_factor(),
                0,
                std::move(data[i])
            );
        }
        
    }

}