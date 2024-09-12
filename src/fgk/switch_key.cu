#include "switch_key.h"
#include "../batch_utils.h"

namespace troy::utils::fgk::switch_key {

    __device__ static void device_set_accumulate(
        size_t decomp_modulus_size, size_t coeff_count, ConstSlice<uint64_t> target_intt, Slice<uint64_t> temp_ntt, ConstSlice<Modulus> key_modulus
    ) {
        size_t key_modulus_size = key_modulus.size();
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_index >= decomp_modulus_size * coeff_count * (decomp_modulus_size + 1)) return;

        size_t k = global_index % coeff_count;
        size_t j = (global_index / coeff_count) % decomp_modulus_size;
        size_t i = global_index / coeff_count / decomp_modulus_size;
        
        size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);
        size_t temp_ntt_index = (i * decomp_modulus_size + j) * coeff_count + k;
        size_t target_intt_index = j * coeff_count + k;

        if (key_modulus[j].value() <= key_modulus[key_index].value()) {
            temp_ntt[temp_ntt_index] = target_intt[target_intt_index];
        } else {
            temp_ntt[temp_ntt_index] = key_modulus[key_index].reduce(target_intt[target_intt_index]);
        }
    }

    __global__ static void kernel_set_accumulate(
        size_t decomp_modulus_size, size_t coeff_count, ConstSlice<uint64_t> target_intt, Slice<uint64_t> temp_ntt, ConstSlice<Modulus> key_modulus
    ) {
        device_set_accumulate(decomp_modulus_size, coeff_count, target_intt, temp_ntt, key_modulus);
    }

    __global__ static void kernel_set_accumulate_batched(
        size_t decomp_modulus_size, size_t coeff_count, ConstSliceArrayRef<uint64_t> target_intt, SliceArrayRef<uint64_t> temp_ntt, ConstSlice<Modulus> key_modulus
    ) {
        size_t i = blockIdx.y;
        device_set_accumulate(decomp_modulus_size, coeff_count, target_intt[i], temp_ntt[i], key_modulus);
    }

    void set_accumulate(
        size_t decomp_modulus_size, size_t coeff_count, ConstSlice<uint64_t> target_intt, Buffer<uint64_t>& temp_ntt, ConstSlice<Modulus> key_modulus
    ) {
        assert(temp_ntt.on_device());
        size_t rns_modulus_size = decomp_modulus_size + 1;

        size_t total = rns_modulus_size * decomp_modulus_size * coeff_count;
        size_t block_count = ceil_div(total, KERNEL_THREAD_COUNT);
        set_device(target_intt.device_index());
        kernel_set_accumulate<<<block_count, KERNEL_THREAD_COUNT>>>(decomp_modulus_size, coeff_count, target_intt, temp_ntt.reference(), key_modulus);
        stream_sync();
    }

    void set_accumulate_batched(
        size_t decomp_modulus_size, size_t coeff_count, const utils::ConstSliceVec<uint64_t>& target_intt, std::vector<Buffer<uint64_t>>& temp_ntt, ConstSlice<Modulus> key_modulus, MemoryPoolHandle pool
    ) {
        assert(key_modulus.on_device());
        if (target_intt.size() != temp_ntt.size()) {
            throw std::invalid_argument("[fgk::set_accumulate_batched] target_intt and temp_ntt must have the same size");
        }
        
        size_t rns_modulus_size = decomp_modulus_size + 1;

        size_t total = rns_modulus_size * decomp_modulus_size * coeff_count;
        size_t block_count = ceil_div(total, KERNEL_THREAD_COUNT);
        set_device(key_modulus.device_index());
        auto target_intt_batched = batch_utils::construct_batch(target_intt, pool, key_modulus);
        auto temp_ntt_slices = batch_utils::rcollect_reference(temp_ntt);
        auto temp_ntt_batched = batch_utils::construct_batch(temp_ntt_slices, pool, key_modulus);
        dim3 block_dims(block_count, target_intt.size());
        kernel_set_accumulate_batched<<<block_dims, KERNEL_THREAD_COUNT>>>(decomp_modulus_size, coeff_count, target_intt_batched, temp_ntt_batched, key_modulus);
        stream_sync();
    }
    
    
    __device__ static void device_accumulate_products(
        size_t decomp_modulus_size, size_t key_component_count, size_t coeff_count, 
        ConstSlice<uint64_t> temp_ntt,
        ConstSlice<Modulus> key_moduli,
        ConstSlice<const uint64_t*> key_vector,
        Slice<uint64_t> poly_prod
    ) {
        

        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * key_component_count) return;
        size_t l = global_index % coeff_count;
        size_t k = global_index / coeff_count;

        size_t key_modulus_size = key_moduli.size();
        size_t key_poly_coeff_size = key_modulus_size * coeff_count;
        size_t rns_modulus_size = decomp_modulus_size + 1;

        for (size_t i = 0; i < rns_modulus_size; i++) {
            size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);
            const Modulus& key_modulus = key_moduli[key_index];

            uint64_t accumulator_l[2] {0, 0}; Slice<uint64_t> accumulator_l_slice(accumulator_l, 2, true, nullptr);
            
            for (size_t j = 0; j < decomp_modulus_size; j++) {
                uint64_t temp_operand = temp_ntt[(i * decomp_modulus_size + j) * coeff_count + l];

                uint64_t qword[2] {0, 0}; Slice<uint64_t> qword_slice(qword, 2, true, nullptr);
                utils::multiply_uint64_uint64(temp_operand, key_vector[j][k * key_poly_coeff_size + key_index * coeff_count + l], qword_slice);
                utils::add_uint128_inplace(qword_slice, accumulator_l_slice.as_const());
                accumulator_l[0] = key_modulus.reduce_uint128_limbs(qword_slice.as_const());
                accumulator_l[1] = 0; 

            }
            
            poly_prod[(k * rns_modulus_size + i) * coeff_count + l] = accumulator_l[0];

        }
    }

    __global__ static void kernel_accumulate_products(
        size_t decomp_modulus_size, size_t key_component_count, size_t coeff_count, 
        ConstSlice<uint64_t> temp_ntt,
        ConstSlice<Modulus> key_moduli,
        ConstSlice<const uint64_t*> key_vector,
        Slice<uint64_t> poly_prod
    ) {
        device_accumulate_products(decomp_modulus_size, key_component_count, coeff_count, temp_ntt, key_moduli, key_vector, poly_prod);
    }

    __global__ static void kernel_accumulate_products_batched(
        size_t decomp_modulus_size, size_t key_component_count, size_t coeff_count, 
        ConstSliceArrayRef<uint64_t> temp_ntt,
        ConstSlice<Modulus> key_moduli,
        ConstSlice<const uint64_t*> key_vector,
        SliceArrayRef<uint64_t> poly_prod
    ) {
        size_t i = blockIdx.y;
        device_accumulate_products(decomp_modulus_size, key_component_count, coeff_count, temp_ntt[i], key_moduli, key_vector, poly_prod[i]);
    }


    void accumulate_products(
        size_t decomp_modulus_size, size_t key_component_count, size_t coeff_count, 
        ConstSlice<uint64_t> temp_ntt, 
        ConstSlice<Modulus> key_moduli,
        ConstSlice<const uint64_t*> key_vector,
        Slice<uint64_t> poly_prod
    ) {
        assert(temp_ntt.on_device());
        assert(poly_prod.on_device());
        size_t total = coeff_count * key_component_count;
        size_t block_count = ceil_div(total, KERNEL_THREAD_COUNT);

        set_device(temp_ntt.device_index());
        kernel_accumulate_products<<<block_count, KERNEL_THREAD_COUNT>>>(decomp_modulus_size, key_component_count, coeff_count, temp_ntt, key_moduli, key_vector, poly_prod);
        stream_sync();
    }

    void accumulate_products_batched(
        size_t decomp_modulus_size, size_t key_component_count, size_t coeff_count, 
        const ConstSliceVec<uint64_t>& temp_ntt, 
        ConstSlice<Modulus> key_moduli,
        ConstSlice<const uint64_t*> key_vector,
        const SliceVec<uint64_t>& poly_prod,
        MemoryPoolHandle pool
    ) {
        if (temp_ntt.size() != poly_prod.size()) {
            throw std::invalid_argument("[fgk::accumulate_products_batched] temp_ntt and poly_prod must have the same size");
        }
        assert(key_moduli.on_device());
        size_t total = coeff_count * key_component_count;
        size_t block_count = ceil_div(total, KERNEL_THREAD_COUNT);

        set_device(key_moduli.device_index());
        auto temp_ntt_batched = batch_utils::construct_batch(temp_ntt, pool, key_moduli);
        auto poly_prod_batched = batch_utils::construct_batch(poly_prod, pool, key_moduli);
        dim3 block_dims(block_count, temp_ntt.size());
        kernel_accumulate_products_batched<<<block_dims, KERNEL_THREAD_COUNT>>>(decomp_modulus_size, key_component_count, coeff_count, temp_ntt_batched, key_moduli, key_vector, poly_prod_batched);
        stream_sync();
    }


}