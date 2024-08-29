#include "ntt_cooperative.h"

#include "ntt_grouped.h"
#include <cooperative_groups.h>
#include <cassert>
#include <mutex>
#include <unordered_map>

namespace troy::utils::fgk::ntt_cooperative {

    static constexpr size_t NTT_KERNEL_THREAD_COUNT = 256;
    static constexpr size_t NTT_KERNEL_THREAD_COUNT_LOG2 = 8;

    static std::map<int, cudaDeviceProp> device_properties;
    static bool device_properties_initialized = false;
    static std::mutex device_properties_mutex;

    static void ensure_device_properties() {
        if (device_properties_initialized) return;
        std::unique_lock<std::mutex> lock(device_properties_mutex);
        if (device_properties_initialized) return;
        int device_count = 0;
        cudaError_t success = cudaGetDeviceCount(&device_count);
        if (success != cudaSuccess) {
            throw std::runtime_error("[ntt_cooperative::ensure_device_properties] Unable to get device count.");
        }
        device_properties.clear();
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            success = cudaGetDeviceProperties(&prop, i);
            if (success != cudaSuccess) {
                throw std::runtime_error("[ntt_cooperative::ensure_device_properties] Unable to get device properties.");
            }
            device_properties[i] = prop;
        }
        
        device_properties_initialized = true;
    }

    __device__ void device_ntt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {

        unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int coeff_modulus_size = component_count;

        unsigned int k = global_index / (coeff_modulus_size << (log_degree - 1));
        unsigned int j = (global_index >> (log_degree - 1)) % coeff_modulus_size;
        
        const NTTTables& table = tables.get(k, j);
        const Modulus& modulus = table.modulus();
        uint64_t two_times_modulus = modulus.value() << 1;
        const MultiplyUint64Operand* r_ptr = use_inv_root_powers ?
            table.inv_root_powers().raw_pointer() :
            table.root_powers().raw_pointer();

        unsigned int block_idx = blockIdx.x % (gridDim.x / (pcount * coeff_modulus_size));

        __shared__ uint64_t sdata[NTT_KERNEL_THREAD_COUNT * 2];

        cooperative_groups::grid_group group = cooperative_groups::this_grid();

        for (unsigned int layer_lower = 0; layer_lower < log_degree; layer_lower += NTT_KERNEL_THREAD_COUNT_LOG2) {

            unsigned int layer_upper = min(layer_lower + NTT_KERNEL_THREAD_COUNT_LOG2, log_degree);

            unsigned int gap_power = log_degree - layer_lower - 1;
            unsigned int E_power = min(static_cast<unsigned int>(get_power_of_two(blockDim.x)), gap_power); // elements in gap
            unsigned int E_mask = (1 << E_power) - 1;
            unsigned int stride_power = gap_power - E_power;
            unsigned int stride_mask = (1 << stride_power) - 1;

            unsigned int coefficient_offset = (block_idx & stride_mask) + (((block_idx >> stride_power) * (blockDim.x >> E_power)) << (gap_power + 1));
            unsigned int global_offset = (k * coeff_modulus_size + j) << log_degree;

            unsigned int from_x_index = 
                ((threadIdx.x >> E_power) << (gap_power + 1))
                + ((threadIdx.x & E_mask) << stride_power) 
                + coefficient_offset 
                + global_offset;
            unsigned int from_y_index = from_x_index + (1 << gap_power);
            unsigned int to_x_index = ((threadIdx.x & (~E_mask)) << 1) + (threadIdx.x & E_mask);
            unsigned int to_y_index = to_x_index + (1 << E_power);
            sdata[to_x_index] = operand[from_x_index];
            sdata[to_y_index] = operand[from_y_index];
            if constexpr (NTT_KERNEL_THREAD_COUNT > 32) __syncthreads();

            coefficient_offset = (block_idx & stride_mask) + (((block_idx >> stride_power) * (blockDim.x >> E_power)) << gap_power);

            for (unsigned int layer = layer_lower; layer < layer_upper; layer++) {

                unsigned int rid = (1 << layer) + (threadIdx.x >> E_power) + ((((threadIdx.x & E_mask) << stride_power) + coefficient_offset) >> gap_power);
                const MultiplyUint64Operand& r = r_ptr[rid];

                unsigned int x_index = ((threadIdx.x & (~E_mask)) << 1) + (threadIdx.x & E_mask); // wrt shared data
                unsigned int y_index = x_index + E_mask + 1;
                
                uint64_t& x = sdata[x_index];
                uint64_t& y = sdata[y_index];
                uint64_t u = (x >= two_times_modulus) ? (x - two_times_modulus) : x;
                uint64_t v = utils::multiply_uint64operand_mod_lazy(y, r, modulus);
                x = u + v;
                y = u + two_times_modulus - v;

                if constexpr (NTT_KERNEL_THREAD_COUNT > 32) __syncthreads();

                E_power -= 1;
                E_mask = (1 << E_power) - 1;
                gap_power -= 1;
            }

            uint64_t mv = modulus.value();
            if (sdata[to_x_index] >= two_times_modulus) sdata[to_x_index] -= two_times_modulus;
            if (sdata[to_x_index] >= mv) sdata[to_x_index] -= mv;
            if (sdata[to_y_index] >= two_times_modulus) sdata[to_y_index] -= two_times_modulus;
            if (sdata[to_y_index] >= mv) sdata[to_y_index] -= mv;

            result[from_x_index] = sdata[to_x_index];
            result[from_y_index] = sdata[to_y_index];

            operand = result.as_const();

            // here we need to sync not only threads but blocks
            if (layer_upper < log_degree) {
                group.sync();
            }

        }

    }

    __global__ void kernel_ntt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {
        device_ntt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
    }

    __global__ void kernel_ntt_batched(ConstSliceArrayRef<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, SliceArrayRef<uint64_t> result, NTTTableIndexer tables) {
        for (size_t i = 0; i < result.size(); i++) {
            device_ntt(operand[i], pcount, component_count, log_degree, use_inv_root_powers, result[i], tables);
        }
    }

    void ntt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {
        bool device = operand.on_device();
        // same device
        if (!device_compatible(operand, tables, result)) {
            throw std::invalid_argument("[ntt_transfer_to_rev] Operand and tables must be on the same device.");
        }
        if (!device) {
            // directly use ntt_grouped's ntt.
            ntt_grouped::ntt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
        } else {
            size_t degree = static_cast<size_t>(1) << log_degree;
            size_t thread_count = std::min(NTT_KERNEL_THREAD_COUNT, degree / 2);
            size_t total = pcount * component_count * (degree / 2);
            size_t block_count = ceil_div<size_t>(total, thread_count);
            assert(block_count == total / thread_count);

            ensure_device_properties();
            int device_index = operand.device_index();
            const cudaDeviceProp& prop = device_properties[device_index];

            if (static_cast<int>(block_count) > prop.multiProcessorCount && prop.cooperativeLaunch) {
                // directly use ntt_grouped's ntt.
                ntt_grouped::ntt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
            } else {
                void* kernel_args[] = {
                    &operand, &pcount, &component_count, &log_degree, &use_inv_root_powers, &result, &tables
                };
                utils::set_device(operand.device_index());
                cudaLaunchCooperativeKernel(
                    (void*)kernel_ntt,
                    block_count, thread_count,
                    kernel_args
                );
                utils::stream_sync();
            }
        }
    }

    void ntt_batched(
        const ConstSliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, 
        const SliceVec<uint64_t>& result, NTTTableIndexer tables,
        MemoryPoolHandle pool
    ) {
        if (operand.size() != result.size()) {
            throw std::invalid_argument("[ntt_transfer_to_rev_batched] Operand and result must have the same size.");
        }
        if (operand.empty()) return;
        bool device = operand[0].on_device();
        if (!device || operand.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < operand.size(); i++) {
                ntt(operand[i], pcount, component_count, log_degree, use_inv_root_powers, result[i], tables);
            }
        } else {
            auto comp_ref = operand[0];

            size_t degree = static_cast<size_t>(1) << log_degree;
            size_t thread_count = std::min(NTT_KERNEL_THREAD_COUNT, degree / 2);
            size_t total = pcount * component_count * (degree / 2);
            size_t block_count = ceil_div<size_t>(total, thread_count);
            assert(block_count == total / thread_count);

            ensure_device_properties();
            int device_index = comp_ref.device_index();
            const cudaDeviceProp& prop = device_properties[device_index];

            if (static_cast<int>(block_count) > prop.multiProcessorCount && prop.cooperativeLaunch) {
                // directly use ntt_grouped's ntt.
                ntt_grouped::ntt_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
            } else {
                auto operand_batch = batch_utils::construct_batch(operand, pool, comp_ref);
                auto result_batch = batch_utils::construct_batch(result, pool, comp_ref);
                void* kernel_args[] = {
                    &operand_batch, &pcount, &component_count, &log_degree, &use_inv_root_powers, &result_batch, &tables
                };
                utils::set_device(device_index);
                cudaLaunchCooperativeKernel(
                    (void*)kernel_ntt_batched,
                    block_count, thread_count,
                    kernel_args
                );
                utils::stream_sync();
            }
        }
    }
    



    __device__ void device_intt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {
        unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int i_upperbound = 1 << (log_degree - 1);
        unsigned int coeff_modulus_size = component_count;

        unsigned int k = global_index / (coeff_modulus_size * i_upperbound);
        unsigned int j = (global_index / i_upperbound) % coeff_modulus_size;

        const NTTTables& table = tables.get(k, j);
        const Modulus& modulus = table.modulus();
        uint64_t two_times_modulus = modulus.value() << 1;
        const MultiplyUint64Operand* r_ptr = use_inv_root_powers ?
            table.inv_root_powers().raw_pointer() :
            table.root_powers().raw_pointer();

        unsigned int block_idx = blockIdx.x % (gridDim.x / (pcount * coeff_modulus_size));

        __shared__ uint64_t sdata[NTT_KERNEL_THREAD_COUNT * 2];
        
        cooperative_groups::grid_group group = cooperative_groups::this_grid();

        for (unsigned int layer_lower = 0; layer_lower < log_degree; layer_lower += NTT_KERNEL_THREAD_COUNT_LOG2) {

            unsigned int layer_upper = min(layer_lower + NTT_KERNEL_THREAD_COUNT_LOG2, log_degree);

            unsigned int gap_power = layer_upper - 1;
            unsigned int E_power = min(static_cast<unsigned int>(get_power_of_two(blockDim.x)), gap_power); // elements in gap
            unsigned int E_mask = (1 << E_power) - 1;
            unsigned int stride_power = gap_power - E_power;
            unsigned int stride_mask = (1 << stride_power) - 1;

            unsigned int global_offset = (k * coeff_modulus_size + j) << log_degree;
            unsigned int coefficient_offset = (block_idx & stride_mask) + (((block_idx >> stride_power) * (blockDim.x >> E_power)) << (gap_power + 1));

            unsigned int from_x_index = 
                ((threadIdx.x >> E_power) << (gap_power + 1))
                + ((threadIdx.x & E_mask) << stride_power) 
                + coefficient_offset 
                + global_offset;
            unsigned int from_y_index = from_x_index + (1 << gap_power);
            unsigned int to_x_index = ((threadIdx.x & (~E_mask)) << 1) + (threadIdx.x & E_mask);
            unsigned int to_y_index = to_x_index + (1 << E_power);
            sdata[to_x_index] = operand[from_x_index];
            sdata[to_y_index] = operand[from_y_index];
            if constexpr (NTT_KERNEL_THREAD_COUNT > 32) __syncthreads();
            
            coefficient_offset = (block_idx & stride_mask) + (((block_idx >> stride_power) * (blockDim.x >> E_power)) << gap_power);

            gap_power -= (layer_upper - layer_lower - 1);
            E_power -= (layer_upper - layer_lower - 1);
            E_mask = (1 << E_power) - 1;

            for (unsigned int layer = layer_lower; layer < layer_upper; layer++) {

                unsigned int rid = (1 << log_degree) - (1 << (log_degree - layer)) + 1
                    + (threadIdx.x >> E_power) + ((((threadIdx.x & E_mask) << stride_power) + coefficient_offset) >> gap_power);

                unsigned int x_index = ((threadIdx.x & (~E_mask)) << 1) + (threadIdx.x & E_mask); // wrt shared data
                unsigned int y_index = x_index + E_mask + 1;

                const MultiplyUint64Operand& r = r_ptr[rid];
                
                uint64_t u = sdata[x_index];
                uint64_t v = sdata[y_index];
                sdata[x_index] = (u + v > two_times_modulus) ? (u + v - two_times_modulus) : (u + v);
                sdata[y_index] = utils::multiply_uint64operand_mod_lazy(u + two_times_modulus - v, r, modulus);

                if constexpr (NTT_KERNEL_THREAD_COUNT > 32) __syncthreads();

                E_power += 1;
                E_mask = (1 << E_power) - 1;
                gap_power += 1;

            }
            
            uint64_t mv = modulus.value();
            if (sdata[to_x_index] >= two_times_modulus) sdata[to_x_index] -= two_times_modulus;
            if (sdata[to_x_index] >= mv) sdata[to_x_index] -= mv;
            if (sdata[to_y_index] >= two_times_modulus) sdata[to_y_index] -= two_times_modulus;
            if (sdata[to_y_index] >= mv) sdata[to_y_index] -= mv;

            if (layer_upper == log_degree) {
                const Modulus& modulus = table.modulus();
                MultiplyUint64Operand scalar = table.inv_degree_modulo();
                sdata[to_x_index] = multiply_uint64operand_mod_lazy(sdata[to_x_index], scalar, modulus);
                sdata[to_y_index] = multiply_uint64operand_mod_lazy(sdata[to_y_index], scalar, modulus);
            }

            result[from_x_index] = sdata[to_x_index];
            result[from_y_index] = sdata[to_y_index];

            operand = result.as_const();

            // here we need to sync not only threads but blocks
            if (layer_upper < log_degree) {
                group.sync();
            }
        }
    }

    __global__ void kernel_intt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {
        device_intt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
    }

    __global__ void kernel_intt_batched(ConstSliceArrayRef<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, SliceArrayRef<uint64_t> result, NTTTableIndexer tables) {
        for (size_t i = 0; i < result.size(); i++) {
            device_intt(operand[i], pcount, component_count, log_degree, use_inv_root_powers, result[i], tables);
        }
    }

    void intt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {
        bool device = operand.on_device();
        // same device
        if (!device_compatible(operand, tables, result)) {
            throw std::invalid_argument("[ntt_transfer_from_rev] Operand and tables must be on the same device.");
        }
        if (!device) {
            ntt_grouped::intt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
        } else {
            size_t degree = static_cast<size_t>(1) << log_degree;
            size_t thread_count = std::min(NTT_KERNEL_THREAD_COUNT, degree / 2);
            size_t total = pcount * component_count * (degree / 2);
            size_t block_count = ceil_div<size_t>(total, thread_count);
            assert(block_count == total / thread_count);

            ensure_device_properties();
            int device_index = operand.device_index();
            const cudaDeviceProp& prop = device_properties[device_index];

            if (static_cast<int>(block_count) > prop.multiProcessorCount && prop.cooperativeLaunch) {
                // directly use ntt_grouped's ntt.
                ntt_grouped::intt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
            } else {
                void* kernel_args[] = {
                    &operand, &pcount, &component_count, &log_degree, &use_inv_root_powers, &result, &tables
                };
                utils::set_device(operand.device_index());
                cudaLaunchCooperativeKernel(
                    (void*)kernel_intt,
                    block_count, thread_count,
                    kernel_args
                );
                utils::stream_sync();
            }
        }
    }

    void intt_batched(
        const ConstSliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, 
        const SliceVec<uint64_t>& result, NTTTableIndexer tables,
        MemoryPoolHandle pool
    ) {
        if (operand.size() != result.size()) {
            throw std::invalid_argument("[ntt_transfer_to_rev_batched] Operand and result must have the same size.");
        }
        if (operand.empty()) return;
        bool device = operand[0].on_device();
        if (!device || operand.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < operand.size(); i++) {
                intt(operand[i], pcount, component_count, log_degree, use_inv_root_powers, result[i], tables);
            }
        } else {
            auto comp_ref = operand[0];

            size_t degree = static_cast<size_t>(1) << log_degree;
            size_t thread_count = std::min(NTT_KERNEL_THREAD_COUNT, degree / 2);
            size_t total = pcount * component_count * (degree / 2);
            size_t block_count = ceil_div<size_t>(total, thread_count);
            assert(block_count == total / thread_count);

            ensure_device_properties();
            int device_index = comp_ref.device_index();
            const cudaDeviceProp& prop = device_properties[device_index];

            if (static_cast<int>(block_count) > prop.multiProcessorCount && prop.cooperativeLaunch) {
                // directly use ntt_grouped's ntt.
                ntt_grouped::intt_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
            } else {
                auto operand_batch = batch_utils::construct_batch(operand, pool, comp_ref);
                auto result_batch = batch_utils::construct_batch(result, pool, comp_ref);
                void* kernel_args[] = {
                    &operand_batch, &pcount, &component_count, &log_degree, &use_inv_root_powers, &result_batch, &tables
                };
                utils::set_device(device_index);
                cudaLaunchCooperativeKernel(
                    (void*)kernel_intt_batched,
                    block_count, thread_count,
                    kernel_args
                );
                utils::stream_sync();
            }
        }
    }


}
