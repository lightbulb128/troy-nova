#include "ntt_grouped.h"
#include <cassert>

namespace troy::utils::fgk::ntt_grouped {

    static constexpr size_t NTT_KERNEL_THREAD_COUNT = 256;
    static constexpr size_t NTT_KERNEL_THREAD_COUNT_LOG2 = 8;

    void host_ntt_transfer_to_rev_layer(size_t layer, ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result) {
        size_t m = 1 << layer;
        size_t gap_power = log_degree - layer - 1;
        size_t gap = 1 << gap_power;
        size_t i_upperbound = 1 << (log_degree - 1);
        size_t coeff_modulus_size = tables.size();
        for (size_t j = 0; j < coeff_modulus_size; j++) {
            const Modulus& modulus = tables[j].modulus();
            uint64_t two_times_modulus = modulus.value() << 1;
            for (size_t i = 0; i < i_upperbound; i++) {
                size_t rid = m + (i >> gap_power);
                size_t coeff_index = ((i >> gap_power) << (gap_power + 1)) + (i & (gap - 1));
                MultiplyUint64Operand r = use_inv_root_powers ?
                    tables[j].inv_root_powers()[rid] :
                    tables[j].root_powers()[rid];
                for (size_t k = 0; k < pcount; k++) {
                    size_t x_index = ((k * coeff_modulus_size + j) << log_degree) + coeff_index;
                    size_t y_index = x_index + gap;
                    uint64_t x = operand[x_index];
                    uint64_t y = operand[y_index];
                    uint64_t u = (x >= two_times_modulus) ? (x - two_times_modulus) : x;
                    uint64_t v = utils::multiply_uint64operand_mod_lazy(y, r, modulus);
                    x = u + v;
                    y = u + two_times_modulus - v;
                    result[x_index] = x;
                    result[y_index] = y;
                }
            }
        }
        if (layer == log_degree - 1) {
            size_t n = 1 << log_degree;
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                const Modulus& modulus = tables[j].modulus();
                uint64_t mv = modulus.value();
                uint64_t tmv = modulus.value() << 1;
                for (size_t i = 0; i < n; i++) {
                    for (size_t k = 0; k < pcount; k++) {
                        size_t index = ((k * coeff_modulus_size + j) << log_degree) + i;
                        if (result[index] >= tmv) result[index] -= tmv;
                        if (result[index] >= mv) result[index] -= mv;
                    }
                }
            }
        }
    }

    /*
    __global__ void kernel_ntt_transfer_to_rev_layer1(size_t layer, Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i_upperbound = 1 << (log_degree - 1);
        size_t coeff_modulus_size = tables.size();
        if (global_index >= (pcount * coeff_modulus_size * i_upperbound)) {
            return;
        }

        size_t k = global_index / (coeff_modulus_size * i_upperbound);
        size_t j = (global_index / i_upperbound) % coeff_modulus_size;
        size_t i = global_index % i_upperbound;

        size_t m = 1 << layer;
        size_t gap_power = log_degree - layer - 1;
        size_t gap = 1 << gap_power;

        const Modulus& modulus = tables[j].modulus();
        uint64_t two_times_modulus = modulus.value() << 1;
        size_t rid = m + (i >> gap_power);
        size_t coeff_index = ((i >> gap_power) << (gap_power + 1)) + (i & (gap - 1));
        MultiplyUint64Operand r = use_inv_root_powers ?
            tables[j].inv_root_powers()[rid] :
            tables[j].root_powers()[rid];
        
        size_t x_index = ((k * coeff_modulus_size + j) << log_degree) + coeff_index;
        size_t y_index = x_index + gap;
        uint64_t x = operand[x_index];
        uint64_t y = operand[y_index];
        uint64_t u = (x >= two_times_modulus) ? (x - two_times_modulus) : x;
        uint64_t v = utils::multiply_uint64operand_mod_lazy(y, r, modulus);
        x = u + v;
        y = u + two_times_modulus - v;

        operand[x_index] = x;
        operand[y_index] = y;
    }
    */

    /* // This old version does not use shared memory
    __global__ void kernel_ntt_transfer_to_rev_layers(size_t layer_lower, size_t layer_upper, Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i_upperbound = 1 << (log_degree - 1);
        size_t coeff_modulus_size = tables.size();
        if (global_index >= (pcount * coeff_modulus_size * i_upperbound)) {
            return;
        }

        size_t k = global_index / (coeff_modulus_size * i_upperbound);
        size_t j = (global_index / i_upperbound) % coeff_modulus_size;

        size_t block_idx = static_cast<size_t>(blockIdx.x) % (gridDim.x / (pcount * coeff_modulus_size));
        size_t gap_power = log_degree - layer_lower - 1;
        size_t gap = 1 << gap_power;
        size_t E = min(static_cast<size_t>(blockDim.x), gap); // elements in gap
        size_t C = blockDim.x / E; // gaps crossed
        size_t stride = gap / E;

        size_t component_global_offset = (k * coeff_modulus_size + j) << log_degree;
        size_t coefficient_offset = block_idx % stride + (block_idx / stride) * C * 2 * gap;

        const Modulus& modulus = tables[j].modulus();
        uint64_t two_times_modulus = modulus.value() << 1;

        for (size_t dl = 0; dl < layer_upper - layer_lower; dl++) {

            size_t layer = layer_lower + dl;

            size_t x_index = threadIdx.x / E * 2 * gap + threadIdx.x % E * stride + coefficient_offset;
            
            size_t m = 1 << layer;

            size_t i = ((x_index >> (gap_power + 1)) << gap_power) + (x_index & (gap - 1));
            size_t rid = m + (i / gap);

            x_index += component_global_offset;
            size_t y_index = x_index + gap;
            
            MultiplyUint64Operand r = use_inv_root_powers ?
                tables[j].inv_root_powers()[rid] :
                tables[j].root_powers()[rid];
            uint64_t x = operand[x_index];
            uint64_t y = operand[y_index];
            uint64_t u = (x >= two_times_modulus) ? (x - two_times_modulus) : x;
            uint64_t v = utils::multiply_uint64operand_mod_lazy(y, r, modulus);
            x = u + v;
            y = u + two_times_modulus - v;

            operand[x_index] = x;
            operand[y_index] = y;

            __syncthreads();

            E >>= 1;
            gap >>= 1;
            gap_power -= 1;

        }
    }
    */

    __global__ void kernel_ntt_transfer_to_rev_layers(size_t layer_lower, size_t layer_upper, ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result) {
        unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int coeff_modulus_size = tables.size();

        unsigned int k = global_index / (coeff_modulus_size << (log_degree - 1));
        unsigned int j = (global_index >> (log_degree - 1)) % coeff_modulus_size;
        
        const Modulus& modulus = tables[j].modulus();
        uint64_t two_times_modulus = modulus.value() << 1;
        const MultiplyUint64Operand* r_ptr = use_inv_root_powers ?
            tables[j].inv_root_powers().raw_pointer() :
            tables[j].root_powers().raw_pointer();

        unsigned int block_idx = blockIdx.x % (gridDim.x / (pcount * coeff_modulus_size));
        unsigned int gap_power = log_degree - layer_lower - 1;
        unsigned int E_power = min(static_cast<unsigned int>(get_power_of_two(blockDim.x)), gap_power); // elements in gap
        unsigned int E_mask = (1 << E_power) - 1;
        unsigned int stride_power = gap_power - E_power;
        unsigned int stride_mask = (1 << stride_power) - 1;

        unsigned int coefficient_offset = (block_idx & stride_mask) + (((block_idx >> stride_power) * (blockDim.x >> E_power)) << (gap_power + 1));
        unsigned int global_offset = (k * coeff_modulus_size + j) << log_degree;

        __shared__ uint64_t sdata[NTT_KERNEL_THREAD_COUNT * 2];
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
        __syncthreads();

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

            __syncthreads();

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
    }

    void ntt(ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result) {
        bool device = operand.on_device();
        // same device
        if (!device_compatible(operand, tables, result)) {
            throw std::invalid_argument("[ntt_transfer_to_rev] Operand and tables must be on the same device.");
        }
        if (!device) {
            for (size_t layer = 0; layer < log_degree; layer++) {
                host_ntt_transfer_to_rev_layer(layer, operand, pcount, log_degree, tables, use_inv_root_powers, result);
                operand = result.as_const();
            }
        } else {
            if (log_degree <= NTT_KERNEL_THREAD_COUNT_LOG2) {
                size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                size_t thread_count = 1 << (log_degree - 1);
                size_t block_count = ceil_div<size_t>(total, thread_count);
                assert(block_count == total / thread_count);
                utils::set_device(operand.device_index());
                kernel_ntt_transfer_to_rev_layers<<<block_count, thread_count>>>(
                    0, log_degree, operand, pcount, log_degree, tables, use_inv_root_powers, result
                );
                utils::stream_sync();
            } else {
                for (size_t layer_lower = 0; layer_lower < log_degree; layer_lower += NTT_KERNEL_THREAD_COUNT_LOG2) {
                    size_t layer_upper = std::min(layer_lower + NTT_KERNEL_THREAD_COUNT_LOG2, log_degree);
                    size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                    size_t block_count = ceil_div<size_t>(total, NTT_KERNEL_THREAD_COUNT);
                    assert(block_count == total / NTT_KERNEL_THREAD_COUNT);
                    utils::set_device(operand.device_index());
                    kernel_ntt_transfer_to_rev_layers<<<block_count, NTT_KERNEL_THREAD_COUNT>>>(
                        layer_lower, layer_upper, operand, pcount, log_degree, tables, use_inv_root_powers, result
                    );
                    utils::stream_sync();
                    operand = result.as_const();
                }
            }
        }
    }

    void host_ntt_transfer_from_rev_layer(size_t layer, ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result) {
        size_t m = 1 << (log_degree - layer - 1);
        size_t gap_power = layer;
        size_t gap = 1 << gap_power;
        size_t i_upperbound = 1 << (log_degree - 1);
        size_t coeff_modulus_size = tables.size();
        for (size_t j = 0; j < coeff_modulus_size; j++) {
            const Modulus& modulus = tables[j].modulus();
            uint64_t two_times_modulus = modulus.value() << 1;
            for (size_t i = 0; i < i_upperbound; i++) {
                size_t rid = (1 << log_degree) - (m << 1) + 1 + (i >> gap_power);
                size_t coeff_index = ((i >> gap_power) << (gap_power + 1)) + (i & (gap - 1));
                MultiplyUint64Operand r = use_inv_root_powers ?
                    tables[j].inv_root_powers()[rid] :
                    tables[j].root_powers()[rid];
                for (size_t k = 0; k < pcount; k++) {
                    size_t x_index = ((k * coeff_modulus_size + j) << log_degree) + coeff_index;
                    size_t y_index = x_index + gap;
                    uint64_t u = operand[x_index];
                    uint64_t v = operand[y_index];
                    result[x_index] = (u + v > two_times_modulus) ? (u + v - two_times_modulus) : (u + v);
                    result[y_index] = utils::multiply_uint64operand_mod_lazy(u + two_times_modulus - v, r, modulus);
                }
            }
        }
        if (layer == log_degree - 1) {
            size_t n = 1 << log_degree;
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                const Modulus& modulus = tables[j].modulus();
                MultiplyUint64Operand scalar = tables[j].inv_degree_modulo();
                uint64_t mv = modulus.value();
                uint64_t tmv = modulus.value() << 1;
                for (size_t i = 0; i < n; i++) {
                    for (size_t k = 0; k < pcount; k++) {
                        size_t index = ((k * coeff_modulus_size + j) << log_degree) + i;
                        if (result[index] >= tmv) result[index] -= tmv;
                        if (result[index] >= mv) result[index] -= mv;
                        result[index] = multiply_uint64operand_mod_lazy(result[index], scalar, modulus);
                    }
                }
            }
        }
        
    }

    /*
    __global__ void kernel_ntt_transfer_from_rev_layer1(size_t layer, Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i_upperbound = 1 << (log_degree - 1);
        size_t coeff_modulus_size = tables.size();
        if (global_index >= (pcount * coeff_modulus_size * i_upperbound)) {
            return;
        }
        
        size_t k = global_index / (coeff_modulus_size * i_upperbound);
        size_t j = (global_index / i_upperbound) % coeff_modulus_size;
        size_t i = global_index % i_upperbound;

        size_t m = 1 << (log_degree - layer - 1);
        size_t gap_power = layer;
        size_t gap = 1 << gap_power;

        const Modulus& modulus = tables[j].modulus();
        uint64_t two_times_modulus = modulus.value() << 1;
        size_t rid = (1 << log_degree) - (m << 1) + 1 + (i >> gap_power);
        size_t coeff_index = ((i >> gap_power) << (gap_power + 1)) + (i & (gap - 1));
        MultiplyUint64Operand r = use_inv_root_powers ?
            tables[j].inv_root_powers()[rid] :
            tables[j].root_powers()[rid];
        
        size_t x_index = ((k * coeff_modulus_size + j) << log_degree) + coeff_index;
        size_t y_index = x_index + gap;
        uint64_t u = operand[x_index];
        uint64_t v = operand[y_index];
        operand[x_index] = (u + v > two_times_modulus) ? (u + v - two_times_modulus) : (u + v);
        operand[y_index] = utils::multiply_uint64operand_mod_lazy(u + two_times_modulus - v, r, modulus);

    }
    */

    /* // This old version does not use shared memory
    __global__ void kernel_ntt_transfer_from_rev_layers(size_t layer_lower, size_t layer_upper, Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t i_upperbound = 1 << (log_degree - 1);
        size_t coeff_modulus_size = tables.size();
        if (global_index >= (pcount * coeff_modulus_size * i_upperbound)) {
            return;
        }

        size_t k = global_index / (coeff_modulus_size * i_upperbound);
        size_t j = (global_index / i_upperbound) % coeff_modulus_size;

        size_t block_idx = static_cast<size_t>(blockIdx.x) % (gridDim.x / (pcount * coeff_modulus_size));
        size_t gap_power = layer_upper - 1;
        size_t gap = 1 << gap_power;
        size_t E = min(static_cast<size_t>(blockDim.x), gap); // elements in gap
        size_t C = blockDim.x / E; // gaps crossed
        size_t stride = gap / E;

        size_t component_global_offset = (k * coeff_modulus_size + j) << log_degree;
        size_t coefficient_offset = block_idx % stride + (block_idx / stride) * C * 2 * gap;

        gap >>= (layer_upper - layer_lower - 1);
        gap_power -= (layer_upper - layer_lower - 1);
        E >>= (layer_upper - layer_lower - 1);

        const Modulus& modulus = tables[j].modulus();
        uint64_t two_times_modulus = modulus.value() << 1;
        const MultiplyUint64Operand* r_ptr = use_inv_root_powers ?
            tables[j].inv_root_powers().raw_pointer() :
            tables[j].root_powers().raw_pointer();

        for (size_t layer = layer_lower; layer < layer_upper; layer++) {

            size_t x_index = threadIdx.x / E * 2 * gap + threadIdx.x % E * stride + coefficient_offset;
            
            size_t m = 1 << (log_degree - layer - 1);

            size_t i = ((x_index >> (gap_power + 1)) << gap_power) + (x_index & (gap - 1));
            size_t rid = (1 << log_degree) - (m << 1) + 1 + (i >> gap_power);

            x_index += component_global_offset;
            size_t y_index = x_index + gap;

            const MultiplyUint64Operand& r = r_ptr[rid];
            
            uint64_t u = operand[x_index];
            uint64_t v = operand[y_index];
            operand[x_index] = (u + v > two_times_modulus) ? (u + v - two_times_modulus) : (u + v);
            operand[y_index] = utils::multiply_uint64operand_mod_lazy(u + two_times_modulus - v, r, modulus);

            __syncthreads();

            E <<= 1;
            gap <<= 1;
            gap_power += 1;

        }
    }
    */

    __global__ void kernel_ntt_transfer_from_rev_layers(size_t layer_lower, size_t layer_upper, ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result) {
        unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int i_upperbound = 1 << (log_degree - 1);
        unsigned int coeff_modulus_size = tables.size();

        unsigned int k = global_index / (coeff_modulus_size * i_upperbound);
        unsigned int j = (global_index / i_upperbound) % coeff_modulus_size;

        const Modulus& modulus = tables[j].modulus();
        uint64_t two_times_modulus = modulus.value() << 1;
        const MultiplyUint64Operand* r_ptr = use_inv_root_powers ?
            tables[j].inv_root_powers().raw_pointer() :
            tables[j].root_powers().raw_pointer();

        unsigned int block_idx = blockIdx.x % (gridDim.x / (pcount * coeff_modulus_size));
        unsigned int gap_power = layer_upper - 1;
        unsigned int E_power = min(static_cast<unsigned int>(get_power_of_two(blockDim.x)), gap_power); // elements in gap
        unsigned int E_mask = (1 << E_power) - 1;
        unsigned int stride_power = gap_power - E_power;
        unsigned int stride_mask = (1 << stride_power) - 1;

        unsigned int global_offset = (k * coeff_modulus_size + j) << log_degree;
        unsigned int coefficient_offset = (block_idx & stride_mask) + (((block_idx >> stride_power) * (blockDim.x >> E_power)) << (gap_power + 1));

        __shared__ uint64_t sdata[NTT_KERNEL_THREAD_COUNT * 2];
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
        __syncthreads();
        
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

            __syncthreads();

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
            const Modulus& modulus = tables[j].modulus();
            MultiplyUint64Operand scalar = tables[j].inv_degree_modulo();
            sdata[to_x_index] = multiply_uint64operand_mod_lazy(sdata[to_x_index], scalar, modulus);
            sdata[to_y_index] = multiply_uint64operand_mod_lazy(sdata[to_y_index], scalar, modulus);
        }

        result[from_x_index] = sdata[to_x_index];
        result[from_y_index] = sdata[to_y_index];
    }

    void intt(ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result) {
        bool device = operand.on_device();
        // same device
        if (!device_compatible(operand, tables, result)) {
            throw std::invalid_argument("[ntt_transfer_from_rev] Operand and tables must be on the same device.");
        }
        if (!device) {
            for (size_t layer = 0; layer < log_degree; layer++) {
                host_ntt_transfer_from_rev_layer(layer, operand, pcount, log_degree, tables, use_inv_root_powers, result);
                operand = result.as_const();
            }
        } else {
            if (log_degree <= NTT_KERNEL_THREAD_COUNT_LOG2) {
                size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                size_t thread_count = 1 << (log_degree - 1);
                size_t block_count = ceil_div<size_t>(total, thread_count);
                assert(block_count == total / thread_count);
                utils::set_device(operand.device_index());
                kernel_ntt_transfer_from_rev_layers<<<block_count, thread_count>>>(
                    0, log_degree, operand, pcount, log_degree, tables, use_inv_root_powers, result
                );
                utils::stream_sync();
            } else {
                for (size_t layer_lower = 0; layer_lower < log_degree; layer_lower += NTT_KERNEL_THREAD_COUNT_LOG2) {
                    size_t layer_upper = std::min(layer_lower + NTT_KERNEL_THREAD_COUNT_LOG2, log_degree);
                    size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                    size_t block_count = ceil_div<size_t>(total, NTT_KERNEL_THREAD_COUNT);
                    assert(block_count == total / NTT_KERNEL_THREAD_COUNT);
                    utils::set_device(operand.device_index());
                    kernel_ntt_transfer_from_rev_layers<<<block_count, NTT_KERNEL_THREAD_COUNT>>>(
                        layer_lower, layer_upper, operand, pcount, log_degree, tables, use_inv_root_powers, result
                    );
                    utils::stream_sync();
                    operand = result.as_const();
                }
            }
        }
    }

}