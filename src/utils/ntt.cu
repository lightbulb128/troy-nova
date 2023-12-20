#include "ntt.cuh"
#include "timer.h"

namespace troy {namespace utils {

    static const size_t NTT_KERNEL_THREAD_COUNT = 256;
    static const size_t NTT_KERNEL_THREAD_COUNT_LOG2 = 8;

    NTTTables::NTTTables(size_t coeff_count_power, const Modulus& modulus) {

        size_t coeff_count = static_cast<size_t>(1) << coeff_count_power;
        
        // We defer parameter checking to try_minimal_primitive_root(...)

        uint64_t root = 0;
        if (!utils::try_minimal_primitive_root(
            static_cast<uint64_t>(2 * coeff_count),
            modulus,
            root
        )) {
            throw std::invalid_argument("[NTTTables::NTTTables] Invalid modulus, unable to find primitive root.");
        }

        uint64_t inv_root = 0;
        if (!try_invert_uint64_mod(root, modulus, inv_root)) {
            throw std::invalid_argument("[NTTTables::NTTTables] Invalid modulus, unable to invert.");
        }

        // Populate tables with powers of root in specific orders.
        
        Array<MultiplyUint64Operand> root_powers(coeff_count, false);
        MultiplyUint64Operand root_operand(root, modulus);
        uint64_t power = root;
        for (size_t i = 1; i < coeff_count; i++) {
            root_powers[static_cast<size_t>(utils::reverse_bits_uint64(
                static_cast<uint64_t>(i),
                coeff_count_power
            ))] = MultiplyUint64Operand(power, modulus);
            power = utils::multiply_uint64operand_mod(power, root_operand, modulus);
        }
        root_powers[0] = MultiplyUint64Operand(1, modulus);

        Array<MultiplyUint64Operand> inv_root_powers(coeff_count, false);
        root_operand = MultiplyUint64Operand(inv_root, modulus);
        power = inv_root;
        for (size_t i = 1; i < coeff_count; i++) {
            inv_root_powers[static_cast<size_t>(utils::reverse_bits_uint64(
                static_cast<uint64_t>(i - 1),
                coeff_count_power
            )) + 1] = MultiplyUint64Operand(power, modulus);
            power = utils::multiply_uint64operand_mod(power, root_operand, modulus);
        }
        inv_root_powers[0] = MultiplyUint64Operand(1, modulus);

        uint64_t degree_uint64 = static_cast<uint64_t>(coeff_count);
        uint64_t inv_degree_modulo = 0;
        if (!utils::try_invert_uint64_mod(degree_uint64, modulus, inv_degree_modulo)) {
            throw std::invalid_argument("[NTTTables::NTTTables] Invalid modulus, unable to invert degree.");
        }
        MultiplyUint64Operand inv_degree_modulo_operand(inv_degree_modulo, modulus);

        this->root_ = root;
        this->coeff_count_ = coeff_count;
        this->coeff_count_power_ = coeff_count_power;
        this->modulus_ = modulus;
        this->inv_degree_modulo_ = inv_degree_modulo_operand;
        this->root_powers_ = std::move(root_powers);
        this->inv_root_powers_ = std::move(inv_root_powers);
        this->device = false;

    }

    void host_ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        size_t degree = static_cast<size_t>(1) << log_degree; 
        for (size_t j = 0; j < tables.size(); j++) {
            const Modulus& modulus = tables[j].modulus();
            MultiplyUint64Operand scalar = tables[j].inv_degree_modulo();
            for (size_t k = 0; k < pcount; k++) {
                for (size_t i = 0; i < degree; i++) {
                    size_t x_index = ((k * tables.size() + j) << log_degree) + i;
                    operand[x_index] = multiply_uint64operand_mod_lazy(operand[x_index], scalar, modulus);
                }
            }
        }
    }

    __global__ void kernel_ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t degree = static_cast<size_t>(1) << log_degree;
        size_t total = pcount * tables.size() * degree;
        if (global_index < total) {
            // size_t k = global_index / (tables.size() * degree);
            size_t j = (global_index / degree) % tables.size();
            // size_t i = global_index % degree;
            const Modulus& modulus = tables[j].modulus();
            MultiplyUint64Operand scalar = tables[j].inv_degree_modulo();
            operand[global_index] = multiply_uint64operand_mod_lazy(operand[global_index], scalar, modulus);
        }
    }

    void ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        bool device = operand.on_device();
        // same device
        if (device != tables.on_device()) {
            throw std::invalid_argument("[ntt_multiply_inv_degree] Operand and tables must be on the same device.");
        }
        if (device) {
            size_t total = (pcount * tables.size()) << log_degree;
            size_t block_count = ceil_div<size_t>(total, KERNEL_THREAD_COUNT);
            kernel_ntt_multiply_inv_degree<<<block_count, KERNEL_THREAD_COUNT>>>(operand, pcount, log_degree, tables);
        } else {
            host_ntt_multiply_inv_degree(operand, pcount, log_degree, tables);
        }
    }

    void host_ntt_transfer_to_rev_layer(size_t layer, Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
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
                    operand[x_index] = x;
                    operand[y_index] = y;
                }
            }
        }
    }

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


    /* This old version does not use shared memory
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

    __global__ void kernel_ntt_transfer_to_rev_layers(size_t layer_lower, size_t layer_upper, Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int i_upperbound = 1 << (log_degree - 1);
        unsigned int coeff_modulus_size = tables.size();
        if (global_index >= (pcount * coeff_modulus_size * i_upperbound)) {
            return;
        }

        unsigned int k = global_index / (coeff_modulus_size * i_upperbound);
        unsigned int j = (global_index / i_upperbound) % coeff_modulus_size;
        
        const Modulus& modulus = tables[j].modulus();
        uint64_t two_times_modulus = modulus.value() << 1;

        unsigned int block_idx = blockIdx.x % (gridDim.x / (pcount * coeff_modulus_size));
        unsigned int gap_power = log_degree - layer_lower - 1;
        unsigned int gap = 1 << gap_power;
        unsigned int E = min(blockDim.x, gap); // elements in gap
        unsigned int stride = gap / E;

        unsigned int coefficient_offset = block_idx % stride + (block_idx / stride) * (blockDim.x / E) * 2 * gap;
        unsigned int global_offset = (k * coeff_modulus_size + j) << log_degree;

        __shared__ uint64_t sdata[NTT_KERNEL_THREAD_COUNT * 2];
        unsigned int from_x_index = 
            threadIdx.x / E * 2 * gap 
            + threadIdx.x % E * stride 
            + coefficient_offset 
            + global_offset;

        unsigned int from_y_index = from_x_index + gap;
        sdata[threadIdx.x] = operand[from_x_index];
        sdata[threadIdx.x + blockDim.x] = operand[from_y_index];

        const MultiplyUint64Operand* r_ptr = use_inv_root_powers ?
            tables[j].inv_root_powers().raw_pointer() :
            tables[j].root_powers().raw_pointer();

        __syncthreads();

        coefficient_offset = block_idx % stride + (block_idx / stride) * (blockDim.x / E) * gap;

        for (unsigned int layer = layer_lower; layer < layer_upper; layer++) {

            unsigned int i = threadIdx.x / E * gap + threadIdx.x % E * stride + coefficient_offset;
            unsigned int rid = (1 << layer) + (i >> gap_power);
            
            MultiplyUint64Operand r = r_ptr[rid];

            unsigned int sgap = blockDim.x >> (layer - layer_lower);
            unsigned int x_index = threadIdx.x / sgap * 2 * sgap + threadIdx.x % sgap; // wrt shared data
            unsigned int y_index = x_index + sgap; 
            
            uint64_t x = sdata[x_index];
            uint64_t y = sdata[y_index];
            uint64_t u = (x >= two_times_modulus) ? (x - two_times_modulus) : x;
            uint64_t v = utils::multiply_uint64operand_mod_lazy(y, r, modulus);
            x = u + v;
            y = u + two_times_modulus - v;

            sdata[x_index] = x;
            sdata[y_index] = y;

            __syncthreads();

            E >>= 1;
            gap >>= 1;
            gap_power -= 1;

        }
        
        operand[from_x_index] = sdata[threadIdx.x];
        operand[from_y_index] = sdata[threadIdx.x + blockDim.x];

    }

    void ntt_transfer_to_rev(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        bool device = operand.on_device();
        // same device
        if (device != tables.on_device()) {
            throw std::invalid_argument("[ntt_transfer_to_rev] Operand and tables must be on the same device.");
        }
        if (!device) {
            for (size_t layer = 0; layer < log_degree; layer++) {
                host_ntt_transfer_to_rev_layer(layer, operand, pcount, log_degree, tables, use_inv_root_powers);
            }
        } else {
            if (log_degree <= NTT_KERNEL_THREAD_COUNT_LOG2) {
                size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                size_t thread_count = 1 << (log_degree - 1);
                size_t block_count = ceil_div<size_t>(total, thread_count);
                kernel_ntt_transfer_to_rev_layers<<<block_count, thread_count>>>(
                    0, log_degree, operand, pcount, log_degree, tables, use_inv_root_powers
                );
            } else {
                for (size_t layer_lower = 0; layer_lower < log_degree; layer_lower += NTT_KERNEL_THREAD_COUNT_LOG2) {
                    size_t layer_upper = std::min(layer_lower + NTT_KERNEL_THREAD_COUNT_LOG2, log_degree);
                    size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                    size_t block_count = ceil_div<size_t>(total, NTT_KERNEL_THREAD_COUNT);
                    kernel_ntt_transfer_to_rev_layers<<<block_count, NTT_KERNEL_THREAD_COUNT>>>(
                        layer_lower, layer_upper, operand, pcount, log_degree, tables, use_inv_root_powers
                    );
                }
            }
        }
    }

    void host_ntt_transfer_from_rev_layer(size_t layer, Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
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
                    operand[x_index] = (u + v > two_times_modulus) ? (u + v - two_times_modulus) : (u + v);
                    operand[y_index] = utils::multiply_uint64operand_mod_lazy(u + two_times_modulus - v, r, modulus);
                }
            }
        }
    }

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

    void ntt_transfer_from_rev(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        bool device = operand.on_device();
        // same device
        if (device != tables.on_device()) {
            throw std::invalid_argument("[ntt_transfer_from_rev] Operand and tables must be on the same device.");
        }
        if (!device) {
            for (size_t layer = 0; layer < log_degree; layer++) {
                host_ntt_transfer_from_rev_layer(layer, operand, pcount, log_degree, tables, use_inv_root_powers);
            }
        } else {
            if (log_degree <= NTT_KERNEL_THREAD_COUNT_LOG2) {
                size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                size_t thread_count = 1 << (log_degree - 1);
                size_t block_count = ceil_div<size_t>(total, thread_count);
                kernel_ntt_transfer_from_rev_layers<<<block_count, thread_count>>>(
                    0, log_degree, operand, pcount, log_degree, tables, use_inv_root_powers
                );
            } else {
                for (size_t layer_lower = 0; layer_lower < log_degree; layer_lower += NTT_KERNEL_THREAD_COUNT_LOG2) {
                    size_t layer_upper = std::min(layer_lower + NTT_KERNEL_THREAD_COUNT_LOG2, log_degree);
                    size_t total = pcount * tables.size() * (1 << (log_degree - 1));
                    size_t block_count = ceil_div<size_t>(total, NTT_KERNEL_THREAD_COUNT);
                    kernel_ntt_transfer_from_rev_layers<<<block_count, NTT_KERNEL_THREAD_COUNT>>>(
                        layer_lower, layer_upper, operand, pcount, log_degree, tables, use_inv_root_powers
                    );
                }
            }
        }
        ntt_multiply_inv_degree(
            operand, pcount, log_degree, tables
        );
    }

    void host_ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        size_t degree = static_cast<size_t>(1) << log_degree; 
        for (size_t j = 0; j < tables.size(); j++) {
            uint64_t modulus = tables[j].modulus().value();
            uint64_t two_times_modulus = modulus << 1;
            for (size_t k = 0; k < pcount; k++) {
                for (size_t i = 0; i < degree; i++) {
                    size_t x_index = ((k * tables.size() + j) << log_degree) + i;
                    uint64_t x = operand[x_index];
                    if (x >= two_times_modulus) x -= two_times_modulus;
                    if (x >= modulus) x -= modulus;
                    operand[x_index] = x;
                }
            }
        }
    }

    __global__ void kernel_ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t degree = static_cast<size_t>(1) << log_degree;
        size_t total = pcount * tables.size() * degree;
        if (global_index < total) {
            // size_t k = global_index / (tables.size() * degree);
            size_t j = (global_index / degree) % tables.size();
            // size_t i = global_index % degree;
            uint64_t x = operand[global_index];
            uint64_t modulus = tables[j].modulus().value();
            uint64_t two_times_modulus = modulus << 1;
            if (x >= two_times_modulus) x -= two_times_modulus;
            if (x >= modulus) x -= modulus;
            operand[global_index] = x;
        }
    }

    void ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        bool device = operand.on_device();
        // same device=
        if (device != tables.on_device()) {
            throw std::invalid_argument("[ntt_transfer_last_reduce] Operand and tables must be on the same device.");
        }
        if (device) {
            size_t total = (pcount * tables.size()) << log_degree;
            size_t block_count = ceil_div<size_t>(total, KERNEL_THREAD_COUNT);
            kernel_ntt_transfer_last_reduce<<<block_count, KERNEL_THREAD_COUNT>>>(operand, pcount, log_degree, tables);
        } else {
            host_ntt_transfer_last_reduce(operand, pcount, log_degree, tables);
        }
    }

}}