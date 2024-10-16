#include "box_batch.h"
#include "constants.h"
#include "ntt.h"

#include <cassert>
#include "timer.h"
#include "../fgk/ntt_grouped.h"
#include "../fgk/ntt_cooperative.h"

namespace troy {namespace utils {

    static constexpr bool NTT_USE_COOPERATIVE = false;

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
        
        Array<MultiplyUint64Operand> root_powers(coeff_count, false, nullptr);
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

        Array<MultiplyUint64Operand> inv_root_powers(coeff_count, false, nullptr);
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

    void host_ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables, uint64_t scalar) {
        size_t degree = static_cast<size_t>(1) << log_degree; 
        for (size_t j = 0; j < component_count; j++) {
            for (size_t k = 0; k < pcount; k++) {
                const Modulus& modulus = tables.get(k, j).modulus();
                MultiplyUint64Operand invd = tables.get(k, j).inv_degree_modulo();
                for (size_t i = 0; i < degree; i++) {
                    size_t x_index = ((k * component_count + j) << log_degree) + i;
                    uint64_t result = multiply_uint64operand_mod_lazy(operand[x_index], invd, modulus);
                    operand[x_index] = multiply_uint64_mod(result, scalar, modulus);
                }
            }
        }
    }

    __device__ void device_ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables, uint64_t scalar) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t degree = static_cast<size_t>(1) << log_degree;
        size_t total = pcount * component_count * degree;
        if (global_index < total) {
            size_t k = global_index / (component_count * degree);
            size_t j = (global_index / degree) % component_count;
            // size_t i = global_index % degree;
            const NTTTables& table = tables.get(k, j);
            const Modulus& modulus = table.modulus();
            MultiplyUint64Operand invd = table.inv_degree_modulo();
            uint64_t result = multiply_uint64operand_mod_lazy(operand[global_index], invd, modulus);
            operand[global_index] = multiply_uint64_mod(result, scalar, modulus);
        }
    }

    __global__ void kernel_ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables, uint64_t scalar) {
        device_ntt_multiply_inv_degree(operand, pcount, component_count, log_degree, tables, scalar);
    }

    __global__ void kernel_ntt_multiply_inv_degree_batched(SliceArrayRef<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables, uint64_t scalar) {
        for (size_t i = 0; i < operand.size(); i++) {
            device_ntt_multiply_inv_degree(operand[i], pcount, component_count, log_degree, tables, scalar);
        }
    }
    

    void ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables, uint64_t scalar) {
        bool device = operand.on_device();
        if (!device_compatible(operand, tables)) {
            throw std::invalid_argument("[ntt_multiply_inv_degree] Operand and tables must be on the same device.");
        }
        if (device) {
            size_t total = (pcount * component_count) << log_degree;
            size_t block_count = ceil_div<size_t>(total, KERNEL_THREAD_COUNT);
            utils::set_device(operand.device_index());
            kernel_ntt_multiply_inv_degree<<<block_count, KERNEL_THREAD_COUNT>>>(operand, pcount, component_count, log_degree, tables, scalar);
            utils::stream_sync();
        } else {
            host_ntt_multiply_inv_degree(operand, pcount, component_count, log_degree, tables, scalar);
        }
    }
    
    void ntt_multiply_inv_degree_batched(const SliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables, uint64_t scalar, MemoryPoolHandle pool) {
        if (!tables.on_device() || operand.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < operand.size(); i++) {
                ntt_multiply_inv_degree(operand[i], pcount, component_count, log_degree, tables, scalar);
            }
        } else {
            size_t total = (pcount * component_count) << log_degree;
            size_t block_count = ceil_div<size_t>(total, KERNEL_THREAD_COUNT);
            utils::set_device(tables.device_index());
            auto operand_batched = construct_batch(operand, pool, operand[0]);
            kernel_ntt_multiply_inv_degree_batched<<<block_count, KERNEL_THREAD_COUNT>>>(operand_batched, pcount, component_count, log_degree, tables, scalar);
            utils::stream_sync();
        }
    }

    void ntt_transfer_to_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::ntt_inplace(operand, pcount, component_count, log_degree, use_inv_root_powers, tables);
        else fgk::ntt_grouped::ntt_inplace(operand, pcount, component_count, log_degree, use_inv_root_powers, tables);
    }

    void ntt_transfer_from_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::intt_inplace(operand, pcount, component_count, log_degree, use_inv_root_powers, tables);
        else fgk::ntt_grouped::intt_inplace(operand, pcount, component_count, log_degree, use_inv_root_powers, tables);
    }

    void ntt_transfer_to_rev(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::ntt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
        else fgk::ntt_grouped::ntt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
    }

    void ntt_transfer_from_rev(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::intt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
        else fgk::ntt_grouped::intt(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables);
    }

    
    void ntt_transfer_to_rev_inplace_batched(const SliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables, MemoryPoolHandle pool) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::ntt_inplace_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, tables, pool);
        else fgk::ntt_grouped::ntt_inplace_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, tables, pool);
    }

    void ntt_transfer_from_rev_inplace_batched(const SliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables, MemoryPoolHandle pool) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::intt_inplace_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, tables, pool);
        else fgk::ntt_grouped::intt_inplace_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, tables, pool);
    }

    void ntt_transfer_to_rev_batched(const ConstSliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, const SliceVec<uint64_t>& result, NTTTableIndexer tables, MemoryPoolHandle pool) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::ntt_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables, pool);
        else fgk::ntt_grouped::ntt_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables, pool);
    }

    void ntt_transfer_from_rev_batched(const ConstSliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, const SliceVec<uint64_t>& result, NTTTableIndexer tables, MemoryPoolHandle pool) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::intt_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables, pool);
        else fgk::ntt_grouped::intt_batched(operand, pcount, component_count, log_degree, use_inv_root_powers, result, tables, pool);
    }

    void host_ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables) {
        size_t degree = static_cast<size_t>(1) << log_degree; 
        for (size_t j = 0; j < component_count; j++) {
            for (size_t k = 0; k < pcount; k++) {
                const NTTTables& table = tables.get(k, j);
                uint64_t modulus = table.modulus().value();
                uint64_t two_times_modulus = modulus << 1;
                for (size_t i = 0; i < degree; i++) {
                    size_t x_index = ((k * component_count + j) << log_degree) + i;
                    uint64_t x = operand[x_index];
                    if (x >= two_times_modulus) x -= two_times_modulus;
                    if (x >= modulus) x -= modulus;
                    operand[x_index] = x;
                }
            }
        }
    }

    __global__ void kernel_ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t degree = static_cast<size_t>(1) << log_degree;
        size_t total = pcount * component_count* degree;
        if (global_index < total) {
            size_t k = global_index / (component_count * degree);
            size_t j = (global_index / degree) % component_count;
            // size_t i = global_index % degree;
            uint64_t x = operand[global_index];
            uint64_t modulus = tables.get(k, j).modulus().value();
            uint64_t two_times_modulus = modulus << 1;
            if (x >= two_times_modulus) x -= two_times_modulus;
            if (x >= modulus) x -= modulus;
            operand[global_index] = x;
        }
    }

    void ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables) {
        bool device = operand.on_device();
        if (!device_compatible(operand, tables)) {
            throw std::invalid_argument("[ntt_transfer_last_reduce] Operand and tables must be on the same device.");
        }
        if (device) {
            size_t total = (pcount * component_count) << log_degree;
            size_t block_count = ceil_div<size_t>(total, KERNEL_THREAD_COUNT);
            kernel_ntt_transfer_last_reduce<<<block_count, KERNEL_THREAD_COUNT>>>(operand, pcount, component_count, log_degree, tables);
            utils::stream_sync();
        } else {
            host_ntt_transfer_last_reduce(operand, pcount, component_count, log_degree, tables);
        }
    }

}}