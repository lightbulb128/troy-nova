#include "ntt.h"

#include <cassert>
#include "timer.h"
#include "../fgk/ntt_grouped.h"
#include "../fgk/ntt_cooperative.h"

namespace troy {namespace utils {

    static constexpr bool NTT_USE_COOPERATIVE = true;

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
        if (!device_compatible(operand, tables)) {
            throw std::invalid_argument("[ntt_multiply_inv_degree] Operand and tables must be on the same device.");
        }
        if (device) {
            size_t total = (pcount * tables.size()) << log_degree;
            size_t block_count = ceil_div<size_t>(total, KERNEL_THREAD_COUNT);
            utils::set_device(operand.device_index());
            kernel_ntt_multiply_inv_degree<<<block_count, KERNEL_THREAD_COUNT>>>(operand, pcount, log_degree, tables);
            utils::stream_sync();
        } else {
            host_ntt_multiply_inv_degree(operand, pcount, log_degree, tables);
        }
    }

    void ntt_transfer_to_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::ntt(operand, pcount, log_degree, tables, use_inv_root_powers);
        else fgk::ntt_grouped::ntt_inplace(operand, pcount, log_degree, tables, use_inv_root_powers);
    }

    void ntt_transfer_from_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        if constexpr (NTT_USE_COOPERATIVE) fgk::ntt_cooperative::intt(operand, pcount, log_degree, tables, use_inv_root_powers);
        else fgk::ntt_grouped::intt_inplace(operand, pcount, log_degree, tables, use_inv_root_powers);
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
        if (!device_compatible(operand, tables)) {
            throw std::invalid_argument("[ntt_transfer_last_reduce] Operand and tables must be on the same device.");
        }
        if (device) {
            size_t total = (pcount * tables.size()) << log_degree;
            size_t block_count = ceil_div<size_t>(total, KERNEL_THREAD_COUNT);
            kernel_ntt_transfer_last_reduce<<<block_count, KERNEL_THREAD_COUNT>>>(operand, pcount, log_degree, tables);
            utils::stream_sync();
        } else {
            host_ntt_transfer_last_reduce(operand, pcount, log_degree, tables);
        }
    }

}}