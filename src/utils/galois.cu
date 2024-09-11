#include "box_batch.h"
#include "constants.h"
#include "galois.h"
#include "memory_pool.h"
#include <thread>

namespace troy {namespace utils {

    GaloisTool::GaloisTool(size_t coeff_count_power) {
        size_t coeff_count = 1 << coeff_count_power;
        if (coeff_count > utils::HE_POLY_MOD_DEGREE_MAX || coeff_count < utils::HE_POLY_MOD_DEGREE_MIN) {
            throw std::invalid_argument("[GaloisTool::GaloisTool] coeff_count_power is invalid");
        }
        this->coeff_count_ = coeff_count;
        this->coeff_count_power_ = coeff_count_power;
        this->device = false;
        this->permutation_tables = std::vector<Array<size_t>>();
        this->permutation_tables.reserve(coeff_count);
        this->initialized = std::vector<bool>(coeff_count, false);
        for (size_t i = 0; i < coeff_count; i++) {
            this->permutation_tables.push_back(Array<size_t>());
        }
    }

    Array<size_t> GaloisTool::generate_table_ntt(size_t coeff_count_power, size_t galois_element) {
        size_t coeff_count = 1 << coeff_count_power;
        Array<size_t> result(coeff_count, false, nullptr);
        size_t mask = coeff_count - 1;
        for (size_t i = 0; i < coeff_count; i++) {
            uint32_t reversed = utils::reverse_bits_uint32(
                static_cast<uint32_t>(i + coeff_count),
                coeff_count_power + 1
            );
            uint64_t index_raw = ((static_cast<uint64_t>(galois_element) * static_cast<uint64_t>(reversed)) >> 1)
                & static_cast<uint64_t>(mask);
            result[i] = utils::reverse_bits_uint32(
                static_cast<uint32_t>(index_raw),
                coeff_count_power
            );
        }
        return result;
    }
    
    size_t GaloisTool::get_element_from_step(int step) const {
        size_t n = this->coeff_count();
        size_t m = n * 2; 
        if (step == 0) return m - 1;
        else {
            // Extract sign of steps. When steps is positive, the rotation
            // is to the left; when steps is negative, it is to the right.
            bool sign = step < 0;
            size_t pos_step = (step < 0) ? -step : step;
            if (pos_step >= (n >> 1)) {
                throw std::invalid_argument("[GaloisTool::get_element_from_step] Step count too large");
            }
            size_t true_step = sign ? ((n >> 1) - pos_step) : pos_step;
            size_t gen = GALOIS_GENERATOR; size_t galois_element = 1;
            for (size_t i = 0; i < true_step; i++) {
                galois_element = (galois_element * gen) & (m - 1);
            }
            return galois_element;
        }
    }
    
    std::vector<size_t> GaloisTool::get_elements_all() const {
        size_t n = this->coeff_count();
        size_t m = n * 2; 
        // Generate Galois keys for m - 1 (X -> X^{m-1})
        std::vector<size_t> galois_elements = { m - 1 };
        galois_elements.reserve(this->coeff_count_power() * 2 - 1);
        
        // Generate Galois key for power of generator_ mod m (X -> X^{3^k}) and
        // for negative power of generator_ mod m (X -> X^{-3^k})
        size_t pos_power = GALOIS_GENERATOR;
        uint64_t neg_power_uint64 = 0;
        utils::try_invert_uint64_mod_uint64(
            static_cast<uint64_t>(pos_power),
            static_cast<uint64_t>(m),
            neg_power_uint64
        );
        size_t neg_power = static_cast<size_t>(neg_power_uint64);

        for (size_t i = 0; i < this->coeff_count_power() - 1; i++) {
            galois_elements.push_back(pos_power);
            galois_elements.push_back(neg_power);
            pos_power = (pos_power * pos_power) & (m - 1);
            neg_power = (neg_power * neg_power) & (m - 1);
        }
        return galois_elements;
    }
    
    void GaloisTool::ensure_permutation_table(size_t galois_element, MemoryPoolHandle pool) const {
        size_t index = GaloisTool::get_index_from_element(galois_element);

        // acquire lock
        std::shared_lock<std::shared_mutex> read_lock(this->permutation_tables_rwlock);

        // is empty?
        if (this->initialized[index]) {
            // std::cerr << "[" << std::this_thread::get_id() << "] (" << index << ") read_lock check return\n";
            return;
        }

        read_lock.unlock();
        std::this_thread::yield();
        
        // std::cerr << "[" << std::this_thread::get_id() << "] (" << index << ") read_lock check proceed\n";

        // acquire lock
        std::unique_lock<std::shared_mutex> write_lock(this->permutation_tables_rwlock);

        // is empty?
        if (this->initialized[index]) {
            // std::cerr << "[" << std::this_thread::get_id() << "] (" << index << ") write_lock check return\n";
            return;
        } else {
            // std::cerr << "[" << std::this_thread::get_id() << "] (" << index << ") write_lock check proceed\n";
        }

        this->permutation_tables[index] = GaloisTool::generate_table_ntt(
            this->coeff_count_power(),
            galois_element
        );
        if (this->on_device()) {
            this->permutation_tables[index].to_device_inplace(pool);
            pool->set_device();
            cudaDeviceSynchronize();
        }

        this->initialized[index] = true;
        // std::cerr << "[" << std::this_thread::get_id() << "] (" << index << ") write_lock done\n";
    }

    static void host_apply_ps(const GaloisTool& self, ConstSlice<uint64_t> polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t mask = self.coeff_count() - 1;
        for (size_t k = 0; k < pcount; k++) {
            for (size_t j = 0; j < moduli.size(); j++) {
                for (size_t i = 0; i < self.coeff_count(); i++) {
                    size_t index_raw = i * galois_element;
                    size_t index = index_raw & mask;
                    size_t input_index = (k * moduli.size() + j) * self.coeff_count() + i;
                    size_t result_index = (k * moduli.size() + j) * self.coeff_count() + index;
                    uint64_t input = input_index >= polys.size() ? 0 : polys[input_index];
                    result[result_index] = (((index_raw >> self.coeff_count_power()) & 1) > 0)
                        ? utils::negate_uint64_mod(input, moduli[j])
                        : input;
                }
            }
        }
    }

    __device__ static void device_apply_ps(size_t coeff_count_power, ConstSlice<uint64_t> polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t coeff_count = 1 << coeff_count_power;
        size_t moduli_count = moduli.size();
        if (global_index >= coeff_count * moduli_count * pcount) return;
        size_t k = global_index / (coeff_count * moduli_count);
        size_t j = (global_index / coeff_count) % moduli_count;
        size_t i = global_index % coeff_count;
        size_t index_raw = i * galois_element;
        size_t index = index_raw & (coeff_count - 1);
        size_t result_index = (k * moduli_count + j) * coeff_count + index;
        uint64_t input = global_index >= polys.size() ? 0 : polys[global_index];
        result[result_index] = (((index_raw >> coeff_count_power) & 1) > 0)
            ? utils::negate_uint64_mod(input, moduli[j])
            : input;
    }
    
    __global__ static void kernel_apply_ps(size_t coeff_count_power, ConstSlice<uint64_t> polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) {
        device_apply_ps(coeff_count_power, polys, pcount, galois_element, moduli, result);
    }
    __global__ static void kernel_apply_bps(size_t coeff_count_power, ConstSliceArrayRef<uint64_t> polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, SliceArrayRef<uint64_t> result) {
        size_t i = blockIdx.y;
        device_apply_ps(coeff_count_power, polys[i], pcount, galois_element, moduli, result[i]);
    }
    
    void GaloisTool::apply_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) const {
        bool device = this->on_device();
        if (!utils::device_compatible(polys, moduli, result)) {
            throw std::invalid_argument("[GaloisTool::apply_ps] Arguments are not on the same device");
        }
        if (device) {
            size_t total = this->coeff_count() * moduli.size() * pcount;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_apply_ps<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                this->coeff_count_power(),
                polys,
                pcount,
                galois_element,
                moduli,
                result
            );
            utils::stream_sync();
        } else {
            host_apply_ps(*this, polys, pcount, galois_element, moduli, result);
        }
    }

    
    void GaloisTool::apply_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) const {
        if (polys.size() != result.size()) {
            throw std::invalid_argument("[GaloisTool::apply_bps] polys and result must have the same size");
        }
        if (polys.size() == 0) return;
        bool device = this->on_device();
        if (!device || polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys.size(); i++) {
                this->apply_ps(polys[i], pcount, galois_element, moduli, result[i]);
            }
        } else {
            size_t n = polys.size();
            for (size_t i = 0; i < n; i++) {
                if (!utils::device_compatible(polys[i], moduli, result[i])) {
                    throw std::invalid_argument("[GaloisTool::apply_ps] Arguments are not on the same device");
                }
            }
            size_t total = this->coeff_count() * moduli.size() * pcount;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            dim3 block_dims(block_count, polys.size());
            utils::set_device(polys[0].device_index());
            auto polys_batched = utils::construct_batch(polys, pool, moduli);
            auto result_batched = utils::construct_batch(result, pool, moduli);
            kernel_apply_bps<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                this->coeff_count_power(),
                polys_batched,
                pcount,
                galois_element,
                moduli,
                result_batched
            );
            utils::stream_sync();
        }
    }

    static void host_apply_ntt_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t coeff_modulus_size, size_t coeff_count, Slice<uint64_t> result, ConstSlice<size_t> permutation_table) {
        for (size_t k = 0; k < pcount; k++) {
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                for (size_t i = 0; i < coeff_count; i++) {
                    size_t result_index = (k * coeff_modulus_size + j) * coeff_count + i;
                    size_t input_index = (k * coeff_modulus_size + j) * coeff_count + permutation_table[i];
                    result[result_index] = polys[input_index];
                }
            }
        }
    }

    __device__ static void device_apply_ntt_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t coeff_modulus_size, size_t coeff_count, Slice<uint64_t> result, ConstSlice<size_t> permutation_table) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= coeff_count * coeff_modulus_size * pcount) return;
        size_t k = global_index / (coeff_count * coeff_modulus_size);
        size_t j = (global_index / coeff_count) % coeff_modulus_size;
        size_t i = global_index % coeff_count;
        size_t input_index = (k * coeff_modulus_size + j) * coeff_count + permutation_table[i];
        result[global_index] = polys[input_index];
    }
    __global__ static void kernel_apply_ntt_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t coeff_modulus_size, size_t coeff_count, Slice<uint64_t> result, ConstSlice<size_t> permutation_table) {
        device_apply_ntt_ps(polys, pcount, coeff_modulus_size, coeff_count, result, permutation_table);
    }
    __global__ static void kernel_apply_ntt_bps(ConstSliceArrayRef<uint64_t> polys, size_t pcount, size_t coeff_modulus_size, size_t coeff_count, SliceArrayRef<uint64_t> result, ConstSlice<size_t> permutation_table) {
        size_t i = blockIdx.y;
        device_apply_ntt_ps(polys[i], pcount, coeff_modulus_size, coeff_count, result[i], permutation_table);
    }

    void GaloisTool::apply_ntt_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t coeff_modulus_size, size_t galois_element, Slice<uint64_t> result, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        this->ensure_permutation_table(galois_element, pool);
        // obtain read lock
        size_t index = GaloisTool::get_index_from_element(galois_element);
        std::shared_lock<std::shared_mutex> read_lock(this->permutation_tables_rwlock);
        if (!this->initialized[index]) {
            throw std::logic_error("[GaloisTool::apply_ntt_ps] Permutation table not initialized");
        }
        ConstSlice<size_t> permutation_table = this->permutation_tables[index].const_reference();
        
        if (!utils::device_compatible(permutation_table, polys, result)) {
            throw std::invalid_argument("[GaloisTool::apply_ntt_ps] Arguments are not on the same device");
        }
        if (device) {
            size_t total = this->coeff_count() * coeff_modulus_size * pcount;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            utils::set_device(result.device_index());
            kernel_apply_ntt_ps<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                polys,
                pcount,
                coeff_modulus_size,
                this->coeff_count(),
                result,
                permutation_table
            );
            utils::stream_sync();
        } else {
            host_apply_ntt_ps(
                polys, pcount, coeff_modulus_size, 
                this->coeff_count(), result,
                permutation_table
            );
        }
    }

    
    void GaloisTool::apply_ntt_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t coeff_modulus_size, size_t galois_element, const SliceVec<uint64_t>& result, MemoryPoolHandle pool) const {
        if (polys.size() != result.size()) {
            throw std::invalid_argument("[GaloisTool::apply_bps] polys and result must have the same size");
        }
        if (polys.size() == 0) return;
        bool device = this->on_device();
        if (!device || polys.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < polys.size(); i++) {
                this->apply_ntt_ps(polys[i], pcount, coeff_modulus_size, galois_element, result[i], pool);
            }
        } else {
            this->ensure_permutation_table(galois_element, pool);
            // obtain read lock
            size_t index = GaloisTool::get_index_from_element(galois_element);
            std::shared_lock<std::shared_mutex> read_lock(this->permutation_tables_rwlock);
            if (!this->initialized[index]) {
                throw std::logic_error("[GaloisTool::apply_ntt_ps] Permutation table not initialized");
            }
            ConstSlice<size_t> permutation_table = this->permutation_tables[index].const_reference();
            size_t n = polys.size();
            for (size_t i = 0; i < n; i++) {
                if (!utils::device_compatible(polys[i], permutation_table, result[i])) {
                    throw std::invalid_argument("[GaloisTool::apply_ps] Arguments are not on the same device");
                }
            }
            size_t total = this->coeff_count() * coeff_modulus_size * pcount;
            size_t block_count = utils::ceil_div(total, utils::KERNEL_THREAD_COUNT);
            dim3 block_dims(block_count, polys.size());
            utils::set_device(result[0].device_index());
            auto polys_batched = utils::construct_batch(polys, pool, permutation_table);
            auto result_batched = utils::construct_batch(result, pool, permutation_table);
            kernel_apply_ntt_bps<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                polys_batched,
                pcount,
                coeff_modulus_size,
                this->coeff_count(),
                result_batched,
                permutation_table
            );
            utils::stream_sync();
        }
}
}}