#pragma once
#include <shared_mutex>
#include <mutex>
#include <vector>
#include "box.h"
#include "basics.h"
#include "box_batch.h"
#include "number_theory.h"

namespace troy {namespace utils {

    const size_t GALOIS_GENERATOR = 3;

    class GaloisTool {

        bool device;
        size_t coeff_count_power_;
        size_t coeff_count_;
        
        mutable std::shared_mutex permutation_tables_rwlock;
        mutable std::vector<Array<size_t>> permutation_tables;
        mutable std::vector<bool> initialized;

        static Array<size_t> generate_table_ntt(size_t coeff_count_power, size_t galois_element);

        void ensure_permutation_table(size_t galois_element, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

    public:

        GaloisTool(const GaloisTool& other) = delete;
        GaloisTool(GaloisTool&& other) {
            this->device = other.device;
            this->coeff_count_power_ = other.coeff_count_power_;
            this->coeff_count_ = other.coeff_count_;
            this->permutation_tables = std::move(other.permutation_tables);
            this->initialized = std::move(other.initialized);
        }

        inline GaloisTool& operator=(const GaloisTool& other) = delete;
        inline GaloisTool& operator=(GaloisTool&& other) {
            this->device = other.device;
            this->coeff_count_power_ = other.coeff_count_power_;
            this->coeff_count_ = other.coeff_count_;
            // lock the other
            std::unique_lock<std::shared_mutex> lock(other.permutation_tables_rwlock);
            this->permutation_tables = std::move(other.permutation_tables);
            this->initialized = std::move(other.initialized);
            // unlock the other
            lock.unlock();
            return *this;
        }

        inline bool on_device() const noexcept { return device; }
        inline size_t coeff_count_power() const noexcept { return coeff_count_power_; }
        inline size_t coeff_count() const noexcept { return coeff_count_; }

        inline GaloisTool clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            GaloisTool cloned;
            cloned.device = this->device;
            cloned.coeff_count_power_ = this->coeff_count_power_;
            cloned.coeff_count_ = this->coeff_count_;
            cloned.permutation_tables = std::vector<Array<size_t>>(); permutation_tables.reserve(this->permutation_tables.size());
            for (size_t i = 0; i < this->permutation_tables.size(); i++) {
                cloned.permutation_tables.push_back(this->permutation_tables[i].clone(pool));
            }
            cloned.initialized = this->initialized;
            return cloned;
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            std::unique_lock<std::shared_mutex> lock(permutation_tables_rwlock);
            if (device) {
                return;
            }
            device = true;
            for (size_t i = 0; i < permutation_tables.size(); i++) {
                permutation_tables[i].to_device_inplace(pool);
            }
        }

        inline GaloisTool to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            GaloisTool result = this->clone(pool);
            result.to_device_inplace(pool);
            return result;
        }
    
        inline GaloisTool() : device(false) {}
        
        GaloisTool(size_t coeff_count_power);

        size_t get_element_from_step(int step) const;
        inline std::vector<size_t> get_elements_from_steps(const std::vector<int>& steps) const {
            std::vector<size_t> result; result.reserve(steps.size());
            for (auto step : steps) {
                result.push_back(get_element_from_step(step));
            }
            return result;
        }

        std::vector<size_t> get_elements_all() const;

        /**
        Compute the index in the range of 0 to (coeff_count_ - 1) of a given Galois element.
        */
        inline static size_t get_index_from_element(size_t galois_element) {
            if ((galois_element & 1) == 0) {
                throw std::invalid_argument("[GaloisTool::get_index_from_element] galois_element must be odd");
            }
            return (galois_element - 1) >> 1;
        }

        void apply_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) const;
        void apply_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t galois_element, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        inline void apply_p(ConstSlice<uint64_t> poly, size_t galois_element, ConstSlice<Modulus> moduli, Slice<uint64_t> result) const {
            apply_ps(poly, 1, galois_element, moduli, result);
        }
        inline void apply_bp(const ConstSliceVec<uint64_t>& poly, size_t galois_element, ConstSlice<Modulus> moduli, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            apply_bps(poly, 1, galois_element, moduli, result, pool);
        }

        inline void apply(ConstSlice<uint64_t> component, size_t galois_element, ConstPointer<Modulus> modulus, Slice<uint64_t> result) const {
            apply_ps(component, 1, galois_element, ConstSlice<Modulus>::from_pointer(modulus), result);
        }
        inline void apply_b(const ConstSliceVec<uint64_t>& component, size_t galois_element, ConstPointer<Modulus> modulus, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            apply_bps(component, 1, galois_element, ConstSlice<Modulus>::from_pointer(modulus), result, pool);
        }

        void apply_ntt_ps(ConstSlice<uint64_t> polys, size_t pcount, size_t coeff_modulus_size, size_t galois_element, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        void apply_ntt_bps(const ConstSliceVec<uint64_t>& polys, size_t pcount, size_t coeff_modulus_size, size_t galois_element, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        inline void apply_ntt_p(ConstSlice<uint64_t> poly, size_t coeff_modulus_size, size_t galois_element, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            apply_ntt_ps(poly, 1, coeff_modulus_size, galois_element, result, pool);
        }
        inline void apply_ntt_bp(const ConstSliceVec<uint64_t>& poly, size_t coeff_modulus_size, size_t galois_element, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            apply_ntt_bps(poly, 1, coeff_modulus_size, galois_element, result, pool);
        }

        inline void apply_ntt(ConstSlice<uint64_t> component, size_t galois_element, Slice<uint64_t> result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            apply_ntt_ps(component, 1, 1, galois_element, result, pool);
        }
        inline void apply_ntt_b(const ConstSliceVec<uint64_t>& component, size_t galois_element, const SliceVec<uint64_t>& result, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            apply_ntt_bps(component, 1, 1, galois_element, result, pool);
        }

        

    };

}}