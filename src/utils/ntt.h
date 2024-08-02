#pragma once
#include <cstdint>
#include "basics.h"
#include "../modulus.h"
#include "uint_small_mod.h"
#include "number_theory.h"

namespace troy {namespace utils {

    class NTTTables {

        uint64_t root_;
        size_t coeff_count_power_;
        size_t coeff_count_;
        Modulus modulus_;
        MultiplyUint64Operand inv_degree_modulo_;
        Array<MultiplyUint64Operand> root_powers_;
        Array<MultiplyUint64Operand> inv_root_powers_;
        bool device;

    public:

        inline MemoryPoolHandle pool() const { return root_powers_.pool(); }
        inline size_t device_index() const { return root_powers_.device_index(); }

        NTTTables(): device(false) {}

        NTTTables(size_t coeff_count_power, const Modulus& modulus);

        inline NTTTables clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            NTTTables copied;
            copied.root_ = root_;
            copied.coeff_count_power_ = coeff_count_power_;
            copied.coeff_count_ = coeff_count_;
            copied.modulus_ = modulus_;
            copied.inv_degree_modulo_ = inv_degree_modulo_;
            copied.root_powers_ = root_powers_.clone(pool);
            copied.inv_root_powers_ = inv_root_powers_.clone(pool);
            return copied;
        }

        __host__ __device__ bool on_device() const {
            return device;
        }

        __host__ __device__ uint64_t root() const { return root_; }
        __host__ __device__ ConstSlice<MultiplyUint64Operand> root_powers() const {
            return root_powers_.detached_const_reference();
        }
        __host__ __device__ ConstSlice<MultiplyUint64Operand> inv_root_powers() const {
            return inv_root_powers_.detached_const_reference();
        }
        __host__ __device__ MultiplyUint64Operand inv_degree_modulo() const {
            return inv_degree_modulo_;
        }
        __host__ __device__ const Modulus& modulus() const { return modulus_; }
        __host__ __device__ size_t coeff_count_power() const { return coeff_count_power_; }
        __host__ __device__ size_t coeff_count() const { return coeff_count_; }

        /* This function moves all the arrays in the struct into device. */
        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            root_powers_.to_device_inplace(pool);
            inv_root_powers_.to_device_inplace(pool);
            device = true;
        }

        /* This function moves all the arrays in the struct into device. */
        inline NTTTables to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            NTTTables tables = this->clone(pool);
            tables.to_device_inplace(pool);
            return tables;
        }

        inline static Array<NTTTables> create_ntt_tables(size_t coeff_count_power, ConstSlice<Modulus> moduli) {
            if (moduli.size() == 0) {
                return Array<NTTTables>(0, false, nullptr);
            }
            Array<NTTTables> tables(moduli.size(), false, nullptr);
            for (size_t i = 0; i < moduli.size(); i++) {
                tables[i] = NTTTables(coeff_count_power, moduli[i]);
            }
            return tables;
        }

    };

    void ntt_transfer_to_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers);

    void ntt_transfer_from_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers);

    void ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables);

    void ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables);

    inline void ntt_negacyclic_harvey_lazy_ps(Slice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[ntt_negacyclic_harvey_lazy_ps] degree is invalid");
        }
        size_t log_degree = static_cast<size_t>(logd);
        ntt_transfer_to_rev_inplace(operand, pcount, log_degree, tables, false);
    }

    inline void ntt_negacyclic_harvey_ps(Slice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[ntt_negacyclic_harvey_ps] degree is invalid");
        }
        // size_t log_degree = static_cast<size_t>(logd);
        ntt_negacyclic_harvey_lazy_ps(operand, pcount, degree, tables);
        // ntt_transfer_last_reduce(operand, pcount, log_degree, tables);
    }

    inline void inverse_ntt_negacyclic_harvey_lazy_ps(Slice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[inverse_ntt_negacyclic_harvey_lazy_ps] degree is invalid");
        }
        size_t log_degree = static_cast<size_t>(logd);
        ntt_transfer_from_rev_inplace(operand, pcount, log_degree, tables, true);
    }

    inline void inverse_ntt_negacyclic_harvey_ps(Slice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[inverse_ntt_negacyclic_harvey_ps] degree is invalid");
        }
        // size_t log_degree = static_cast<size_t>(logd);
        inverse_ntt_negacyclic_harvey_lazy_ps(operand, pcount, degree, tables);
        // ntt_transfer_last_reduce(operand, pcount, log_degree, tables);
    }

    inline void ntt_negacyclic_harvey_lazy_p(Slice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables) {
        ntt_negacyclic_harvey_lazy_ps(operand, 1, degree, tables);
    }

    inline void ntt_negacyclic_harvey_p(Slice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables) {
        ntt_negacyclic_harvey_ps(operand, 1, degree, tables);
    }

    inline void inverse_ntt_negacyclic_harvey_lazy_p(Slice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables) {
        inverse_ntt_negacyclic_harvey_lazy_ps(operand, 1, degree, tables);
    }

    inline void inverse_ntt_negacyclic_harvey_p(Slice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables) {
        inverse_ntt_negacyclic_harvey_ps(operand, 1, degree, tables);
    }

    inline void ntt_negacyclic_harvey_lazy(Slice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables) {
        ntt_negacyclic_harvey_lazy_ps(operand, 1, degree, ConstSlice<NTTTables>::from_pointer(tables));
    }

    inline void ntt_negacyclic_harvey(Slice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables) {
        ntt_negacyclic_harvey_ps(operand, 1, degree, ConstSlice<NTTTables>::from_pointer(tables));
    }

    inline void inverse_ntt_negacyclic_harvey_lazy(Slice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables) {
        inverse_ntt_negacyclic_harvey_lazy_ps(operand, 1, degree, ConstSlice<NTTTables>::from_pointer(tables));
    }

    inline void inverse_ntt_negacyclic_harvey(Slice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables) {
        inverse_ntt_negacyclic_harvey_ps(operand, 1, degree, ConstSlice<NTTTables>::from_pointer(tables));
    }

}}
