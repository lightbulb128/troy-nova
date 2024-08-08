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

    class NTTTableIndexer {
    public:
        inline bool on_device() const { return tables_.on_device(); }
        inline size_t device_index() const { return tables_.device_index(); }
        enum NTTTableIndexerType {
            Componentwise,
            KeySwitchingSetProducts,
            KeySwitchingSkipFinals
        };
        inline explicit NTTTableIndexer(ConstSlice<NTTTables> tables): type_(Componentwise), tables_(tables) {}
        inline static NTTTableIndexer key_switching_set_products(ConstSlice<NTTTables> tables, size_t decomp_modulus_size) {
            return NTTTableIndexer(NTTTableIndexerType::KeySwitchingSetProducts, decomp_modulus_size, tables);
        }
        inline static NTTTableIndexer key_switching_skip_finals(ConstSlice<NTTTables> tables, size_t decomp_modulus_size) {
            return NTTTableIndexer(NTTTableIndexerType::KeySwitchingSkipFinals, decomp_modulus_size, tables);
        }
        inline const NTTTables& __host__ __device__ get(size_t poly_index, size_t component_index) const {
            switch (type_) {
                case NTTTableIndexerType::Componentwise:
                    return tables_[component_index];
                case NTTTableIndexerType::KeySwitchingSetProducts:
                    if (poly_index == decomp_modulus_size_) {
                        return tables_[tables_.size() - 1];
                    } else {
                        return tables_[poly_index];
                    }
                case NTTTableIndexerType::KeySwitchingSkipFinals:
                    if (component_index == decomp_modulus_size_) {
                        return tables_[tables_.size() - 1];
                    } else {
                        return tables_[component_index];
                    }
                default:
                    return tables_[component_index];
            }
        }
    private:
        inline NTTTableIndexer(NTTTableIndexerType type, size_t decomp_modulus_size, ConstSlice<NTTTables> tables): type_(type), decomp_modulus_size_(decomp_modulus_size), tables_(tables) {}
        NTTTableIndexerType type_;
        size_t decomp_modulus_size_;
        ConstSlice<NTTTables> tables_;
    };

    void ntt_transfer_to_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables);
    void ntt_transfer_from_rev_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables);
    void ntt_transfer_to_rev(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables);
    void ntt_transfer_from_rev(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables);

    void ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables);
    void ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, NTTTableIndexer tables);

    inline void ntt_transfer_last_reduce(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        ntt_transfer_last_reduce(operand, pcount, tables.size(), log_degree, NTTTableIndexer(tables));
    }
    inline void ntt_multiply_inv_degree(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables) {
        ntt_multiply_inv_degree(operand, pcount, tables.size(), log_degree, NTTTableIndexer(tables));
    }

    /*
        Note: originally the implementations of ntt_lazy and ntt were different, where the lazy does not execute the last 4m to m reduction. (ntt_transfer_last_reduce)
        but for GPU this difference would make a new kernel launch which might be slower than just doing the last reduction.
    */

    inline void ntt_inplace_ps(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t degree, NTTTableIndexer tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[ntt_inplace_ps] degree is invalid");
        }
        size_t log_degree = static_cast<size_t>(logd);
        ntt_transfer_to_rev_inplace(operand, pcount, component_count, log_degree, false, tables);
    }

    inline void intt_inplace_ps(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t degree, NTTTableIndexer tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[inverse_ntt_negacyclic_harvey_lazy_ps] degree is invalid");
        }
        size_t log_degree = static_cast<size_t>(logd);
        ntt_transfer_from_rev_inplace(operand, pcount, component_count, log_degree, true, tables);
    }

    inline void ntt_inplace_p(Slice<uint64_t> operand, size_t component_count, size_t degree, NTTTableIndexer tables) {
        ntt_inplace_ps(operand, 1, component_count, degree, tables);
    }

    inline void intt_inplace_p(Slice<uint64_t> operand, size_t component_count, size_t degree, NTTTableIndexer tables) {
        intt_inplace_ps(operand, 1, component_count, degree, tables);
    }

    inline void ntt_inplace(Slice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables) {
        ConstSlice<NTTTables> table_one_slice(tables);
        NTTTableIndexer indexer(table_one_slice);
        ntt_inplace_ps(operand, 1, 1, degree, indexer);
    }

    inline void intt_inplace(Slice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables) {
        ConstSlice<NTTTables> table_one_slice(tables);
        NTTTableIndexer indexer(table_one_slice);
        intt_inplace_ps(operand, 1, 1, degree, indexer);
    }

    inline void ntt_ps(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t degree, Slice<uint64_t> result, NTTTableIndexer tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[inverse_ntt_negacyclic_harvey_lazy_ps] degree is invalid");
        }
        size_t log_degree = static_cast<size_t>(logd);
        ntt_transfer_to_rev(operand, pcount, component_count, log_degree, false, result, tables);
    }
    inline void ntt_p(ConstSlice<uint64_t> operand, size_t component_count, size_t degree, Slice<uint64_t> result, NTTTableIndexer tables) {
        ntt_ps(operand, 1, component_count, degree, result, tables);
    }
    inline void ntt(ConstSlice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables, Slice<uint64_t> result) {
        ConstSlice<NTTTables> table_one_slice(tables);
        NTTTableIndexer indexer(table_one_slice);
        ntt_ps(operand, 1, 1, degree, result, indexer);
    }
    inline void intt_ps(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t degree, Slice<uint64_t> result, NTTTableIndexer tables) {
        int logd = utils::get_power_of_two(degree);
        if (logd < 0) {
            throw std::invalid_argument("[inverse_ntt_negacyclic_harvey_lazy_ps] degree is invalid");
        }
        size_t log_degree = static_cast<size_t>(logd);
        ntt_transfer_from_rev(operand, pcount, component_count, log_degree, true, result, tables);
    }
    inline void intt_p(ConstSlice<uint64_t> operand, size_t component_count, size_t degree, Slice<uint64_t> result, NTTTableIndexer tables) {
        intt_ps(operand, 1, component_count, degree, result, tables);
    }
    inline void intt(ConstSlice<uint64_t> operand, size_t degree, ConstPointer<NTTTables> tables, Slice<uint64_t> result) {
        ConstSlice<NTTTables> table_one_slice(tables);
        NTTTableIndexer indexer(table_one_slice);
        intt_ps(operand, 1, 1, degree, result, indexer);
    }


    inline void ntt_inplace_ps(Slice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables) {
        ntt_inplace_ps(operand, pcount, tables.size(), degree, NTTTableIndexer(tables));
    }
    inline void ntt_inplace_p(Slice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables) {
        ntt_inplace_ps(operand, 1, tables.size(), degree, NTTTableIndexer(tables));
    }
    inline void intt_inplace_ps(Slice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables) {
        intt_inplace_ps(operand, pcount, tables.size(), degree, NTTTableIndexer(tables));
    }
    inline void intt_inplace_p(Slice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables) {
        intt_inplace_ps(operand, 1, tables.size(), degree, NTTTableIndexer(tables));
    }
    inline void ntt_ps(ConstSlice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables, Slice<uint64_t> result) {
        ntt_ps(operand, pcount, tables.size(), degree, result, NTTTableIndexer(tables));
    }
    inline void ntt_p(ConstSlice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables, Slice<uint64_t> result) {
        ntt_ps(operand, 1, tables.size(), degree, result, NTTTableIndexer(tables));
    }
    inline void intt_ps(ConstSlice<uint64_t> operand, size_t pcount, size_t degree, ConstSlice<NTTTables> tables, Slice<uint64_t> result) {
        intt_ps(operand, pcount, tables.size(), degree, result, NTTTableIndexer(tables));
    }
    inline void intt_p(ConstSlice<uint64_t> operand, size_t degree, ConstSlice<NTTTables> tables, Slice<uint64_t> result) {
        intt_ps(operand, 1, tables.size(), degree, result, NTTTableIndexer(tables));
    }

}}
