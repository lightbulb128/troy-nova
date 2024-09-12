#pragma once

#include "../utils/ntt.h"
#include "../utils/box_batch.h"
#include "../batch_utils.h"

namespace troy::utils::fgk::ntt_grouped {

    void ntt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables);
    void ntt_batched(
        const ConstSliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, 
        const SliceVec<uint64_t>& result, NTTTableIndexer tables,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );

    void intt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables);
    void intt_batched(
        const ConstSliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, 
        const SliceVec<uint64_t>& result, NTTTableIndexer tables,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );

    inline void ntt_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables) {
        ntt(operand.as_const(), pcount, component_count, log_degree, use_inv_root_powers, operand, tables);
    }
    inline void ntt_inplace_batched(
        const SliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        ntt_batched(batch_utils::rcollect_as_const(operand), pcount, component_count, log_degree, use_inv_root_powers, operand, tables, pool);
    }

    inline void intt_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables) {
        intt(operand.as_const(), pcount, component_count, log_degree, use_inv_root_powers, operand, tables);
    }
    inline void intt_inplace_batched(
        const SliceVec<uint64_t>& operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    ) {
        intt_batched(batch_utils::rcollect_as_const(operand), pcount, component_count, log_degree, use_inv_root_powers, operand, tables, pool);
    }

}