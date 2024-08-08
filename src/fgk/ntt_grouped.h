#pragma once

#include "../utils/ntt.h"

namespace troy::utils::fgk::ntt_grouped {

    void ntt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables);

    void intt(ConstSlice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, Slice<uint64_t> result, NTTTableIndexer tables);

    inline void ntt_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables) {
        ntt(operand.as_const(), pcount, component_count, log_degree, use_inv_root_powers, operand, tables);
    }

    inline void intt_inplace(Slice<uint64_t> operand, size_t pcount, size_t component_count, size_t log_degree, bool use_inv_root_powers, NTTTableIndexer tables) {
        intt(operand.as_const(), pcount, component_count, log_degree, use_inv_root_powers, operand, tables);
    }

}