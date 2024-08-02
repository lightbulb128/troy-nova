#pragma once

#include "../utils/ntt.h"

namespace troy::utils::fgk::ntt_cooperative {

    void ntt(ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result);

    void intt(ConstSlice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers, Slice<uint64_t> result);

    inline void ntt_inplace(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        ntt(operand.as_const(), pcount, log_degree, tables, use_inv_root_powers, operand);
    }

    inline void intt_inplace(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers) {
        intt(operand.as_const(), pcount, log_degree, tables, use_inv_root_powers, operand);
    }
    
}