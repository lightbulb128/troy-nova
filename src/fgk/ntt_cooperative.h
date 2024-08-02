#pragma once

#include "../utils/ntt.h"

namespace troy::utils::fgk::ntt_cooperative {

    void ntt(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers);

    void intt(Slice<uint64_t> operand, size_t pcount, size_t log_degree, ConstSlice<NTTTables> tables, bool use_inv_root_powers);

}