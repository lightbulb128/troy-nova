#pragma once

#include "../utils/rns_tool.h"

namespace troy::utils::fgk::rns_tool {

    void fast_b_conv_m_tilde_sm_mrq(
        ConstSlice<uint64_t> input,
        size_t coeff_count,
        uint64_t m_tilde_value,
        ConstSlice<Modulus> base_q,

        const BaseConverter& base_q_to_Bsk_conv,
        const BaseConverter& base_q_to_m_tilde_conv,
        
        MultiplyUint64Operand neg_inv_prod_q_mod_m_tilde,
        ConstSlice<uint64_t> prod_q_mod_Bsk,
        ConstSlice<MultiplyUint64Operand> inv_m_tilde_mod_Bsk,

        Slice<uint64_t> destination,
        MemoryPoolHandle pool
    );

    void fast_floor_fast_b_conv_sk(
        ConstSlice<uint64_t> input_q, 
        ConstSlice<uint64_t> input_Bsk, 

        const RNSTool& rns_tool,
        size_t dest_size,
        
        Slice<uint64_t> destination, 
        MemoryPoolHandle pool
    );
    
}