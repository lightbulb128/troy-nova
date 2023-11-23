#pragma once
#include "rns_base.cuh"
#include "ntt.cuh"
#include <optional>

namespace troy {namespace utils {

    class RNSTool {

    private:

        size_t coeff_count_;

        RNSBase base_q_;
        RNSBase base_B_;
        RNSBase base_Bsk_;
        RNSBase base_Bsk_m_tilde_;
        std::optional<RNSBase> base_t_gamma_;

        BaseConverter base_q_to_Bsk_conv_;
        BaseConverter base_q_to_m_tilde_conv_;
        BaseConverter base_B_to_q_conv_;
        BaseConverter base_B_to_m_sk_conv_;
        std::optional<BaseConverter> base_q_to_t_gamma_conv_;
        std::optional<BaseConverter> base_q_to_t_conv_;

        Array<MultiplyUint64Operand> inv_prod_q_mod_Bsk_;
        MultiplyUint64Operand neg_inv_prod_q_mod_m_tilde_;
        MultiplyUint64Operand inv_prod_B_mod_m_sk_;
        std::optional<MultiplyUint64Operand> inv_gamma_mod_t_;
        Array<uint64_t> prod_B_mod_q_;
        Array<MultiplyUint64Operand> inv_m_tilde_mod_Bsk_;
        Array<uint64_t> prod_q_mod_Bsk_;
        std::optional<Array<MultiplyUint64Operand>> neg_inv_q_mod_t_gamma_;
        std::optional<Array<MultiplyUint64Operand>> prod_t_gamma_mod_q_;
        Array<MultiplyUint64Operand> inv_q_last_mod_q_;
        Array<NTTTables> base_Bsk_ntt_tables_;

        Box<Modulus> m_tilde_;
        Box<Modulus> m_sk_;
        Box<Modulus> t_;
        Box<Modulus> gamma_;

        uint64_t inv_q_last_mod_t_;
        uint64_t q_last_mod_t_;
        uint64_t q_last_half_;
        uint64_t gamma_half;

    public:

        // RNSTool(size_t poly_modulus_degree, const RNSBase& q, const Modulus& t);

    };

}}