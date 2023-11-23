#pragma once
#include "rns_base.cuh"
#include "ntt.cuh"
#include "poly_small_mod.cuh"
#include <optional>
#include <vector>


namespace troy {namespace utils {

    class RNSTool {

    private:

        bool device;

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
        // uint64_t gamma_half_;

    public:

        inline size_t coeff_count() const noexcept { return coeff_count_; }

        inline const RNSBase& base_q() const noexcept { return base_q_; }
        inline const RNSBase& base_B() const noexcept { return base_B_; }
        inline const RNSBase& base_Bsk() const noexcept { return base_Bsk_; }
        inline const RNSBase& base_Bsk_m_tilde() const noexcept { return base_Bsk_m_tilde_; }
        inline const RNSBase& base_t_gamma() const { return base_t_gamma_.value(); }

        inline ConstSlice<MultiplyUint64Operand> inv_q_last_mod_q() const { return inv_q_last_mod_q_.const_reference(); }

        inline uint64_t q_last_half() const noexcept { return q_last_half_; }

        RNSTool() {}

        inline bool on_device() const noexcept { return device; } 

        RNSTool(size_t poly_modulus_degree, const RNSBase& q, const Modulus& t);

        RNSTool clone() const;
        void to_device_inplace();
        inline RNSTool to_device() const {
            RNSTool res = clone();
            res.to_device_inplace();
            return res;
        }

        void divide_and_round_q_last_inplace(Slice<uint64_t> input) const;

    };

}}