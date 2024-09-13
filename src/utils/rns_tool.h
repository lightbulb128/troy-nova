#pragma once
#include "rns_base.h"
#include "ntt.h"
#include "poly_small_mod.h"
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

        uint64_t m_tilde_value_;
        uint64_t inv_q_last_mod_t_;
        uint64_t q_last_mod_t_;
        uint64_t q_last_half_;
        // uint64_t gamma_half_;

    public:

        inline MemoryPoolHandle pool() const { return base_q_.pool(); }
        inline size_t device_index() const { return base_q_.device_index(); }

        inline size_t coeff_count() const noexcept { return coeff_count_; }

        inline const RNSBase& base_q() const noexcept { return base_q_; }
        inline const RNSBase& base_B() const noexcept { return base_B_; }
        inline const RNSBase& base_Bsk() const noexcept { return base_Bsk_; }
        inline const RNSBase& base_Bsk_m_tilde() const noexcept { return base_Bsk_m_tilde_; }
        inline const RNSBase& base_t_gamma() const { return base_t_gamma_.value(); }

        inline const BaseConverter& base_q_to_Bsk_conv() const noexcept { return base_q_to_Bsk_conv_; }
        inline const BaseConverter& base_q_to_m_tilde_conv() const noexcept { return base_q_to_m_tilde_conv_; }
        inline const BaseConverter& base_B_to_q_conv() const noexcept { return base_B_to_q_conv_; }
        inline const BaseConverter& base_B_to_m_sk_conv() const noexcept { return base_B_to_m_sk_conv_; }
        inline const BaseConverter& base_q_to_t_gamma_conv() const { return base_q_to_t_gamma_conv_.value(); }
        inline const BaseConverter& base_q_to_t_conv() const { return base_q_to_t_conv_.value(); }

        inline ConstSlice<MultiplyUint64Operand> inv_prod_q_mod_Bsk() const { return inv_prod_q_mod_Bsk_.const_reference(); }
        inline MultiplyUint64Operand neg_inv_prod_q_mod_m_tilde() const noexcept { return neg_inv_prod_q_mod_m_tilde_; }
        inline MultiplyUint64Operand inv_prod_B_mod_m_sk() const noexcept { return inv_prod_B_mod_m_sk_; }
        inline MultiplyUint64Operand inv_gamma_mod_t() const { return inv_gamma_mod_t_.value(); }
        inline ConstSlice<uint64_t> prod_B_mod_q() const { return prod_B_mod_q_.const_reference(); }
        inline ConstSlice<MultiplyUint64Operand> inv_m_tilde_mod_Bsk() const { return inv_m_tilde_mod_Bsk_.const_reference(); }
        inline ConstSlice<uint64_t> prod_q_mod_Bsk() const { return prod_q_mod_Bsk_.const_reference(); }
        inline ConstSlice<MultiplyUint64Operand> neg_inv_q_mod_t_gamma() const { return neg_inv_q_mod_t_gamma_.value().const_reference(); }
        inline ConstSlice<MultiplyUint64Operand> prod_t_gamma_mod_q() const { return prod_t_gamma_mod_q_.value().const_reference(); }
        inline ConstSlice<MultiplyUint64Operand> inv_q_last_mod_q() const { return inv_q_last_mod_q_.const_reference(); }
        inline ConstSlice<NTTTables> base_Bsk_ntt_tables() const { return base_Bsk_ntt_tables_.const_reference(); }

        inline ConstPointer<Modulus> m_tilde() const { return m_tilde_.as_const_pointer(); }
        inline ConstPointer<Modulus> m_sk() const { return m_sk_.as_const_pointer(); }
        inline ConstPointer<Modulus> t() const { return t_.as_const_pointer(); }
        inline ConstPointer<Modulus> gamma() const { return gamma_.as_const_pointer(); }

        inline uint64_t m_tilde_value() const noexcept { return m_tilde_value_; }
        inline uint64_t q_last_half() const noexcept { return q_last_half_; }
        inline uint64_t inv_q_last_mod_t() const noexcept { return inv_q_last_mod_t_; }
        inline uint64_t q_last_mod_t() const noexcept { return q_last_mod_t_; }


        RNSTool() {}

        inline bool on_device() const noexcept { return device; } 

        RNSTool(size_t poly_modulus_degree, const RNSBase& q, const Modulus& t);

        RNSTool clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool());
        inline RNSTool to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            RNSTool res = clone(pool);
            res.to_device_inplace(pool);
            return res;
        }

        void divide_and_round_q_last(ConstSlice<uint64_t> input, size_t input_pcount, Slice<uint64_t> destination) const;

        void divide_and_round_q_last_batched(const ConstSliceVec<uint64_t>& input, size_t input_pcount, const SliceVec<uint64_t>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void divide_and_round_q_last_ntt(ConstSlice<uint64_t> input, size_t input_pcount, Slice<uint64_t> destination, ConstSlice<NTTTables> rns_ntt_tables, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void divide_and_round_q_last_ntt_batched(
            const ConstSliceVec<uint64_t>& input, size_t input_pcount, 
            const SliceVec<uint64_t>& destination, ConstSlice<NTTTables> rns_ntt_tables, 
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const;

        void fast_b_conv_sk(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void sm_mrq(ConstSlice<uint64_t> input, Slice<uint64_t> destination) const;

        void fast_floor(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void fast_floor_fast_b_conv_sk(ConstSlice<uint64_t> input_q, ConstSlice<uint64_t> input_Bsk, Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void fast_b_conv_m_tilde(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        // = fast_b_conv_m_tilde + sm_mrq, the output to the former is the input to the latter
        void fast_b_conv_m_tilde_sm_mrq(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void decrypt_scale_and_round(ConstSlice<uint64_t> phase, size_t phase_coeff_count, Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void decrypt_scale_and_round_batched(const ConstSliceVec<uint64_t>& phase, size_t phase_coeff_count, const SliceVec<uint64_t>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void mod_t_and_divide_q_last_inplace(Slice<uint64_t> input, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        void mod_t_and_divide_q_last_ntt(ConstSlice<uint64_t> input, size_t pcount, Slice<uint64_t> destination, ConstSlice<NTTTables> rns_ntt_tables, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        
        void mod_t_and_divide_q_last_ntt_batched(const ConstSliceVec<uint64_t>& input, size_t pcount, const SliceVec<uint64_t>& destination, ConstSlice<NTTTables> rns_ntt_tables, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;

        inline void decrypt_mod_t(ConstSlice<uint64_t> phase, Slice<uint64_t> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            this->base_q_to_t_conv().exact_convey_array(phase, destination, pool);
        }

    };

}}