#pragma once
#include <optional>
#include <memory>
#include "encryption_parameters.h"
#include "utils/ntt.h"
#include "utils/galois.h"
#include "utils/rns_tool.h"
#include "coeff_modulus.h"

namespace troy {

    class HeContext;

    class ContextData {

        friend class HeContext;

    private:
        bool device;
        EncryptionParameters parms_;
        EncryptionParameterQualifiers qualifiers_;
        std::optional<utils::RNSTool> rns_tool_;
        utils::Array<utils::NTTTables> small_ntt_tables_;
        std::optional<utils::Box<utils::NTTTables>> plain_ntt_tables_;
        std::optional<utils::GaloisTool> galois_tool_;
        utils::Array<uint64_t> total_coeff_modulus_;
        size_t total_coeff_modulus_bit_count_;
        utils::Array<utils::MultiplyUint64Operand> coeff_div_plain_modulus_;
        uint64_t plain_upper_half_threshold_;
        utils::Array<uint64_t> plain_upper_half_increment_;
        utils::Array<uint64_t> upper_half_threshold_;
        utils::Array<uint64_t> upper_half_increment_;
        uint64_t coeff_modulus_mod_plain_modulus_;

        std::optional<std::weak_ptr<const ContextData>> prev_context_data_;
        std::optional<std::shared_ptr<const ContextData>> next_context_data_;
        size_t chain_index_;

        std::optional<std::weak_ptr<HeContext>> context_;

        void validate(SecurityLevel sec_level);

        void set_context(std::shared_ptr<HeContext> context);

    public:

        inline MemoryPoolHandle pool() const { return parms_.pool(); }
        inline size_t device_index() const { return parms_.device_index(); }

        std::optional<std::shared_ptr<HeContext>> context() const;
        std::shared_ptr<HeContext> context_pointer() const;

        inline bool on_device() const noexcept {
            return device;
        }

        inline const EncryptionParameters& parms() const noexcept {
            return parms_;
        }

        inline const EncryptionParameterQualifiers& qualifiers() const noexcept {
            return qualifiers_;
        }

        inline const utils::RNSTool& rns_tool() const {
            return rns_tool_.value();
        }

        inline utils::ConstSlice<utils::NTTTables> small_ntt_tables() const {
            return small_ntt_tables_.const_reference();
        }

        inline utils::ConstPointer<utils::NTTTables> plain_ntt_tables() const {
            return plain_ntt_tables_.value().as_const_pointer();
        }

        inline const utils::GaloisTool& galois_tool() const {
            return galois_tool_.value();
        }

        inline utils::ConstSlice<uint64_t> total_coeff_modulus() const {
            return total_coeff_modulus_.const_reference();
        }

        inline size_t total_coeff_modulus_bit_count() const noexcept {
            return total_coeff_modulus_bit_count_;
        }

        inline utils::ConstSlice<utils::MultiplyUint64Operand> coeff_div_plain_modulus() const {
            return coeff_div_plain_modulus_.const_reference();
        }

        inline uint64_t plain_upper_half_threshold() const noexcept {
            return plain_upper_half_threshold_;
        }

        inline utils::ConstSlice<uint64_t> plain_upper_half_increment() const {
            return plain_upper_half_increment_.const_reference();
        }

        inline utils::ConstSlice<uint64_t> upper_half_threshold() const {
            return upper_half_threshold_.const_reference();
        }

        inline utils::ConstSlice<uint64_t> upper_half_increment() const {
            return upper_half_increment_.const_reference();
        }

        inline uint64_t coeff_modulus_mod_plain_modulus() const noexcept {
            return coeff_modulus_mod_plain_modulus_;
        }

        inline std::optional<std::weak_ptr<const ContextData>> prev_context_data() const noexcept {
            return prev_context_data_;
        }

        inline std::optional<std::shared_ptr<const ContextData>> next_context_data() const noexcept {
            return next_context_data_;
        }

        inline std::weak_ptr<const ContextData> prev_context_data_pointer() const noexcept {
            if (!prev_context_data_.has_value()) {
                return std::weak_ptr<const ContextData>();
            } else {
                return prev_context_data_.value();
            }
        }

        inline std::shared_ptr<const ContextData> next_context_data_pointer() const noexcept {
            if (!next_context_data_.has_value()) {
                return std::shared_ptr<const ContextData>(nullptr);
            } else {
                return next_context_data_.value();
            }
        }

        inline size_t chain_index() const noexcept {
            return chain_index_;
        }

        inline const ParmsID& parms_id() const {
            return parms_.parms_id();
        }

        inline bool is_bfv() const noexcept {
            return parms_.scheme() == SchemeType::BFV;
        }

        inline bool is_ckks() const noexcept {
            return parms_.scheme() == SchemeType::CKKS;
        }

        inline bool is_bgv() const noexcept {
            return parms_.scheme() == SchemeType::BGV;
        }

        inline ContextData(EncryptionParameters parms):
            parms_(std::move(parms)), device(false) {}

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool());

    };

    using ContextDataPointer = std::shared_ptr<const ContextData>;

}