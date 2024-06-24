#pragma once
#include "../utils/basics.cuh"
#include "../modulus.cuh"
#include "../he_context.cuh"
#include "../plaintext.cuh"


namespace troy::linear {

    // Only available for T = uint32_t, uint64_t and uint128_t
    template<typename T>
    struct is_compatible_ring2k {
        static constexpr bool value = false;
    };

    template<> struct is_compatible_ring2k<uint32_t> 
    {static constexpr bool value = true;};
    template<> struct is_compatible_ring2k<uint64_t>
    {static constexpr bool value = true;};
    template<> struct is_compatible_ring2k<__uint128_t>
    {static constexpr bool value = true;};

    template <typename T>
    class PolynomialEncoderRNSHelper {

        static_assert(is_compatible_ring2k<T>::value, "T must be uint32_t, uint64_t or uint128_t");

        private:
            troy::ParmsID parms_id_;
            size_t t_bit_length_;
            troy::Modulus gamma_host_;
            troy::utils::Box<troy::Modulus> gamma_;
            T neg_inv_Q_mod_t_;
            T inv_gamma_mod_t_;
            troy::utils::Box<troy::utils::MultiplyUint64Operand> neg_inv_Q_mod_gamma_;
            troy::utils::Array<T> punctured_q_mod_t_;
            troy::utils::Array<troy::utils::MultiplyUint64Operand> gamma_t_mod_Q_;
            troy::utils::BaseConverter base_Q_to_gamma_;
            T mod_t_mask_;
            T t_half_;
            T Q_mod_t_;
            troy::utils::Array<troy::utils::MultiplyUint64Operand> Q_div_t_mod_qi_;
        
        public:

            inline ParmsID parms_id() const noexcept { return parms_id_; }
            inline size_t t_bit_length () const noexcept { return t_bit_length_; }
            inline const Modulus& gamma_host() const noexcept { return gamma_host_; }
            inline utils::ConstPointer<Modulus> gamma() const noexcept { return gamma_.as_const_pointer(); }
            inline T neg_inv_Q_mod_t() const noexcept { return neg_inv_Q_mod_t_; }
            inline T inv_gamma_mod_t() const noexcept { return inv_gamma_mod_t_; }
            inline utils::ConstPointer<utils::MultiplyUint64Operand> neg_inv_Q_mod_gamma() const noexcept { return neg_inv_Q_mod_gamma_.as_const_pointer(); }
            inline utils::ConstSlice<T> punctured_q_mod_t() const noexcept { return punctured_q_mod_t_.const_reference(); }
            inline utils::ConstSlice<utils::MultiplyUint64Operand> gamma_t_mod_Q() const noexcept { return gamma_t_mod_Q_.const_reference(); }
            inline const utils::BaseConverter& base_Q_to_gamma() const noexcept { return base_Q_to_gamma_; }
            inline T mod_t_mask() const noexcept { return mod_t_mask_; }
            inline T t_half() const noexcept { return t_half_; }
            inline T Q_mod_t() const noexcept { return Q_mod_t_; }
            inline utils::ConstSlice<utils::MultiplyUint64Operand> Q_div_t_mod_qi() const noexcept { return Q_div_t_mod_qi_.const_reference(); }
            inline bool on_device() const noexcept { return gamma_.on_device(); }

            PolynomialEncoderRNSHelper(ContextDataPointer context_data, size_t t_bit_length);
            void to_device_inplace();

            void scale_up_component(utils::ConstSlice<T> source, const HeContext& context, size_t modulus_index, utils::Slice<uint64_t> destination) const;
            void centralize_at_component(utils::ConstSlice<T> source, const HeContext& context, size_t modulus_index, utils::Slice<uint64_t> destination) const;

            void scale_up(utils::ConstSlice<T> source, const HeContext& context, Plaintext& destination) const {
                if (source.on_device() != this->on_device()) {
                    throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source and helper must be on the same device");
                }
                if (on_device()) destination.to_device_inplace();
                else destination.to_host_inplace();
                destination.is_ntt_form() = false;
                destination.resize_rns(context, parms_id_);
                for (size_t i = 0; i < destination.coeff_modulus_size(); i++) {
                    this->scale_up_component(source, context, i, destination.component(i));
                }
            }
            void centralize(utils::ConstSlice<T> source, const HeContext& context, Plaintext& destination) const {
                if (source.on_device() != this->on_device()) {
                    throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source and helper must be on the same device");
                }
                if (on_device()) destination.to_device_inplace();
                else destination.to_host_inplace();
                destination.is_ntt_form() = false;
                destination.resize_rns(context, parms_id_);
                for (size_t i = 0; i < destination.coeff_modulus_size(); i++) {
                    this->centralize_at_component(source, context, i, destination.component(i));
                }
            }
            
            void scale_down(const Plaintext& input, const HeContext& context, utils::Slice<T> destination) const;

    };

    template <typename T>
    class PolynomialEncoderRing2k {
        
        static_assert(is_compatible_ring2k<T>::value, "T must be uint32_t, uint64_t or uint128_t");

        private:
            HeContextPointer context_;
            size_t t_bit_length_;
            std::unordered_map<ParmsID, std::shared_ptr<PolynomialEncoderRNSHelper<T>>, std::TroyHashParmsID> helpers_;
    
        public:

            inline HeContextPointer context() const noexcept { return context_; }
            inline size_t t_bit_length() const noexcept { return t_bit_length_; }
            inline bool on_device() const noexcept { return context_->on_device(); }

            PolynomialEncoderRing2k(HeContextPointer context, size_t t_bit_length);
            std::optional<std::shared_ptr<PolynomialEncoderRNSHelper<T>>> get_helper(const ParmsID& parms_id) const {
                auto it = helpers_.find(parms_id);
                if (it == helpers_.end()) {
                    return std::nullopt;
                }
                return it->second;
            }

            inline void to_device_inplace() {
                for (auto& [_, helper]: helpers_) {
                    helper->to_device_inplace();
                }
            }

            void scale_up(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id, Plaintext& destination) const {
                ParmsID pid = parms_id.value_or(context_->first_parms_id());
                auto helper = get_helper(pid);
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k:scale_up] No helper found for the given parms_id");
                }
                helper.value()->scale_up(source, *context_, destination);
            }
            void scale_up(const std::vector<T>& source, std::optional<ParmsID> parms_id, Plaintext& destination) const {
                if (on_device()) {
                    utils::Array<T> source_array(source.size(), true); 
                    source_array.copy_from_slice(utils::ConstSlice<T>(source.data(), source.size(), false));
                    this->scale_up(source_array.const_reference(), parms_id, destination);
                } else {
                    this->scale_up(utils::ConstSlice<T>(source.data(), source.size(), false), parms_id, destination);
                }
            }

            Plaintext scale_up_new(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id) const {
                Plaintext destination;
                this->scale_up(source, parms_id, destination);
                return destination;
            }
            Plaintext scale_up_new(const std::vector<T>& source, std::optional<ParmsID> parms_id) const {
                Plaintext destination;
                this->scale_up(source, parms_id, destination);
                return destination;
            }

            void centralize(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id, Plaintext& destination) const {
                ParmsID pid = parms_id.value_or(context_->first_parms_id());
                auto helper = get_helper(pid);
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k:centralize] No helper found for the given parms_id");
                }
                helper.value()->centralize(source, *context_, destination);
            }
            void centralize(const std::vector<T>& source, std::optional<ParmsID> parms_id, Plaintext& destination) const {
                if (on_device()) {
                    utils::Array<T> source_array(source.size(), true); 
                    source_array.copy_from_slice(utils::ConstSlice<T>(source.data(), source.size(), false));
                    this->centralize(source_array.const_reference(), parms_id, destination);
                } else {
                    this->centralize(utils::ConstSlice<T>(source.data(), source.size(), false), parms_id, destination);
                }
            }

            Plaintext centralize_new(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id) const {
                Plaintext destination;
                this->centralize(source, parms_id, destination);
                return destination;
            }
            Plaintext centralize_new(const std::vector<T>& source, std::optional<ParmsID> parms_id) const {
                Plaintext destination;
                this->centralize(source, parms_id, destination);
                return destination;
            }

            void scale_down(const Plaintext& input, utils::Slice<T> destination) const {
                auto helper = get_helper(input.parms_id());
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k:scale_down] No helper found for the given parms_id");
                }
                helper.value()->scale_down(input, *context_, destination);
            }

            std::vector<T> scale_down_new(const Plaintext& input) const {
                if (input.on_device()) {
                    utils::Array<T> destination(input.poly_modulus_degree(), true);
                    this->scale_down(input, destination.reference());
                    return destination.to_vector();
                } else {
                    std::vector<T> destination(input.poly_modulus_degree());
                    this->scale_down(input, utils::Slice<T>(destination.data(), destination.size(), false));
                    return destination;
                }
            }

    };

}