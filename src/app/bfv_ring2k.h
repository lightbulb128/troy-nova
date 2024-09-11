#pragma once
#include "../utils/basics.h"
#include "../modulus.h"
#include "../he_context.h"
#include "../plaintext.h"


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
            inline const utils::RNSBase& base_Q() const { return base_Q_to_gamma_.input_base(); }
            inline T mod_t_mask() const noexcept { return mod_t_mask_; }
            inline T t_half() const noexcept { return t_half_; }
            inline T Q_mod_t() const noexcept { return Q_mod_t_; }
            inline utils::ConstSlice<utils::MultiplyUint64Operand> Q_div_t_mod_qi() const noexcept { return Q_div_t_mod_qi_.const_reference(); }
            inline bool on_device() const noexcept { return gamma_.on_device(); }
            inline size_t device_index() const { return gamma_.device_index(); }

            PolynomialEncoderRNSHelper(ContextDataPointer context_data, size_t t_bit_length);
            void to_device_inplace(MemoryPoolHandle pool);

            void scale_up(utils::ConstSlice<T> source, const HeContext& context, Plaintext& destination, MemoryPoolHandle pool) const;
            void scale_up_batched(const utils::ConstSliceVec<T>& source, const HeContext& context, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const;
            void centralize(utils::ConstSlice<T> source, const HeContext& context, Plaintext& destination, MemoryPoolHandle pool) const;
            void centralize_batched(const utils::ConstSliceVec<T>& source, const HeContext& context, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const;
            
            void scale_down(const Plaintext& input, const HeContext& context, utils::Slice<T> destination, MemoryPoolHandle pool) const;
            void decentralize(const Plaintext& input, const HeContext& context, utils::Slice<T> destination, T correction_factor, MemoryPoolHandle pool) const;

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
            inline T t_mask() const noexcept { 
                if constexpr (std::is_same_v<T, uint32_t>) {
                    if (t_bit_length_ == 32) return static_cast<T>(-1); else return (static_cast<T>(1) << t_bit_length_) - 1;
                } else if constexpr (std::is_same_v<T, uint64_t>) {
                    if (t_bit_length_ == 64) return static_cast<T>(-1); else return (static_cast<T>(1) << t_bit_length_) - 1;
                } else {
                    static_assert(std::is_same_v<T, __uint128_t>);
                    if (t_bit_length_ == 128) return static_cast<T>(-1); else return (static_cast<T>(1) << t_bit_length_) - 1;
                }
            }
            inline bool on_device() const noexcept { return context_->on_device(); }
            inline size_t slot_count() const { return context_->first_context_data_pointer()->parms().poly_modulus_degree();}

            PolynomialEncoderRing2k(HeContextPointer context, size_t t_bit_length);
            std::optional<std::shared_ptr<PolynomialEncoderRNSHelper<T>>> get_helper(const ParmsID& parms_id) const {
                auto it = helpers_.find(parms_id);
                if (it == helpers_.end()) {
                    return std::nullopt;
                }
                return it->second;
            }

            inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
                for (auto& [_, helper]: helpers_) {
                    helper->to_device_inplace(pool);
                }
            }

            void scale_up_slice(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                ParmsID pid = parms_id.value_or(context_->first_parms_id());
                auto helper = get_helper(pid);
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k:scale_up] No helper found for the given parms_id");
                }
                helper.value()->scale_up(source, *context_, destination, pool);
            }
            void scale_up_slice_batched(const utils::ConstSliceVec<T>& source, std::optional<ParmsID> parms_id, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                ParmsID pid = parms_id.value_or(context_->first_parms_id());
                auto helper = get_helper(pid);
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k:scale_up_batched] No helper found for the given parms_id");
                }
                helper.value()->scale_up_batched(source, *context_, destination, pool);
            }
            void scale_up(const std::vector<T>& source, std::optional<ParmsID> parms_id, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                if (on_device()) {
                    utils::Array<T> source_array = utils::Array<T>::create_uninitialized(source.size(), true, pool); 
                    source_array.copy_from_slice(utils::ConstSlice<T>(source.data(), source.size(), false, nullptr));
                    this->scale_up_slice(source_array.const_reference(), parms_id, destination, pool);
                } else {
                    this->scale_up_slice(utils::ConstSlice<T>(source.data(), source.size(), false, nullptr), parms_id, destination, pool);
                }
            }

            Plaintext scale_up_slice_new(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                Plaintext destination;
                this->scale_up_slice(source, parms_id, destination, pool);
                return destination;
            }
            Plaintext scale_up_new(const std::vector<T>& source, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                Plaintext destination;
                this->scale_up(source, parms_id, destination, pool);
                return destination;
            }

            void centralize_slice(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                ParmsID pid = parms_id.value_or(context_->first_parms_id());
                auto helper = get_helper(pid);
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k:centralize] No helper found for the given parms_id");
                }
                helper.value()->centralize(source, *context_, destination, pool);
            }
            void centralize_slice_batched(const utils::ConstSliceVec<T>& source, std::optional<ParmsID> parms_id, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                ParmsID pid = parms_id.value_or(context_->first_parms_id());
                auto helper = get_helper(pid);
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k:centralize_batched] No helper found for the given parms_id");
                }
                helper.value()->centralize_batched(source, *context_, destination, pool);
            }
            void centralize(const std::vector<T>& source, std::optional<ParmsID> parms_id, Plaintext& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                if (on_device()) {
                    utils::Array<T> source_array = utils::Array<T>::create_uninitialized(source.size(), true, pool); 
                    source_array.copy_from_slice(utils::ConstSlice<T>(source.data(), source.size(), false, nullptr));
                    this->centralize_slice(source_array.const_reference(), parms_id, destination, pool);
                } else {
                    this->centralize_slice(utils::ConstSlice<T>(source.data(), source.size(), false, nullptr), parms_id, destination, pool);
                }
            }

            Plaintext centralize_slice_new(utils::ConstSlice<T> source, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                Plaintext destination;
                this->centralize_slice(source, parms_id, destination, pool);
                return destination;
            }
            Plaintext centralize_new(const std::vector<T>& source, std::optional<ParmsID> parms_id, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                Plaintext destination;
                this->centralize(source, parms_id, destination, pool);
                return destination;
            }

            void scale_down_slice(const Plaintext& input, utils::Slice<T> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                auto helper = get_helper(input.parms_id());
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k::scale_down_slice] No helper found for the given parms_id");
                }
                helper.value()->scale_down(input, *context_, destination, pool);
            }
            void scale_down(const Plaintext& input, std::vector<T>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                if (input.on_device()) {
                    utils::Array<T> destination_array(input.coeff_count(), true, pool);
                    this->scale_down_slice(input, destination_array.reference(), pool);
                    destination = destination_array.to_vector();
                } else {
                    destination.resize(input.coeff_count());
                    this->scale_down_slice(input, utils::Slice<T>(destination.data(), destination.size(), false, nullptr), pool);
                }
            }

            utils::Array<T> scale_down_slice_new(const Plaintext& input, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                utils::Array<T> destination(input.coeff_count(), on_device(), pool);
                this->scale_down_slice(input, destination.reference(), pool);
                return destination;
            }
            std::vector<T> scale_down_new(const Plaintext& input, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                std::vector<T> destination;
                this->scale_down(input, destination, pool);
                return destination;
            }


            void decentralize_slice(const Plaintext& input, utils::Slice<T> destination, T correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                auto helper = get_helper(input.parms_id());
                if (!helper.has_value()) {
                    throw std::invalid_argument("[PolynomialEncoderRing2k::decentralize_slice] No helper found for the given parms_id");
                }
                helper.value()->decentralize(input, *context_, destination, correction_factor, pool);
            }
            void decentralize(const Plaintext& input, std::vector<T>& destination, T correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                if (input.on_device()) {
                    utils::Array<T> destination_array(input.coeff_count(), true, pool);
                    this->decentralize_slice(input, destination_array.reference(), correction_factor, pool);
                    destination = destination_array.to_vector();
                } else {
                    destination.resize(input.coeff_count());
                    this->decentralize_slice(input, utils::Slice<T>(destination.data(), destination.size(), false, nullptr), correction_factor, pool);
                }
            }

            utils::Array<T> decentralize_slice_new(const Plaintext& input, T correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                utils::Array<T> destination(input.coeff_count(), on_device(), pool);
                this->decentralize_slice(input, destination.reference(), correction_factor, pool);
                return destination;
            }
            std::vector<T> decentralize_new(const Plaintext& input, T correction_factor = 1, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
                std::vector<T> destination;
                this->decentralize(input, destination, correction_factor, pool);
                return destination;
            }
            

    };

}