#pragma once
#include "encryption_parameters.cuh"
#include "he_context.cuh"
#include "utils/dynamic_array.cuh"
#include "utils/serialize.h"
#include "utils/poly_to_string.cuh"

namespace troy {

    class Plaintext {
    
    private:
        size_t coeff_count_;
        utils::DynamicArray<uint64_t> data_;
        ParmsID parms_id_;
        double scale_;
        bool is_ntt_form_;
        size_t coeff_modulus_size_;
        size_t poly_modulus_degree_;
        
        void resize_rns_internal(size_t poly_modulus_degree, size_t coeff_modulus_size);

    public:

        inline Plaintext():
            coeff_count_(0),
            data_(0, false),
            parms_id_(parms_id_zero),
            scale_(1.0),
            is_ntt_form_(false),
            coeff_modulus_size_(0),
            poly_modulus_degree_(0) {}

        inline Plaintext(Plaintext&& source) = default;
        inline Plaintext(const Plaintext& copy): Plaintext(copy.clone()) {}

        inline Plaintext& operator =(Plaintext&& source) = default;
        inline Plaintext& operator =(const Plaintext& assign) {
            if (this == &assign) {
                return *this;
            }
            *this = assign.clone();
            return *this;
        }

        inline Plaintext clone() const {
            Plaintext result;
            result.coeff_count_ = this->coeff_count_;
            result.data_ = this->data_.clone();
            result.parms_id_ = this->parms_id_;
            result.scale_ = this->scale_;
            result.is_ntt_form_ = this->is_ntt_form_;
            result.coeff_modulus_size_ = this->coeff_modulus_size_;
            result.poly_modulus_degree_ = this->poly_modulus_degree_;
            return result;
        }

        inline void to_device_inplace() {
            this->data_.to_device_inplace();
        }

        inline void to_host_inplace() {
            this->data_.to_host_inplace();
        }

        inline Plaintext to_device() {
            Plaintext result = this->clone();
            result.to_device_inplace();
            return result;
        }

        inline Plaintext to_host() {
            Plaintext result = this->clone();
            result.to_host_inplace();
            return result;
        }

        inline bool on_device() const noexcept {
            return data_.on_device();
        }

        inline const ParmsID& parms_id() const noexcept {
            return parms_id_;
        }

        inline ParmsID& parms_id() noexcept {
            return parms_id_;
        }

        inline size_t coeff_count() const {
            if (this->parms_id_ != parms_id_zero) {
                throw std::logic_error("[Plaintext::coeff_count] Coefficient count is only meaningful when the plaintext is under plaintext modulus t.");
            }
            return coeff_count_;
        }

        inline size_t& coeff_count() {
            if (this->parms_id_ != parms_id_zero) {
                throw std::logic_error("[Plaintext::coeff_count] Coefficient count is only meaningful when the plaintext is under plaintext modulus t.");
            }
            return coeff_count_;
        }

        inline double scale() const noexcept {
            return scale_;
        }

        inline double& scale() noexcept {
            return scale_;
        }

        inline utils::DynamicArray<uint64_t>& data() noexcept {
            return data_;
        }

        inline const utils::DynamicArray<uint64_t>& data() const noexcept {
            return data_;
        }

        inline const utils::DynamicArray<uint64_t>& const_data() const noexcept {
            return data_;
        }

        inline utils::ConstSlice<uint64_t> poly() const noexcept {
            return data_.const_reference();
        }

        inline utils::Slice<uint64_t> poly() noexcept {
            return data_.reference();
        }

        inline utils::ConstSlice<uint64_t> const_poly() const noexcept {
            return data_.const_reference();
        }

        inline utils::ConstSlice<uint64_t> component(size_t index) const {
            if (this->parms_id_ == parms_id_zero) {
                if (index != 0) {
                    throw std::out_of_range("[Plaintext::component] Index out of range");
                }
                return this->poly();
            } else {
                size_t start = index * this->poly_modulus_degree();
                return this->data_.const_slice(start, start + this->poly_modulus_degree());
            }
        }

        inline utils::Slice<uint64_t> component(size_t index) {
            if (this->parms_id_ == parms_id_zero) {
                if (index != 0) {
                    throw std::out_of_range("[Plaintext::component] Index out of range");
                }
                return this->poly();
            } else {
                size_t start = index * this->poly_modulus_degree();
                return this->data_.slice(start, start + this->poly_modulus_degree());
            }
        }

        inline utils::ConstSlice<uint64_t> const_component(size_t index) const {
            return this->component(index);
        }

        inline utils::ConstSlice<uint64_t> reference() const noexcept {return this->poly();}
        inline utils::Slice<uint64_t> reference() noexcept {return this->poly();}
        inline utils::ConstSlice<uint64_t> const_reference() const noexcept {return this->const_poly();}

        inline void resize(size_t coeff_count) {
            if (this->parms_id_ != parms_id_zero) {
                throw std::invalid_argument("[Plaintext::resize] Cannot resize if the plaintext is not mod t. Call resize_rns instead.");
            }
            this->coeff_count_ = coeff_count;
            this->data_.resize(coeff_count);
        }

        inline size_t coeff_modulus_size () const noexcept {
            return coeff_modulus_size_;
        }

        inline size_t& coeff_modulus_size() noexcept {
            return coeff_modulus_size_;
        }

        inline size_t poly_modulus_degree() const noexcept {
            return poly_modulus_degree_;
        }

        inline size_t& poly_modulus_degree() noexcept {
            return poly_modulus_degree_;
        }

        void resize_rns(HeContextPointer context, const ParmsID& parms_id);

        inline bool is_ntt_form() const {
            return is_ntt_form_;
        }

        inline bool& is_ntt_form() {
            return is_ntt_form_;
        }

        void save(std::ostream& stream) const;
        void load(std::istream& stream);
        inline static Plaintext load_new(std::istream& stream) {
            Plaintext result;
            result.load(stream);
            return result;
        }
        size_t serialized_size() const;

        inline std::string to_string() const {
            if (is_ntt_form() || parms_id_ != parms_id_zero)
            {
                throw std::invalid_argument("cannot convert NTT or RNS plaintext to string");
            }
            std::vector<uint64_t> copied = data_.to_vector();
            return utils::poly_to_hex_string(copied.data(), coeff_count_, 1);
        }
        
    };

}