#pragma once
#include "encryption_parameters.cuh"
#include "utils/dynamic_array.cuh"

namespace troy {

    class Plaintext {
    
    private:
        size_t coeff_count_;
        utils::DynamicArray<uint64_t> data_;
        ParmsID parms_id_;
        double scale_;

    public:

        inline Plaintext():
            coeff_count_(0),
            data_(0, false),
            parms_id_(parms_id_zero),
            scale_(1.0) {}

        inline Plaintext clone() const {
            Plaintext result;
            result.coeff_count_ = this->coeff_count_;
            result.data_ = this->data_.clone();
            result.parms_id_ = this->parms_id_;
            result.scale_ = this->scale_;
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

        inline size_t coeff_count() const noexcept {
            return coeff_count_;
        }

        inline size_t& coeff_count() noexcept {
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

        inline utils::ConstSlice<uint64_t> poly() const noexcept {
            return data_.const_reference();
        }

        inline utils::Slice<uint64_t> poly() noexcept {
            return data_.reference();
        }

        inline void resize(size_t coeff_count) {
            if (this->is_ntt_form()) {
                throw std::invalid_argument("[Plaintext::resize] Cannot resize ntt form plaintext");
            }
            this->coeff_count_ = coeff_count;
            this->data_.resize(coeff_count);
        }

        inline bool is_ntt_form() const {
            return this->parms_id_ != parms_id_zero;
        }

        inline utils::ConstSlice<uint64_t> component(size_t index) const {
            return this->data_.const_slice(
                index * this->coeff_count_,
                (index + 1) * this->coeff_count_
            );
        }

        inline utils::Slice<uint64_t> component(size_t index) {
            return this->data_.slice(
                index * this->coeff_count_,
                (index + 1) * this->coeff_count_
            );
        }
        
    };

}