#pragma once
#include "he_context.cuh"
#include "encryption_parameters.cuh"
#include "utils/dynamic_array.cuh"
#include "utils/reduction.cuh"

namespace troy {

    class Ciphertext {

    private:
        size_t polynomial_count_;
        size_t coeff_modulus_size_;
        size_t poly_modulus_degree_;
        utils::DynamicArray<uint64_t> data_;
        ParmsID parms_id_;
        double scale_;
        bool is_ntt_form_;
        uint64_t correction_factor_;
        uint64_t seed_;

        void resize_internal(size_t polynomial_count, size_t coeff_modulus_size, size_t poly_modulus_degree);

    public:
    
        inline static Ciphertext from_members(
            size_t polynomial_count, size_t coeff_modulus_size, size_t poly_modulus_degree, 
            const ParmsID& parms_id, double scale, bool is_ntt_form, uint64_t correction_factor, uint64_t seed, 
            utils::DynamicArray<uint64_t>&& data
        ) {
            Ciphertext result;
            result.polynomial_count_ = polynomial_count;
            result.coeff_modulus_size_ = coeff_modulus_size;
            result.poly_modulus_degree_ = poly_modulus_degree;
            result.data_ = std::move(data);
            result.parms_id_ = parms_id;
            result.scale_ = scale;
            result.is_ntt_form_ = is_ntt_form;
            result.correction_factor_ = correction_factor;
            result.seed_ = seed;
            return result;
        }

        inline Ciphertext():
            polynomial_count_(0),
            coeff_modulus_size_(0),
            poly_modulus_degree_(0),
            data_(),
            parms_id_(parms_id_zero),
            scale_(1.0),
            is_ntt_form_(false),
            correction_factor_(1),
            seed_(0) {}

        inline Ciphertext clone() const {
            Ciphertext result;
            result.polynomial_count_ = polynomial_count_;
            result.coeff_modulus_size_ = coeff_modulus_size_;
            result.poly_modulus_degree_ = poly_modulus_degree_;
            result.data_ = data_.clone();
            result.parms_id_ = parms_id_;
            result.scale_ = scale_;
            result.is_ntt_form_ = is_ntt_form_;
            result.correction_factor_ = correction_factor_;
            result.seed_ = seed_;
            return result;
        }

        inline Ciphertext(Ciphertext&& source) = default;
        inline Ciphertext(const Ciphertext& copy): Ciphertext(copy.clone()) {}

        inline Ciphertext& operator =(Ciphertext&& source) = default;
        inline Ciphertext& operator =(const Ciphertext& assign) {
            if (this == &assign) {
                return *this;
            }
            *this = assign.clone();
            return *this;
        }

        inline const ParmsID& parms_id() const noexcept {
            return parms_id_;
        }

        inline ParmsID& parms_id() noexcept {
            return parms_id_;
        }

        inline size_t polynomial_count() const noexcept {
            return polynomial_count_;
        }

        inline size_t coeff_modulus_size() const noexcept {
            return coeff_modulus_size_;
        }

        inline size_t poly_modulus_degree() const noexcept {
            return poly_modulus_degree_;
        }

        inline double scale() const noexcept {
            return scale_;
        }

        inline double& scale() noexcept {
            return scale_;
        }

        inline bool is_ntt_form() const noexcept {
            return is_ntt_form_;
        }

        inline bool& is_ntt_form() noexcept {
            return is_ntt_form_;
        }

        inline uint64_t correction_factor() const noexcept {
            return correction_factor_;
        }

        inline uint64_t& correction_factor() noexcept {
            return correction_factor_;
        }

        inline uint64_t seed() const noexcept {
            return seed_;
        }

        inline uint64_t& seed() noexcept {
            return seed_;
        }

        inline bool contains_seed() const noexcept {
            return seed_ != 0;
        }

        inline bool on_device() const noexcept {
            return data_.on_device();
        }

        inline void to_device_inplace() {
            data_.to_device_inplace();
        }

        inline void to_host_inplace() {
            data_.to_host_inplace();
        }

        inline Ciphertext to_device() {
            Ciphertext result = clone();
            result.to_device_inplace();
            return result;
        }

        inline Ciphertext to_host() {
            Ciphertext result = clone();
            result.to_host_inplace();
            return result;
        }

        inline const utils::DynamicArray<uint64_t>& data() const noexcept {
            return data_;
        }

        inline utils::DynamicArray<uint64_t>& data() noexcept {
            return data_;
        }

        void resize(HeContextPointer context, const ParmsID& parms_id, size_t polynomial_count);

        inline utils::ConstSlice<uint64_t> reference() const noexcept {
            return data_.const_reference();
        }

        inline utils::Slice<uint64_t> reference() noexcept {
            return data_.reference();
        }

        inline utils::ConstSlice<uint64_t> const_reference() const noexcept {
            return this->reference();
        }

        inline utils::ConstSlice<uint64_t> poly(size_t poly_id) const {
            size_t d = coeff_modulus_size_ * poly_modulus_degree_;
            return data_.const_slice(poly_id * d, (poly_id + 1) * d);
        }

        inline utils::Slice<uint64_t> poly(size_t poly_id) {
            size_t d = coeff_modulus_size_ * poly_modulus_degree_;
            return data_.slice(poly_id * d, (poly_id + 1) * d);
        }

        inline utils::ConstSlice<uint64_t> const_poly(size_t poly_id) const {
            return this->poly(poly_id);
        }

        inline utils::ConstSlice<uint64_t> polys(size_t lower_poly_id, size_t upper_poly_id) const {
            size_t d = coeff_modulus_size_ * poly_modulus_degree_;
            return data_.const_slice(lower_poly_id * d, upper_poly_id * d);
        }

        inline utils::Slice<uint64_t> polys(size_t lower_poly_id, size_t upper_poly_id) {
            size_t d = coeff_modulus_size_ * poly_modulus_degree_;
            return data_.slice(lower_poly_id * d, upper_poly_id * d);
        }

        inline utils::ConstSlice<uint64_t> const_polys(size_t lower_poly_id, size_t upper_poly_id) const {
            return this->polys(lower_poly_id, upper_poly_id);
        }

        inline utils::ConstSlice<uint64_t> poly_component(size_t poly_id, size_t component_id) const {
            size_t offset = poly_modulus_degree_ * (poly_id * coeff_modulus_size_ + component_id);
            return data_.const_slice(offset, offset + poly_modulus_degree_);
        }

        inline utils::Slice<uint64_t> poly_component(size_t poly_id, size_t component_id) {
            size_t offset = poly_modulus_degree_ * (poly_id * coeff_modulus_size_ + component_id);
            return data_.slice(offset, offset + poly_modulus_degree_);
        }

        inline utils::ConstSlice<uint64_t> const_poly_component(size_t poly_id, size_t component_id) const {
            return this->poly_component(poly_id, component_id);
        }

        bool is_transparent() const;

    };

}