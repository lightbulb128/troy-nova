#pragma once
#include "he_context.cuh"
#include "encryption_parameters.cuh"
#include "utils/dynamic_array.cuh"
#include "utils/reduction.cuh"
#include "he_context.cuh"
#include "utils/serialize.h"

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
    
        inline MemoryPoolHandle pool() const { return data_.pool(); }
        inline size_t device_index() const { return data_.device_index(); }

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

        inline Ciphertext clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Ciphertext result;
            result.polynomial_count_ = polynomial_count_;
            result.coeff_modulus_size_ = coeff_modulus_size_;
            result.poly_modulus_degree_ = poly_modulus_degree_;
            result.data_ = data_.clone(pool);
            result.parms_id_ = parms_id_;
            result.scale_ = scale_;
            result.is_ntt_form_ = is_ntt_form_;
            result.correction_factor_ = correction_factor_;
            result.seed_ = seed_;
            return std::move(result);
        }

        inline Ciphertext(Ciphertext&& source) = default;
        inline Ciphertext(const Ciphertext& copy): Ciphertext(copy.clone(copy.pool())) {}

        inline Ciphertext& operator =(Ciphertext&& source) = default;
        inline Ciphertext& operator =(const Ciphertext& assign) {
            if (this == &assign) {
                return *this;
            }
            *this = assign.clone(assign.pool());
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

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            if (this->contains_seed()) {
                throw std::logic_error("[Ciphertext::to_device_inplace] Cannot copy ciphertext with seed to device");
            }
            data_.to_device_inplace(pool);
        }

        inline void to_host_inplace() {
            if (this->contains_seed()) {
                throw std::logic_error("[Ciphertext::to_host_inplace] Cannot copy ciphertext with seed to host");
            }
            data_.to_host_inplace();
        }

        inline Ciphertext to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            Ciphertext result = clone(pool);
            result.to_device_inplace(pool);
            return result;
        }

        inline Ciphertext to_host() {
            Ciphertext result = clone(pool());
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

        void expand_seed(HeContextPointer context);

        void save(std::ostream& stream, HeContextPointer context) const;
        void load(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool());
        inline static Ciphertext load_new(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            Ciphertext result;
            result.load(stream, context, pool);
            return result;
        }
        size_t serialized_size(HeContextPointer context) const;

        void save_terms(std::ostream& stream, HeContextPointer context, const std::vector<size_t>& terms, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const;
        void load_terms(std::istream& stream, HeContextPointer context, const std::vector<size_t>& terms, MemoryPoolHandle pool = MemoryPool::GlobalPool());
        inline static Ciphertext load_terms_new(std::istream& stream, HeContextPointer context, const std::vector<size_t>& terms, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            Ciphertext result;
            result.load_terms(stream, context, terms, pool);
            return result;
        }
        size_t serialized_terms_size(HeContextPointer context, const std::vector<size_t>& terms) const;

    };

}