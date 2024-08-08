#pragma once
#include "encryption_parameters.h"
#include "he_context.h"
#include "utils/dynamic_array.h"
#include "utils/serialize.h"
#include "utils/poly_to_string.h"

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
        
        void resize_rns_internal(size_t poly_modulus_degree, size_t coeff_modulus_size, size_t coeff_count, bool fill_extra_with_zeros, bool copy_data);

        size_t save_raw(std::ostream& stream) const;
        void load_raw(std::istream& stream, MemoryPoolHandle pool);
        size_t serialized_raw_size() const;

    public:

        inline MemoryPoolHandle pool() const { return data_.pool(); }
        inline size_t device_index() const { return data_.device_index(); }

        inline Plaintext():
            coeff_count_(0),
            data_(0, false, nullptr),
            parms_id_(parms_id_zero),
            scale_(1.0),
            is_ntt_form_(false),
            coeff_modulus_size_(0),
            poly_modulus_degree_(0) {}

        inline Plaintext(Plaintext&& source) = default;
        inline Plaintext(const Plaintext& copy): Plaintext(copy.clone(copy.pool())) {}

        inline Plaintext& operator =(Plaintext&& source) = default;
        inline Plaintext& operator =(const Plaintext& assign) {
            if (this == &assign) {
                return *this;
            }
            *this = assign.clone(assign.pool());
            return *this;
        }

        static Plaintext like(const Plaintext& other, bool fill_zeros, MemoryPoolHandle pool = MemoryPool::GlobalPool());

        inline Plaintext clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext result;
            result.coeff_count_ = this->coeff_count_;
            result.data_ = this->data_.clone(pool);
            result.parms_id_ = this->parms_id_;
            result.scale_ = this->scale_;
            result.is_ntt_form_ = this->is_ntt_form_;
            result.coeff_modulus_size_ = this->coeff_modulus_size_;
            result.poly_modulus_degree_ = this->poly_modulus_degree_;
            return result;
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            this->data_.to_device_inplace(pool);
        }

        inline void to_host_inplace() {
            this->data_.to_host_inplace();
        }

        inline Plaintext to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Plaintext result = this->clone(pool);
            result.to_device_inplace(pool);
            return result;
        }

        inline Plaintext to_host() {
            Plaintext result = this->clone(this->pool());
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
            return coeff_count_;
        }

        inline size_t& coeff_count() {
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
                size_t start = index * this->coeff_count();
                return this->data_.const_slice(start, start + this->coeff_count());
            }
        }

        inline utils::Slice<uint64_t> component(size_t index) {
            if (this->parms_id_ == parms_id_zero) {
                if (index != 0) {
                    throw std::out_of_range("[Plaintext::component] Index out of range");
                }
                return this->poly();
            } else {
                size_t start = index * this->coeff_count();
                return this->data_.slice(start, start + this->coeff_count());
            }
        }

        inline utils::ConstSlice<uint64_t> const_component(size_t index) const {
            return this->component(index);
        }

        inline utils::ConstSlice<uint64_t> reference() const noexcept {return this->poly();}
        inline utils::Slice<uint64_t> reference() noexcept {return this->poly();}
        inline utils::ConstSlice<uint64_t> const_reference() const noexcept {return this->const_poly();}

        inline void resize(size_t coeff_count, bool fill_extra_with_zeros = true, bool copy_data = true) {
            if (this->parms_id_ != parms_id_zero) {
                throw std::invalid_argument("[Plaintext::resize] Cannot resize if the plaintext is not mod t. Call resize_rns instead.");
            }
            this->coeff_count_ = coeff_count;
            if (fill_extra_with_zeros) {
                this->data_.resize(coeff_count, copy_data);
            } else {
                this->data_.resize_uninitialized(coeff_count, copy_data);
            }
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

        void resize_rns(const HeContext& context, const ParmsID& parms_id, bool fill_extra_with_zeros = true, bool copy_data = true);
        void resize_rns_partial(const HeContext& context, const ParmsID& parms_id, size_t coeff_count, bool fill_extra_with_zeros = true, bool copy_data = true);

        inline bool is_ntt_form() const {
            return is_ntt_form_;
        }

        inline bool& is_ntt_form() {
            return is_ntt_form_;
        }

        inline size_t save(std::ostream& stream, CompressionMode mode = CompressionMode::Nil) const {
            return serialize::compress(stream, [this](std::ostream& stream){return this->save_raw(stream);}, mode);
        }
        inline void load(std::istream& stream, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            serialize::decompress(stream, [this, &pool](std::istream& stream){this->load_raw(stream, pool);});
        }
        inline static Plaintext load_new(std::istream& stream, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            Plaintext result;
            result.load(stream, pool);
            return result;
        }
        inline size_t serialized_size_upperbound(CompressionMode mode = CompressionMode::Nil) const {
            return serialize::serialized_size_upperbound(this->serialized_raw_size(), mode);
        }

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