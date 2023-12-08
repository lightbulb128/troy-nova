#pragma once
#include "box.cuh"

namespace troy {namespace utils {

    template <typename T = uint64_t>
    class Buffer {
        size_t poly_count_; size_t coeff_modulus_size_; size_t coeff_count_;
        Array<T> data_;
    public:

        Buffer(size_t poly_count, size_t coeff_modulus_size, size_t coeff_count, bool device) :
            poly_count_(poly_count), coeff_modulus_size_(coeff_modulus_size), coeff_count_(coeff_count),
            data_(poly_count * coeff_modulus_size * coeff_count, device) {}
        Buffer(size_t coeff_modulus_size, size_t coeff_count, bool device): 
            Buffer(1, coeff_modulus_size, coeff_count, device) {}
        Buffer(size_t coeff_count, bool device): Buffer(1, 1, coeff_count, device) {}

        void resize(size_t poly_count, size_t coeff_modulus_size, size_t coeff_count) {
            poly_count_ = poly_count;
            coeff_modulus_size_ = coeff_modulus_size;
            coeff_count_ = coeff_count;
            data_ = Array<T>(poly_count * coeff_modulus_size * coeff_count, data_.on_device());
        }
        void resize(size_t coeff_modulus_size, size_t coeff_count) {
            resize(1, coeff_modulus_size, coeff_count);
        }
        void resize(size_t coeff_count) {
            resize(1, 1, coeff_count);
        }

        size_t size() const {return data_.size();}
        size_t poly_count() const {return poly_count_;}
        size_t coeff_modulus_size() const {return coeff_modulus_size_;}
        size_t coeff_count() const {return coeff_count_;}

        Buffer clone() const {
            Buffer clone(poly_count_, coeff_modulus_size_, coeff_count_, data_.on_device());
            clone.data_.copy_from_slice(data_);
            return clone;
        }

        void copy_from_slice(ConstSlice<T> slice) {
            data_.copy_from_slice(slice);
        }

        ConstSlice<T> const_reference() const {return data_.const_reference();}
        Slice<T> reference() {return data_.reference();}

        ConstSlice<T> const_poly(size_t poly_id) const {
            return data_.const_slice(
                poly_id * coeff_modulus_size_ * coeff_count_, 
                (poly_id + 1) * coeff_modulus_size_ * coeff_count_);
        }
        Slice<T> poly(size_t poly_id) {
            return data_.slice(
                poly_id * coeff_modulus_size_ * coeff_count_, 
                (poly_id + 1) * coeff_modulus_size_ * coeff_count_);
        }

        ConstSlice<T> const_polys(size_t start, size_t end) const {
            return data_.const_slice(
                start * coeff_modulus_size_ * coeff_count_, 
                end * coeff_modulus_size_ * coeff_count_);
        }
        Slice<T> polys(size_t start, size_t end) {
            return data_.slice(
                start * coeff_modulus_size_ * coeff_count_, 
                end * coeff_modulus_size_ * coeff_count_);
        }

        ConstSlice<T> const_component(size_t poly_id, size_t coeff_modulus_id) const {
            size_t offset = (poly_id * coeff_modulus_size_ + coeff_modulus_id) * coeff_count_;
            return data_.const_slice(offset, offset + coeff_count_);
        }
        Slice<T> component(size_t poly_id, size_t coeff_modulus_id) {
            size_t offset = (poly_id * coeff_modulus_size_ + coeff_modulus_id) * coeff_count_;
            return data_.slice(offset, offset + coeff_count_);
        }

        ConstSlice<T> const_component(size_t coeff_modulus_id) const {
            if (poly_count_ != 1) {
                throw std::logic_error("[Buffer::component] Poly count is not 1.");
            }
            return data_.const_slice(coeff_modulus_id * coeff_count_, (coeff_modulus_id + 1) * coeff_count_);
        }
        Slice<T> component(size_t coeff_modulus_id) {
            if (poly_count_ != 1) {
                throw std::logic_error("[Buffer::component] Poly count is not 1.");
            }
            return data_.slice(coeff_modulus_id * coeff_count_, (coeff_modulus_id + 1) * coeff_count_);
        }

        ConstSlice<T> const_components(size_t start, size_t end) const {
            if (poly_count_ != 1) {
                throw std::logic_error("[Buffer::components] Poly count is not 1.");
            }
            return data_.const_slice(start * coeff_count_, end * coeff_count_);
        }
        Slice<T> components(size_t start, size_t end) {
            if (poly_count_ != 1) {
                throw std::logic_error("[Buffer::components] Poly count is not 1.");
            }
            return data_.slice(start * coeff_count_, end * coeff_count_);
        }

        ConstSlice<T> const_slice(size_t start, size_t end) const {
            return data_.const_slice(start, end);
        }
        Slice<T> slice(size_t start, size_t end) {
            return data_.slice(start, end);
        }
        
    };

    template <typename T = uint64_t>
    std::ostream& operator<<(std::ostream& os, const Buffer<T>& buffer) {
        os << buffer.const_reference();
        return os;
    }

}}