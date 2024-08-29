#pragma once
#include "box.h"

namespace troy {namespace utils {

    template <typename T = uint64_t>
    class Buffer {
        size_t poly_count_; size_t coeff_modulus_size_; size_t coeff_count_;
        Array<T> data_;
    public:

        MemoryPoolHandle pool() const { return data_.pool(); }
        size_t device_index() const { return data_.device_index(); }

        Buffer(size_t poly_count, size_t coeff_modulus_size, size_t coeff_count, bool device, MemoryPoolHandle pool) :
            poly_count_(poly_count), coeff_modulus_size_(coeff_modulus_size), coeff_count_(coeff_count),
            data_(Array<T>::create_uninitialized(poly_count * coeff_modulus_size * coeff_count, device, pool)) {}
        Buffer(size_t coeff_modulus_size, size_t coeff_count, bool device, MemoryPoolHandle pool): 
            Buffer(1, coeff_modulus_size, coeff_count, device, pool) {}
        Buffer(size_t coeff_count, bool device, MemoryPoolHandle pool): Buffer(1, 1, coeff_count, device, pool) {}
        Buffer(): Buffer(0, false, nullptr) {}

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

        bool on_device() const {return data_.on_device();}
        void to_device_inplace() {data_.to_device_inplace();}
        void to_host_inplace() {data_.to_host_inplace();}
        void to_pool_inplace(MemoryPoolHandle pool) {
            data_.to_pool_inplace(pool);
        }

        size_t size() const {return data_.size();}
        size_t poly_count() const {return poly_count_;}
        size_t coeff_modulus_size() const {return coeff_modulus_size_;}
        size_t coeff_count() const {return coeff_count_;}

        Buffer clone(MemoryPoolHandle pool) const {
            Buffer clone(poly_count_, coeff_modulus_size_, coeff_count_, data_.on_device(), pool);
            clone.data_.copy_from_slice(data_);
            return clone;
        }
        
        Buffer<T> to_device(MemoryPoolHandle pool) const {
            Buffer<T> cloned = this->clone(pool);
            cloned.to_device_inplace();
            return cloned;
        }
        Buffer<T> to_host() const {
            Buffer<T> cloned(this->poly_count_, this->coeff_modulus_size_, this->coeff_count_, false, nullptr);
            cloned.data_.copy_from_slice(data_.const_reference());
            return cloned;
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

        const uint64_t& operator[](size_t idx) const {return data_[idx];}
        uint64_t& operator[](size_t idx) {return data_[idx];}

        void set_zero() {data_.set_zero();}
        
    };

    template <typename T = uint64_t>
    std::ostream& operator<<(std::ostream& os, const Buffer<T>& buffer) {
        os << buffer.const_reference();
        return os;
    }

}}