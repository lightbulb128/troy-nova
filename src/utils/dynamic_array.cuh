#pragma once
#include "box.cuh"

namespace troy {namespace utils {

    template <typename T>
    class DynamicArray {
        Array<T> inner;

    public:

        DynamicArray() : inner() {}
        DynamicArray(size_t count, bool device) : inner(count, device) {}

        // move constructor
        DynamicArray(DynamicArray&& other) : inner(std::move(other.inner)) {}

        // copy constructor deleted
        DynamicArray(const DynamicArray& other) = delete;

        // move assignment operator
        DynamicArray& operator=(DynamicArray&& other) {
            inner = std::move(other.inner);
            return *this;
        }

        // copy assignment operator deleted
        DynamicArray& operator=(const DynamicArray& other) = delete;

        __host__ __device__ bool on_device() const {
            return inner.on_device();
        }

        __host__ __device__ size_t size() const {
            return inner.size();
        }

        __host__ __device__
        ConstSlice<T> const_slice(size_t begin, size_t end) const {
            return inner.const_slice(begin, end);
        }

        __host__ __device__
        Slice<T> slice(size_t begin, size_t end) {
            return inner.slice(begin, end);
        }

        __host__ __device__
        ConstSlice<T> const_reference() const {
            return inner.const_reference();
        }

        __host__ __device__
        Slice<T> reference() {
            return inner.reference();
        }

        __host__ __device__ const T& operator[](size_t index) const {
            return inner[index];
        }

        __host__ __device__ T& operator[](size_t index) {
            return inner[index];
        }

        __host__ __device__ Pointer<T> at(size_t index) {
            return inner.at(index);
        }

        __host__ __device__ ConstPointer<T> const_at(size_t index) const {
            return inner.at(index);
        }

        DynamicArray<T> clone() const {
            DynamicArray<T> result;
            result.inner = inner.clone();
            return result;
        }

        DynamicArray<T> to_host() const {
            DynamicArray<T> result;
            result.inner = inner.to_host();
            return result;
        }

        DynamicArray<T> to_device() const {
            DynamicArray<T> result;
            result.inner = inner.to_device();
            return result;
        }

        void to_host_inplace() {
            inner.to_host_inplace();
        }

        void to_device_inplace() {
            inner.to_device_inplace();
        }

        void copy_from_slice(const ConstSlice<T>& slice) {
            inner.copy_from_slice(slice);
        }

        static DynamicArray<T> create_and_copy_from_slice(const ConstSlice<T>& slice) {
            DynamicArray<T> result;
            result.inner = Array<T>::create_and_copy_from_slice(slice);
            return result;
        }

        static DynamicArray<T> from_vector(const std::vector<T>& vec) {
            DynamicArray<T> result;
            result.inner = Array<T>::from_vector(vec);
            return result;
        }

        std::vector<T> to_vector() const {
            return inner.to_vector();
        }

        void resize(size_t new_size) {
            if (new_size == this->inner.size()) {
                return;
            }
            Array<T> new_inner(new_size, this->on_device());
            size_t copy_length = std::min(this->inner.size(), new_size);
            new_inner.slice(0, copy_length).copy_from_slice(this->inner.const_slice(0, copy_length));
            this->inner = std::move(new_inner);
        }

    };

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const DynamicArray<T>& arr) {
        os << arr.const_reference();
        return os;
    }

}}