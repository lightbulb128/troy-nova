#pragma once
#include "box.h"

namespace troy {namespace utils {

    template <typename T>
    class DynamicArray {
        Array<T> inner;

        void resize_internal(size_t new_size, bool fill_extra_with_zeros, bool copy_data) {
            if (new_size == this->inner.size()) {
                return;
            }
            Array<T> new_inner;
            new_inner.len = new_size;
            new_inner.device = this->inner.device;
            new_inner.memory_pool_handle_ = this->inner.memory_pool_handle_;
            if (new_size > 0) {
                if (this->inner.device) {
                    new_inner.pointer = kernel_provider::malloc<T>(*this->inner.memory_pool_handle_, new_size);
                } else {
                    new_inner.pointer = reinterpret_cast<T*>(std::malloc(new_size * sizeof(T)));
                }
                size_t copy_length = std::min(this->inner.size(), new_size);
                if (copy_data) new_inner.slice(0, copy_length).copy_from_slice(this->inner.const_slice(0, copy_length));
                if (fill_extra_with_zeros && new_size > copy_length) new_inner.slice(copy_length, new_size).set_zero();
            }
            this->inner = std::move(new_inner);
        }

    public:

        MemoryPoolHandle pool() const { return inner.pool(); }
        size_t device_index() const { return inner.device_index(); }

        DynamicArray() : inner() {}
        DynamicArray(size_t count, bool device, MemoryPoolHandle pool = MemoryPool::GlobalPool()) : inner(count, device, pool) {}

        // In contrast to the constructor, this does not fill the array with zeros
        static DynamicArray<T> create_uninitialized(size_t count, bool device, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            DynamicArray<T> result;
            result.inner = Array<T>::create_uninitialized(count, device, pool);
            return result;
        }

        // move constructor
        DynamicArray(DynamicArray&& other): inner(std::move(other.inner)) {}

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

        __host__
        ConstSlice<T> const_slice(size_t begin, size_t end) const {
            return inner.const_slice(begin, end);
        }

        __host__
        Slice<T> slice(size_t begin, size_t end) {
            return inner.slice(begin, end);
        }

        __host__
        ConstSlice<T> const_reference() const {
            return inner.const_reference();
        }

        __host__
        Slice<T> reference() {
            return inner.reference();
        }

        __host__ __device__
        ConstSlice<T> detached_const_slice(size_t begin, size_t end) const {
            return inner.detached_const_slice(begin, end);
        }

        __host__ __device__
        Slice<T> detached_slice(size_t begin, size_t end) {
            return inner.detached_slice(begin, end);
        }

        __host__ __device__
        ConstSlice<T> detached_const_reference() const {
            return inner.detached_const_reference();
        }

        __host__ __device__
        Slice<T> detached_reference() {
            return inner.detached_reference();
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

        DynamicArray<T> clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            DynamicArray<T> result;
            result.inner = inner.clone(pool);
            return result;
        }

        DynamicArray<T> to_host() const {
            DynamicArray<T> result;
            result.inner = inner.to_host();
            return result;
        }

        DynamicArray<T> to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            DynamicArray<T> result;
            result.inner = inner.to_device(pool);
            return result;
        }

        const Array<T>& get_inner() const {
            return inner;
        }

        Array<T>& get_inner() {
            return inner;
        }

        void to_host_inplace() {
            inner.to_host_inplace();
        }

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            inner.to_device_inplace(pool);
        }

        void to_pool_inplace(MemoryPoolHandle pool) {
            inner.to_pool_inplace(pool);
        }

        void copy_from_slice(const ConstSlice<T>& slice) {
            inner.copy_from_slice(slice);
        }

        static DynamicArray<T> create_and_copy_from_slice(const ConstSlice<T>& slice, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            DynamicArray<T> result;
            result.inner = Array<T>::create_and_copy_from_slice(slice, pool);
            return result;
        }

        static DynamicArray<T> from_vector(std::vector<T>&& vec) {
            DynamicArray<T> result;
            result.inner = Array<T>::from_vector(std::move(vec));
            return result;
        }

        std::vector<T> to_vector() const {
            return inner.to_vector();
        }

        void resize(size_t new_size, bool copy_data = true) {
            resize_internal(new_size, true, copy_data);
        }

        void resize_uninitialized(size_t new_size, bool copy_data = true) {
            resize_internal(new_size, false, copy_data);
        }

        inline T* raw_pointer() {
            return inner.raw_pointer();
        }

        inline const T* raw_pointer() const {
            return inner.raw_pointer();
        }

    };

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const DynamicArray<T>& arr) {
        os << arr.const_reference();
        return os;
    }

}}