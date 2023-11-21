#pragma once
#include <cstdint>
#include <memory>
#include "../kernel_provider.cuh"

namespace troy { namespace utils {

    template<class T>
    class ConstPointer {
        const T* pointer;
        bool device;
    public:
        __host__ __device__ ConstPointer(const T* pointer, bool device) : pointer(pointer), device(device) {}
        __host__ __device__ const T* operator->() const { return pointer; }
        __host__ __device__ const T& operator*() const { return *pointer; }
        __host__ __device__ const T* get() const { return pointer; }
        __host__ __device__ bool on_device() const { return device; }
    };

    template<class T>
    class Pointer {
        T* pointer;
        bool device;
    public:
        __host__ __device__ Pointer(T* pointer, bool device) : pointer(pointer), device(device) {}
        __host__ __device__ T* operator->() { return pointer; }
        __host__ __device__ T& operator*() { return *pointer; }
        __host__ __device__ T* get() { return pointer; }
        __host__ __device__ ConstPointer<T> as_const() const { return ConstPointer(pointer, device); } 
        __host__ __device__ bool on_device() const { return device; }
    };

    template<class T>
    class Box {
        T* pointer;
        bool device;
    public:

        Box(T* object, bool device) : pointer(object), device(device) {}
        Box(T&& object) : pointer(new T(std::move(object))), device(false) {}
        Box(Box&& other) : pointer(other.pointer), device(other.device) { other.pointer = nullptr; }

        __host__ __device__ bool on_device() const { return device; }
    
        ~Box() { 
            if (!pointer) return;
            if (!device) delete pointer;
            else kernel_provider::free(pointer);
        }

        Box& operator=(T&& object) = delete;
        Box(const Box&) = delete;
        Box& operator=(const Box&) = delete;
        
        Box clone() const {
            T cloned = *pointer;
            return Box(std::move(cloned), device);
        }

        Box to_host() const {
            if (!device) return this->clone();
            T* cloned = new T();
            kernel_provider::copy_device_to_host(&cloned, pointer, 1);
            return Box(cloned, false);
        }

        Box to_device() const {
            if (device) return this->clone();
            T* cloned = kernel_provider::malloc<T>(1);
            kernel_provider::copy_host_to_device(cloned, pointer, 1);
            return Box(cloned, true);
        }

        T* operator->() { return pointer; }
        const T* operator->() const { return pointer; }

        T& operator*() { return *pointer; }
        const T& operator*() const { return *pointer; }

        ConstPointer<T> as_const_pointer() const { return ConstPointer(pointer, device); }
        Pointer<T> as_pointer() { return Pointer(pointer, device); }

    };

    template<class T>
    class ConstSlice {
        const T* pointer;
        size_t len;
        bool device;
    public:
        __host__ __device__ ConstSlice(const T* pointer, size_t len, bool device) : pointer(pointer), len(len), device(device) {}
        __host__ __device__ size_t size() const { return len; }
        __host__ __device__ const T& operator[](size_t index) const { return pointer[index]; }
        __host__ __device__ ConstSlice<T> const_slice(size_t begin, size_t end) const { return ConstSlice<T>(pointer + begin, end - begin, device); }
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ const T* raw_pointer() const { return pointer; }
        __host__ __device__ static ConstSlice<T> from_pointer(ConstPointer<T> pointer) {
            return ConstSlice<T>(pointer.get(), 1, pointer.on_device());
        }
    };

    template<class T>
    class Slice {
        T* pointer;
        size_t len;
        bool device;
    public:
        __host__ __device__ Slice(T* pointer, size_t len, bool device) : pointer(pointer), len(len), device(device) {}
        __host__ __device__ size_t size() const { return len; }
        __host__ __device__ T& operator[](size_t index) { return pointer[index]; }
        __host__ __device__ ConstSlice<T> as_const() const { return ConstSlice<T>(pointer, len, device); }
        __host__ __device__ ConstSlice<T> const_slice(size_t begin, size_t end) const { return ConstSlice<T>(pointer + begin, end - begin, device); }
        __host__ __device__ Slice<T> slice(size_t begin, size_t end) { return Slice<T>(pointer + begin, end - begin, device); }
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ T* raw_pointer() { return pointer; }
        __host__ __device__ static Slice<T> from_pointer(Pointer<T> pointer) {
            return Slice<T>(pointer.get(), 1, pointer.on_device());
        }
    };

    template<class T>
    class Array {
        T* pointer;
        size_t len;
        bool device;
    public:

        Array(size_t count, bool device) : len(count), device(device) {
            if (device) {
                pointer = kernel_provider::malloc<T>(count);
                kernel_provider::memset_zero(pointer, count);
            } else {
                pointer = new T[count];
                memset(pointer, 0, count * sizeof(T));
            }
        }
        ~Array() { 
            if (!pointer) return;
            if (!device) delete[] pointer;
            else kernel_provider::free(pointer);
        }
        
        __host__ __device__ bool on_device() const { return device; }

        Array(Array&& other) : pointer(other.pointer), len(other.len), device(other.device) { other.pointer = nullptr; }

        Array& operator=(T&& object) = delete;
        Array(const Array&) = delete;
        Array& operator=(const Array&) = delete;

        __host__ __device__ size_t size() const { return len; }
        __host__ __device__ ConstSlice<T> const_slice(size_t begin, size_t end) const {
            return ConstSlice<T>(pointer + begin, end - begin, device);
        }
        __host__ __device__ Slice<T> slice(size_t begin, size_t end) {
            return Slice<T>(pointer + begin, end - begin, device);
        }
        __host__ __device__ ConstSlice<T> const_reference() const {
            return ConstSlice<T>(pointer, len, device);
        }
        __host__ __device__ Slice<T> reference() {
            return Slice<T>(pointer, len, device);
        }
        __host__ __device__ const T& operator[](size_t index) const { return pointer[index]; }
        __host__ __device__ T& operator[](size_t index) { return pointer[index]; }

        Array clone() const {
            Array cloned(len, device);
            if (device) {
                kernel_provider::copy_device_to_device(cloned.pointer, pointer, len);
            } else {
                memcpy(cloned.pointer, pointer, len * sizeof(T));
            }
            return cloned;
        }

        Array to_host() const {
            if (!device) return this->clone();
            Array cloned(len, false);
            kernel_provider::copy_device_to_host(cloned.pointer, pointer, len);
            return cloned;
        }

        Array to_device() const {
            if (device) return this->clone();
            Array cloned(len, true);
            kernel_provider::copy_host_to_device(cloned.pointer, pointer, len);
            return cloned;
        }

        inline void copy_from_slice(ConstSlice<T> slice) {
            if (slice.size() != len) throw std::runtime_error("Slice size does not match array size");
            if (slice.on_device() != device) throw std::runtime_error("Slice device does not match array device");
            if (device) {
                kernel_provider::copy_device_to_device(pointer, slice.raw_pointer(), len);
            } else {
                memcpy(pointer, slice.raw_pointer(), len * sizeof(T));
            }
        }

    };

}}