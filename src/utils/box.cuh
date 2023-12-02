#pragma once
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include "../kernel_provider.cuh"

namespace troy { namespace utils {

    template<class T>
    class ConstPointer {
        const T* pointer;
        bool device;
    public:
        __host__ __device__ ConstPointer(const T* pointer, bool device) : pointer(pointer), device(device) {}
        __host__ __device__ const T* operator->() const { 
            #ifndef __CUDA_ARCH__
                if (!pointer) throw std::runtime_error("[ConstPointer::operator->] Null pointer");
            #endif
            return pointer; 
        }
        __host__ __device__ const T& operator*() const { 
            #ifndef __CUDA_ARCH__
                if (!pointer) throw std::runtime_error("[ConstPointer::operator*] Null pointer");
            #endif
            return *pointer; 
        }
        __host__ __device__ const T* get() const { return pointer; }
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ bool is_null() const { return pointer == nullptr; }
        __host__ __device__ static ConstPointer<T> from_reference(const T& reference, bool device) {
            return ConstPointer<T>(&reference, device);
        }
    };

    template<class T>
    class Pointer {
        T* pointer;
        bool device;
    public:
        __host__ __device__ Pointer(T* pointer, bool device) : pointer(pointer), device(device) {}
        __host__ __device__ T* operator->() { 
            #ifndef __CUDA_ARCH__
                if (!pointer) throw std::runtime_error("[Pointer::operator->] Null pointer");
            #endif
            return pointer; 
        }
        __host__ __device__ T& operator*() { 
            #ifndef __CUDA_ARCH__
                if (!pointer) throw std::runtime_error("[Pointer::operator*] Null pointer");
            #endif
            return *pointer; 
        }
        __host__ __device__ T* get() { return pointer; }
        __host__ __device__ ConstPointer<T> as_const() const { return ConstPointer(pointer, device); } 
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ bool is_null() const { return pointer == nullptr; }
        __host__ __device__ static Pointer<T> from_reference(T& reference, bool device) {
            return Pointer<T>(&reference, device);
        }
    };

    template<class T>
    class Box {
        T* pointer;
        bool device;
    public:

        Box(): pointer(nullptr), device(false) {}
        Box(T* object, bool device) : pointer(object), device(device) {}
        Box(Box&& other) : pointer(other.pointer), device(other.device) { other.pointer = nullptr; }

        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ bool is_null() const { return pointer == nullptr; }
        __host__ __device__ T* raw_pointer() { return pointer; }

        inline void release() {
            if (!pointer) return;
            if (!device) free(pointer);
            else kernel_provider::free(pointer);
            pointer = nullptr;
        }
    
        ~Box() { 
            release();
        }

        Box& operator=(Box<T>&& object) {
            release();
            pointer = object.pointer;
            device = object.device;
            object.pointer = nullptr;
            return *this;
        }

        Box(const Box&) = delete;
        Box& operator=(const Box&) = delete;
        
        Box clone() const {
            if (!device) {
                T* cloned = reinterpret_cast<T*>(malloc(sizeof(T)));
                memcpy(cloned, pointer, sizeof(T));
                return Box(cloned, device);
            } else {
                T* cloned = kernel_provider::malloc<T>(1);
                kernel_provider::copy_device_to_device(cloned, pointer, 1);
                return Box(cloned, device);
            }
        }

        Box to_host() const {
            if (!device) return this->clone();
            T* cloned = reinterpret_cast<T*>(malloc(sizeof(T)));
            kernel_provider::copy_device_to_host(&cloned, pointer, 1);
            return Box(cloned, false);
        }

        Box to_device() const {
            if (device) return this->clone();
            T* cloned = kernel_provider::malloc<T>(1);
            kernel_provider::copy_host_to_device(cloned, pointer, 1);
            return Box(cloned, true);
        }

        void to_host_inplace() {
            if (!device) return;
            T* cloned = reinterpret_cast<T*>(malloc(sizeof(T)));
            kernel_provider::copy_device_to_host(&cloned, pointer, 1);
            release();
            pointer = cloned;
            device = false;
        }

        void to_device_inplace() {
            if (device) return;
            T* cloned = kernel_provider::malloc<T>(1);
            kernel_provider::copy_host_to_device(cloned, pointer, 1);
            release();
            pointer = cloned;
            device = true;
        }

        T* operator->() { 
            #ifndef __CUDA_ARCH__
                if (!pointer) throw std::runtime_error("[Box::operator->] Null pointer");
            #endif
            return pointer; 
        }
        const T* operator->() const { 
            #ifndef __CUDA_ARCH__
                if (!pointer) throw std::runtime_error("[Box::operator->] Null pointer");
            #endif
            return pointer; 
        }

        T& operator*() { return *pointer; }
        const T& operator*() const { return *pointer; }

        ConstPointer<T> as_const_pointer() const { return ConstPointer(pointer, device); }
        Pointer<T> as_pointer() { return Pointer(pointer, device); }

    };

    // template<class T> class ConstSlice;

    // template<class T>
    // class Array {
    //     static Array<T> create_and_copy_from_slice(ConstSlice<T> slice);
    //     void to_host_inplace();
    //     ConstSlice<T> const_reference() const;
    // };

    template<class T>
    class ConstSlice {
        const T* pointer;
        size_t len;
        bool device;
    public:
        __host__ __device__ ConstSlice(const T* pointer, size_t len, bool device) : pointer(pointer), len(len), device(device) {}
        __host__ __device__ size_t size() const { return len; }
        __host__ __device__ const T& operator[](size_t index) const { return pointer[index]; }
        __host__ __device__ ConstPointer<T> at(size_t index) const { return ConstPointer<T>(pointer + index, device); }
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
        __host__ __device__ Pointer<T> at(size_t index) { return Pointer<T>(pointer + index, device); }
        __host__ __device__ ConstPointer<T> const_at(size_t index) const { return ConstPointer<T>(pointer + index, device); }
        __host__ __device__ ConstSlice<T> as_const() const { return ConstSlice<T>(pointer, len, device); }
        __host__ __device__ ConstSlice<T> const_slice(size_t begin, size_t end) const { return ConstSlice<T>(pointer + begin, end - begin, device); }
        __host__ __device__ Slice<T> slice(size_t begin, size_t end) { return Slice<T>(pointer + begin, end - begin, device); }
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ T* raw_pointer() { return pointer; }
        __host__ __device__ static Slice<T> from_pointer(Pointer<T> pointer) {
            return Slice<T>(pointer.get(), 1, pointer.on_device());
        }
        void copy_from_slice(ConstSlice<T> slice) {
            if (slice.size() != len) throw std::runtime_error("[Slice::copy_from_slice] Slice size does not match array size");
            if (device && slice.on_device()) {
                kernel_provider::copy_device_to_device(pointer, slice.raw_pointer(), len);
            } else if (!device && !slice.on_device()) {
                memcpy(pointer, slice.raw_pointer(), len * sizeof(T));
            } else if (device && !slice.on_device()) {
                kernel_provider::copy_host_to_device(pointer, slice.raw_pointer(), len);
            } else {
                kernel_provider::copy_device_to_host(pointer, slice.raw_pointer(), len);
            }
        }
        void set_zero() {
            if (len == 0) return;
            if (device) {
                kernel_provider::memset_zero(pointer, len);
            } else {
                memset(pointer, 0, len * sizeof(T));
            }
        }
    };

    template<class T>
    class Array {
        T* pointer;
        size_t len;
        bool device;
    public:

        Array() : len(0), device(false), pointer(nullptr) {}
        Array(size_t count, bool device) : len(count), device(device) {
            if (count == 0) {
                pointer = nullptr;
                return;
            }
            if (device) {
                pointer = kernel_provider::malloc<T>(count);
                kernel_provider::memset_zero(pointer, count);
            } else {
                pointer = reinterpret_cast<T*>(malloc(count * sizeof(T)));
                memset(pointer, 0, count * sizeof(T));
            }
        }

        inline void release() {
            if (!pointer) return;
            if (!device) free(pointer);
            else kernel_provider::free(pointer);
        }
        ~Array() { 
            release();
        }

        Array& operator=(Array&& other) {
            release();
            pointer = other.pointer;
            len = other.len;
            device = other.device;
            other.pointer = nullptr;
            other.len = 0;
            return *this;
        }
        
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ T* raw_pointer() { return pointer; }

        Array(Array&& other) : pointer(other.pointer), len(other.len), device(other.device) { 
            other.pointer = nullptr;
            other.len = 0; 
        }

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
        __host__ __device__ Pointer<T> at(size_t index) { return Pointer<T>(pointer + index, device); }
        __host__ __device__ ConstPointer<T> const_at(size_t index) const { return ConstPointer<T>(pointer + index, device); }

        inline Array clone() const {
            Array cloned(len, device);
            if (pointer && len > 0) {
                if (device) {
                    kernel_provider::copy_device_to_device(cloned.pointer, pointer, len);
                } else {
                    memcpy(cloned.pointer, pointer, len * sizeof(T));
                }
            }
            return cloned;
        }

        inline Array to_host() const {
            if (!device) return this->clone();
            Array cloned(len, false);
            if (len > 0) {
                kernel_provider::copy_device_to_host(cloned.pointer, pointer, len);
            }
            return cloned;
        }

        inline Array to_device() const {
            if (device) return this->clone();
            Array cloned(len, true);
            if (len > 0) {
                kernel_provider::copy_host_to_device(cloned.pointer, pointer, len);
            }
            return cloned;
        }

        inline void to_host_inplace() {
            if (!device) return;
            if (len > 0) {
                T* cloned = reinterpret_cast<T*>(malloc(len * sizeof(T)));
                kernel_provider::copy_device_to_host(cloned, pointer, len);
                release();
                pointer = cloned;
            }
            device = false;
        }

        inline void to_device_inplace() {
            if (device) return;
            if (len > 0) {
                T* cloned = kernel_provider::malloc<T>(len);
                kernel_provider::copy_host_to_device(cloned, pointer, len);
                release();
                pointer = cloned;
            }
            device = true;
        }

        inline void copy_from_slice(ConstSlice<T> slice) {
            this->reference().copy_from_slice(slice);
        }

        inline static Array<T> create_and_copy_from_slice(ConstSlice<T> slice) {
            Array<T> array(slice.size(), slice.on_device());
            array.copy_from_slice(slice);
            return array;
        }

        inline static Array<T> from_vector(std::vector<T>&& vector) {
            Array<T> array(vector.size(), false);
            if (vector.size() > 0) {
                memcpy(array.pointer, vector.data(), vector.size() * sizeof(T));
            }
            return array;
        }

        inline std::vector<T> to_vector() const {
            if (device) {
                Array<T> host_array = this->to_host();
                return host_array.to_vector();
            }
            std::vector<T> vector(len);
            if (len > 0) {
                memcpy(vector.data(), pointer, len * sizeof(T));
            }
            return vector;
        }

        void set_zero() {
            this->reference().set_zero();
        }

    };

    template<class T>
    std::ostream& operator<<(std::ostream& os, const ConstSlice<T>& slice) {
        if (slice.on_device()) {
            os << "device";
            Array<T> host_array = Array<T>::create_and_copy_from_slice(slice);
            host_array.to_host_inplace();
            os << host_array.const_reference();
        } else {
            os << "[";
            for (size_t i = 0; i < slice.size(); i++) {
                std::cout << slice[i];
                if (i != slice.size() - 1) os << ", ";
            }
            os << "]";
        }
        return os;
    }

    template<class T>
    std::ostream& operator<<(std::ostream& os, const Slice<T>& slice) {
        os << slice.as_const();
        return os;
    }

    template<class T>
    std::ostream& operator<<(std::ostream& os, const Array<T>& array) {
        os << array.const_reference();
        return os;
    }

}}