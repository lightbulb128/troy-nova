#pragma once
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include "../kernel_provider.h"

namespace troy { namespace utils {

// allow class-memaccess for gcc
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"

    template<class T>
    class ConstPointer {
        const T* pointer;
        bool device;
        MemoryPool* memory_pool_handle_;
    public:
        __host__ __device__ ConstPointer(const T* pointer, bool device, MemoryPool* memory_pool_handle)
            : pointer(pointer), device(device), memory_pool_handle_(memory_pool_handle) {}
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
        __host__ __device__ static ConstPointer<T> from_reference(const T& reference, bool device, MemoryPoolHandle memory_pool_handle) {
            return ConstPointer<T>(&reference, device, memory_pool_handle);
        }
        __host__ __device__ MemoryPool* pool() const { return memory_pool_handle_; }
        size_t device_index() const { return memory_pool_handle_->get_device(); }
    };

    template<class T>
    class Pointer {
        T* pointer;
        bool device;
        MemoryPool* memory_pool_handle_;
    public:
        __host__ __device__ Pointer(T* pointer, bool device, MemoryPool* memory_pool_handle)
            : pointer(pointer), device(device), memory_pool_handle_(memory_pool_handle) {}
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
        __host__ __device__ ConstPointer<T> as_const() const { return ConstPointer(pointer, device, memory_pool_handle_); } 
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ bool is_null() const { return pointer == nullptr; }
        __host__ __device__ static Pointer<T> from_reference(T& reference, bool device, MemoryPoolHandle memory_pool_handle) {
            return Pointer<T>(&reference, device, memory_pool_handle);
        }
        MemoryPool* pool() const { return memory_pool_handle_; }
        size_t device_index() const { return memory_pool_handle_->get_device(); }
    };

    template<class T>
    class Box {
        T* pointer;
        bool device;
        MemoryPoolHandle memory_pool_handle_;
    public:

        MemoryPoolHandle pool() const { return memory_pool_handle_; }
        size_t device_index() const { return memory_pool_handle_->get_device(); }

        Box(): pointer(nullptr), device(false), memory_pool_handle_(nullptr) {}
        Box(T* object, bool device, MemoryPoolHandle memory_pool_handle = MemoryPool::GlobalPool()) : pointer(object), device(device), memory_pool_handle_(device ? memory_pool_handle : nullptr) {
            if (device && !memory_pool_handle_) throw std::runtime_error("[Box::Box] Memory pool handle is required for device memory");
        }
        Box(Box&& other): pointer(other.pointer), device(other.device), memory_pool_handle_(other.memory_pool_handle_) { other.pointer = nullptr;}

        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ bool is_null() const { return pointer == nullptr; }
        __host__ __device__ T* raw_pointer() { return pointer; }

        inline void release() {
            if (!pointer) return;
            if (!device) free(pointer);
            else {
                if (!memory_pool_handle_) {
                    throw std::runtime_error("[Box::release] Memory pool handle is required for device memory");
                }
                kernel_provider::free(*memory_pool_handle_, pointer);
            }
            pointer = nullptr;
            memory_pool_handle_ = nullptr;
        }
    
        ~Box() { 
            release();
        }

        Box& operator=(Box<T>&& object) {
            release();
            pointer = object.pointer;
            device = object.device;
            memory_pool_handle_ = object.memory_pool_handle_;
            object.pointer = nullptr;
            return *this;
        }

        Box(const Box&) = delete;
        Box& operator=(const Box&) = delete;
        
        Box clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (!device) {
                T* cloned = reinterpret_cast<T*>(malloc(sizeof(T)));
                memcpy(cloned, pointer, sizeof(T));
                return Box(cloned, false, nullptr);
            } else {
                if (!pool) throw std::runtime_error("[Box::clone] Memory pool handle is required for device memory");
                T* cloned = kernel_provider::malloc<T>(*pool, 1);
                kernel_provider::copy_device_to_device(*pool, cloned, pointer, 1);
                return Box(cloned, true, pool);
            }
        }

        Box to_host() const {
            if (!device) return this->clone(memory_pool_handle_);
            if (!memory_pool_handle_) throw std::runtime_error("[Box::to_host] Memory pool handle is required for device memory");
            T* cloned = reinterpret_cast<T*>(malloc(sizeof(T)));
            kernel_provider::copy_device_to_host(*memory_pool_handle_, &cloned, pointer, 1);
            return Box(cloned, false, nullptr);
        }

        Box to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (!pool) throw std::runtime_error("[Box::to_device] Memory pool handle is required for device memory");
            if (device) return this->clone(pool);
            T* cloned = kernel_provider::malloc<T>(*pool, 1);
            kernel_provider::copy_host_to_device(*pool, cloned, pointer, 1);
            return Box(cloned, true, pool);
        }

        void to_host_inplace() {
            if (!device) return;
            T* cloned = reinterpret_cast<T*>(malloc(sizeof(T)));
            if (!memory_pool_handle_) throw std::runtime_error("[Box::to_host_inplace] Memory pool handle is required for device memory");
            kernel_provider::copy_device_to_host(*memory_pool_handle_, cloned, pointer, 1);
            release();
            pointer = cloned;
            device = false;
            memory_pool_handle_ = nullptr;
        }

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            if (device) {
                if (pool != memory_pool_handle_) to_pool_inplace(pool);
                return;
            }
            if (!pool) throw std::runtime_error("[Box::to_device_inplace] Memory pool handle is required for device memory");
            T* cloned = kernel_provider::malloc<T>(*pool, 1);
            kernel_provider::copy_host_to_device(*pool, cloned, pointer, 1);
            release();
            pointer = cloned;
            device = true;
            memory_pool_handle_ = pool;
        }

        void to_pool_inplace(MemoryPoolHandle pool) {
            if (!device) {
                return;
            } else {
                if (!pool) throw std::runtime_error("[Box::to_pool_inplace] Memory pool handle is required for device memory");
                T* cloned = kernel_provider::malloc<T>(*pool, 1);
                kernel_provider::copy_device_to_device(*pool, cloned, pointer, 1);
                release();
                pointer = cloned;
                memory_pool_handle_ = pool;
            }
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

        ConstPointer<T> as_const_pointer() const { return ConstPointer(pointer, device, memory_pool_handle_.get()); }
        Pointer<T> as_pointer() { return Pointer(pointer, device, memory_pool_handle_.get()); }

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
        MemoryPool* memory_pool_handle_;
    public:
    
        MemoryPool* pool() const { return memory_pool_handle_; }
        size_t device_index() const { return memory_pool_handle_->get_device(); }

        __host__ __device__ ConstSlice(std::nullptr_t) : pointer(nullptr), len(0), device(false), memory_pool_handle_(nullptr) {}

        __host__ __device__ ConstSlice(ConstPointer<T> pointer) 
            : pointer(pointer.get()), len(1), device(pointer.on_device()), memory_pool_handle_(pointer.pool()) {}
        __host__ __device__ ConstSlice(Pointer<T> pointer) 
            : pointer(pointer.get()), len(1), device(pointer.on_device()), memory_pool_handle_(pointer.pool()) {}

        __host__ __device__ ConstSlice(const T* pointer, size_t len, bool device, MemoryPool* memory_pool_handle) 
            : pointer(pointer), len(len), device(device), memory_pool_handle_(memory_pool_handle) {}
        __host__ __device__ size_t size() const { return len; }
        __host__ __device__ const T& operator[](size_t index) const { return pointer[index]; }
        __host__ __device__ ConstPointer<T> at(size_t index) const { return ConstPointer<T>(pointer + index, device, memory_pool_handle_); }
        __host__ __device__ ConstSlice<T> const_slice(size_t begin, size_t end) const { return ConstSlice<T>(pointer + begin, end - begin, device, memory_pool_handle_); }
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ const T* raw_pointer() const { return pointer; }
        __host__ __device__ static ConstSlice<T> from_pointer(ConstPointer<T> pointer) {
            return ConstSlice<T>(pointer.get(), 1, pointer.on_device(), pointer.pool());
        }

    };

    template<class T>
    class Slice {
        T* pointer;
        size_t len;
        bool device;
        MemoryPool* memory_pool_handle_;
    public:

        MemoryPool* pool() const { return memory_pool_handle_; }
        size_t device_index() const { return memory_pool_handle_->get_device(); }

        __host__ __device__ Slice(std::nullptr_t) : pointer(nullptr), len(0), device(false), memory_pool_handle_(nullptr) {}

        __host__ __device__ Slice(Pointer<T> pointer) 
            : pointer(pointer.get()), len(1), device(pointer.on_device()), memory_pool_handle_(pointer.pool()) {}

        __host__ __device__ Slice(T* pointer, size_t len, bool device, MemoryPool* memory_pool_handle) 
            : pointer(pointer), len(len), device(device), memory_pool_handle_(memory_pool_handle) {}
        __host__ __device__ size_t size() const { return len; }
        __host__ __device__ T& operator[](size_t index) { return pointer[index]; }
        __host__ __device__ Pointer<T> at(size_t index) { return Pointer<T>(pointer + index, device, memory_pool_handle_); }
        __host__ __device__ ConstPointer<T> const_at(size_t index) const { return ConstPointer<T>(pointer + index, device, memory_pool_handle_); }
        __host__ __device__ ConstSlice<T> as_const() const { return ConstSlice<T>(pointer, len, device, memory_pool_handle_); }
        __host__ __device__ operator ConstSlice<T>() const { return ConstSlice<T>(pointer, len, device, memory_pool_handle_); }
        __host__ __device__ ConstSlice<T> const_slice(size_t begin, size_t end) const { return ConstSlice<T>(pointer + begin, end - begin, device, memory_pool_handle_); }
        __host__ __device__ Slice<T> slice(size_t begin, size_t end) { return Slice<T>(pointer + begin, end - begin, device, memory_pool_handle_); }
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ T* raw_pointer() const { return pointer; }
        __host__ __device__ static Slice<T> from_pointer(Pointer<T> pointer) {
            return Slice<T>(pointer.get(), 1, pointer.on_device(), pointer.pool());
        }
        void copy_from_slice(ConstSlice<T> slice) const {
            if (slice.size() != len) throw std::runtime_error("[Slice::copy_from_slice] Slice size does not match array size");
            if (!device && !slice.on_device()) {
                memcpy(pointer, slice.raw_pointer(), len * sizeof(T));
            } else if (device && slice.on_device()) {
                if (!memory_pool_handle_) throw std::runtime_error("[Slice::copy_from_slice] Memory pool handle is required for device memory");
                kernel_provider::copy_device_to_device(*memory_pool_handle_, pointer, slice.raw_pointer(), len);
            } else if (device && !slice.on_device()) {
                if (!memory_pool_handle_) throw std::runtime_error("[Slice::copy_from_slice] Memory pool handle is required for device memory");
                kernel_provider::copy_host_to_device(*memory_pool_handle_, pointer, slice.raw_pointer(), len);
            } else {
                if (!slice.pool()) throw std::runtime_error("[Slice::copy_from_slice] Memory pool handle is required for device memory");
                kernel_provider::copy_device_to_host(*slice.pool(), pointer, slice.raw_pointer(), len);
            }
        }
        void set_zero() {
            if (len == 0) return;
            if (device) {
                if (!memory_pool_handle_) throw std::runtime_error("[Slice::set_zero] Memory pool handle is required for device memory");
                kernel_provider::memset_zero(*memory_pool_handle_, pointer, len);
            } else {
                memset(pointer, 0, len * sizeof(T));
            }
        }
    };

    template<class T>
    class Array {

        template <typename DT> friend class DynamicArray;

        T* pointer;
        size_t len;
        bool device;
        MemoryPoolHandle memory_pool_handle_;
    public:

        MemoryPoolHandle pool() const { return memory_pool_handle_; }
        size_t device_index() const { return memory_pool_handle_->get_device(); }

        Array() : len(0), device(false), pointer(nullptr), memory_pool_handle_(nullptr) {}
        Array(size_t count, bool device, MemoryPoolHandle memory_pool_handle = MemoryPool::GlobalPool()) : len(count), device(device), memory_pool_handle_(device ? memory_pool_handle : nullptr) {
            if (device && !memory_pool_handle) throw std::runtime_error("[Array::Array] Memory pool handle is required for device memory");
            if (count == 0) {
                pointer = nullptr;
                return;
            }
            if (device) {
                pointer = kernel_provider::malloc<T>(*memory_pool_handle, count);
                kernel_provider::memset_zero(*memory_pool_handle, pointer, count);
            } else {
                pointer = reinterpret_cast<T*>(malloc(count * sizeof(T)));
                memset(pointer, 0, count * sizeof(T));
            }
        }

        // In contrast to the constructor, this function does not initialize the memory by filling with zeros.
        // This is useful when the memory will be overwritten immediately.
        static Array<T> create_uninitialized(size_t count, bool device, MemoryPoolHandle memory_pool_handle = MemoryPool::GlobalPool()) {
            if (device && !memory_pool_handle) throw std::runtime_error("[Array::create_uninitialized] Memory pool handle is required for device memory");
            Array<T> array;
            array.len = count;
            array.device = device;
            array.memory_pool_handle_ = device ? memory_pool_handle : nullptr;
            if (count == 0) {
                array.pointer = nullptr;
                return array;
            }
            if (device) {
                array.pointer = kernel_provider::malloc<T>(*memory_pool_handle, count);
            } else {
                array.pointer = reinterpret_cast<T*>(malloc(count * sizeof(T)));
            }
            return array;
        }

        inline void release() {
            if (!pointer) return;
            if (!device) free(pointer);
            else {
                if (!memory_pool_handle_) {
                    throw std::runtime_error("[Array::release] Memory pool handle is required for device memory");
                }
                kernel_provider::free(*memory_pool_handle_, pointer);
            }
        }
        ~Array() { 
            release();
        }

        Array& operator=(Array&& other) {
            release();
            pointer = other.pointer;
            len = other.len;
            device = other.device;
            memory_pool_handle_ = other.memory_pool_handle_;
            other.pointer = nullptr;
            other.memory_pool_handle_ = nullptr;
            other.len = 0;
            return *this;
        }
        
        __host__ __device__ bool on_device() const { return device; }
        __host__ __device__ T* raw_pointer() { return pointer; }
        __host__ __device__ const T* raw_pointer() const { return pointer; }

        Array(Array&& other): pointer(other.pointer), len(other.len), device(other.device), memory_pool_handle_(other.memory_pool_handle_) { 
            other.pointer = nullptr;
            other.memory_pool_handle_ = nullptr;
            other.len = 0; 
        }

        Array(const Array&) = delete;
        Array& operator=(const Array&) = delete;

        __host__ __device__ size_t size() const { return len; }
        __host__ ConstSlice<T> const_slice(size_t begin, size_t end) const {
            return ConstSlice<T>(pointer + begin, end - begin, device, memory_pool_handle_.get());
        }
        __host__ Slice<T> slice(size_t begin, size_t end) {
            return Slice<T>(pointer + begin, end - begin, device, memory_pool_handle_.get());
        }
        __host__ ConstSlice<T> const_reference() const {
            return ConstSlice<T>(pointer, len, device, memory_pool_handle_.get());
        }
        __host__ Slice<T> reference() {
            return Slice<T>(pointer, len, device, memory_pool_handle_.get());
        }
        __host__ __device__ ConstSlice<T> detached_const_slice(size_t begin, size_t end) const {
            return ConstSlice<T>(pointer + begin, end - begin, device, nullptr);
        }
        __host__ __device__ Slice<T> detached_slice(size_t begin, size_t end) {
            return Slice<T>(pointer + begin, end - begin, device, nullptr);
        }
        __host__ __device__ ConstSlice<T> detached_const_reference() const {
            return ConstSlice<T>(pointer, len, device, nullptr);
        }
        __host__ __device__ Slice<T> detached_reference() {
            return Slice<T>(pointer, len, device, nullptr);
        }
        __host__ __device__ const T& operator[](size_t index) const { return pointer[index]; }
        __host__ __device__ T& operator[](size_t index) { return pointer[index]; }
        __host__ Pointer<T> at(size_t index) { return Pointer<T>(pointer + index, device, memory_pool_handle_.get()); }
        __host__ ConstPointer<T> const_at(size_t index) const { return ConstPointer<T>(pointer + index, device, memory_pool_handle_.get()); }
        __host__ __device__ Pointer<T> detached_at(size_t index) { return Pointer<T>(pointer + index, device, nullptr); }
        __host__ __device__ ConstPointer<T> detached_const_at(size_t index) const { return ConstPointer<T>(pointer + index, device, nullptr); }

        inline Array clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            Array cloned;
            cloned.len = len;
            cloned.memory_pool_handle_ = pool;
            cloned.device = device;
            if (pointer && len > 0) {
                if (device) {
                    if (!pool) throw std::runtime_error("[Array::clone] Memory pool handle is required for device memory");
                    cloned.pointer = kernel_provider::malloc<T>(*pool, len);
                    kernel_provider::copy_device_to_device(*pool, cloned.pointer, pointer, len);
                } else {
                    cloned.pointer = reinterpret_cast<T*>(malloc(len * sizeof(T)));
                    memcpy(cloned.pointer, pointer, len * sizeof(T));
                }
            }
            return cloned;
        }

        inline Array to_host() const {
            if (!device) return this->clone(nullptr);
            Array cloned;
            cloned.len = len;
            cloned.device = false;
            cloned.memory_pool_handle_ = nullptr;
            cloned.pointer = reinterpret_cast<T*>(malloc(len * sizeof(T)));
            if (len > 0) {
                if (!memory_pool_handle_) throw std::runtime_error("[Array::to_host] Memory pool handle is required for device memory");
                kernel_provider::copy_device_to_host(*memory_pool_handle_, cloned.pointer, pointer, len);
            }
            return cloned;
        }

        inline Array to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (!pool) throw std::runtime_error("[Array::to_device] Memory pool handle is required for device memory");
            if (device) return this->clone(pool);
            Array cloned;
            cloned.len = len;
            cloned.device = true;
            cloned.memory_pool_handle_ = pool;
            cloned.pointer = kernel_provider::malloc<T>(*pool, len);
            if (len > 0) {
                kernel_provider::copy_host_to_device(*pool, cloned.pointer, pointer, len);
            }
            return cloned;
        }

        inline void to_host_inplace() {
            if (!device) return;
            if (len > 0) {
                if (!memory_pool_handle_) throw std::runtime_error("[Array::to_host_inplace] Memory pool handle is required for device memory");
                T* cloned = reinterpret_cast<T*>(malloc(len * sizeof(T)));
                kernel_provider::copy_device_to_host(*memory_pool_handle_, cloned, pointer, len);
                release();
                pointer = cloned;
            }
            device = false;
            memory_pool_handle_ = nullptr;
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            if (device) {
                if (pool != memory_pool_handle_) to_pool_inplace(pool);
                return;
            }
            if (!pool) throw std::runtime_error("[Array::to_device_inplace] Memory pool handle is required for device memory");
            if (len > 0) {
                T* cloned = kernel_provider::malloc<T>(*pool, len);
                kernel_provider::copy_host_to_device(*pool, cloned, pointer, len);
                release();
                pointer = cloned;
            }
            device = true;
            memory_pool_handle_ = pool;
        }

        void to_pool_inplace(MemoryPoolHandle pool) {
            if (!device) {
                return;
            } else {
                if (!pool) throw std::runtime_error("[Array::to_pool_inplace] Memory pool handle is required for device memory");
                T* cloned = kernel_provider::malloc<T>(*pool, len);
                kernel_provider::copy_device_to_device(*pool, cloned, pointer, len);
                release();
                pointer = cloned;
                memory_pool_handle_ = pool;
            }
        }

        inline void copy_from_slice(ConstSlice<T> slice) {
            this->reference().copy_from_slice(slice);
        }

        inline static Array<T> create_and_copy_from_slice(ConstSlice<T> slice, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            Array<T> array = Array::create_uninitialized(slice.size(), slice.on_device(), pool);
            array.copy_from_slice(slice);
            return array;
        }

        inline static Array<T> create_and_copy_from_slice(ConstSlice<T> slice, bool device, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            if (device && !pool) throw std::runtime_error("[Array::create_and_copy_from_slice] Memory pool handle is required for device memory");
            Array<T> array = Array::create_uninitialized(slice.size(), device, pool);
            array.copy_from_slice(slice);
            return array;
        }

        inline static Array<T> from_vector(std::vector<T>&& vector) {
            Array<T> array = Array::create_uninitialized(vector.size(), false, nullptr);
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
                os << slice[i];
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

// continue warning class-memaccess for gcc
#pragma GCC diagnostic pop

}}