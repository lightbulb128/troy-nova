#pragma once
#include "box.h"
#include <cstddef>

namespace troy::utils {

    template <typename T>
    using ConstSliceVec = std::vector<ConstSlice<T>>;
    template <typename T>
    using SliceVec = std::vector<Slice<T>>;

    /*
        This class stores an array of slices with the same length.
        When this class is constructed, the memory pool information of each
        slice is discarded.
    */
    template <typename T>
    class ConstSliceArray {
        Array<const T*> pointers;
        bool inner_device_;
        size_t inner_device_index_;
        size_t len; // this means the length of each slice, not the length of the array
        
        // Construction must be on host
        void initialize(const ConstSlice<T>* begin, size_t count) {
            if (count == 0) {
                pointers = Array<const T*>(0, false);
                inner_device_ = false;
                inner_device_index_ = 0;
                len = 0;
            } else {
                len = begin[0].size();
                inner_device_ = begin[0].on_device();
                inner_device_index_ = inner_device_ ? begin[0].device_index() : 0;
                pointers = Array<const T*>(count, false);
                for (size_t i = 0; i < count; i++) {
                    pointers[i] = begin[i].raw_pointer();
                    if (begin[i].size() != len) {
                        throw std::runtime_error("[ConstSliceArray::ConstSliceArray] All slices must have the same length");
                    }
                    if (begin[i].on_device() != inner_device_) {
                        throw std::runtime_error("[ConstSliceArray::ConstSliceArray] All slices must be on host or all on device");
                    }
                    if (inner_device_ && begin[i].device_index() != inner_device_index_) {
                        throw std::runtime_error("[ConstSliceArray::ConstSliceArray] All slices must be on the same device");
                    }
                }
            }
        }

    public:
        MemoryPoolHandle pool() const { return pointers.pool(); }
        bool on_device() const { return pointers.on_device(); }
        bool inner_on_device() const { return inner_device_; }
        size_t device_index() const { return pointers.device_index(); }
        size_t inner_device_index() const { return inner_device_index_; }
        const T* const* raw_pointer() const {
            return pointers.raw_pointer(); 
        }

        // How many slices are there in the collection
        size_t count() const { return pointers.size(); }

        // Length of each slice
        size_t length() const { return len; }

        ConstSliceArray(): pointers(), len(0), inner_device_(false), inner_device_index_(0) {}

        ConstSliceArray(const ConstSlice<T>* begin, size_t count) {
            initialize(begin, count);
        }
        ConstSliceArray(const ConstSliceVec<T>& vec): 
            ConstSliceArray(vec.data(), vec.size()) {}

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            pointers.to_device_inplace(pool);
        }

        // The slices returned by this function does not have memory pool
        ConstSlice<T> operator[](size_t index) const {
            if (pointers.on_device()) {
                throw std::runtime_error("[ConstSliceArray::operator[]] Cannot access slices on device");
            }
            return ConstSlice<T>(pointers[index], len, inner_device_, nullptr);
        }

        template <typename U>
        bool device_compatible(const U& other) const {
            if (inner_device_ != other.on_device()) {
                return false;
            }
            if (inner_device_) {
                if (inner_device_index_ != other.device_index()) {
                    return false;
                }
            }
            return true;
        }
    };


    template <typename T>
    class SliceArray {
        Array<T*> pointers;
        bool inner_device_;
        size_t inner_device_index_;
        size_t len; // this means the length of each slice, not the length of the array
        // Construction must be on host.
        void initialize(const Slice<T>* begin, size_t count) {
            if (count == 0) {
                pointers = Array<T*>(0, false);
                inner_device_ = false;
                inner_device_index_ = 0;
                len = 0;
            } else {
                len = begin[0].size();
                inner_device_ = begin[0].on_device();
                inner_device_index_ = inner_device_ ? begin[0].device_index() : 0;
                pointers = Array<T*>(count, false);
                for (size_t i = 0; i < count; i++) {
                    pointers[i] = begin[i].raw_pointer();
                    if (begin[i].size() != len) {
                        throw std::runtime_error("[SliceArray::SliceArray] All slices must have the same length");
                    }
                    if (begin[i].on_device() != inner_device_) {
                        throw std::runtime_error("[SliceArray::SliceArray] All slices must be on host or all on device");
                    }
                    if (inner_device_ && begin[i].device_index() != inner_device_index_) {
                        throw std::runtime_error("[SliceArray::SliceArray] All slices must be on the same device");
                    }
                }
            }
        }
    public:
        MemoryPoolHandle pool() const { return pointers.pool(); }
        bool on_device() const { return pointers.on_device(); }
        bool inner_on_device() const { return inner_device_; }
        size_t device_index() const { return pointers.device_index(); }
        size_t inner_device_index() const { return inner_device_index_; }
        T* const* raw_pointer() const { return pointers.raw_pointer(); }

        // How many slices are there in the collection
        size_t count() const { return pointers.size(); }

        // Length of each slice
        size_t length() const { return len; }

        SliceArray(): pointers(), len(0), inner_device_(false), inner_device_index_(0) {}

        SliceArray(const Slice<T>* begin, size_t count) {
            initialize(begin, count);
        }
        SliceArray(const SliceVec<T>& vec): 
            SliceArray(vec.data(), vec.size()) {}

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            pointers.to_device_inplace(pool);
        }

        // The slices returned by this function does not have memory pool
        Slice<T> operator[](size_t index) const {
            if (pointers.on_device()) {
                throw std::runtime_error("[SliceArray::operator[]] Cannot access slices on device");
            }
            return Slice<T>(pointers[index], len, inner_device_, nullptr);
        }

        template <typename U>
        bool device_compatible(const U& other) const {
            if (inner_device_ != other.on_device()) {
                return false;
            }
            if (inner_device_) {
                if (inner_device_index_ != other.device_index()) {
                    return false;
                }
            }
            return true;
        }
    };


    template <typename T>
    class ConstSliceArrayRef {
        const T* const* pointers;
        size_t count_;
        size_t len;
        bool inner_device;
    public:
        ConstSliceArrayRef(const ConstSliceArray<T>& array):
            pointers(array.raw_pointer()), count_(array.count()), len(array.length()), inner_device(array.inner_on_device()) {}
        __host__ __device__ size_t count() const { return count_; }
        __host__ __device__ size_t length() const { return len; }
        __host__ __device__ ConstSlice<T> operator[](size_t index) const {
            return ConstSlice<T>(pointers[index], len, inner_device, nullptr);
        }
    };

    template <typename T>
    class SliceArrayRef {
        T* const* pointers;
        size_t count_;
        size_t len;
        bool inner_device;
    public:
        SliceArrayRef(const SliceArray<T>& array):
            pointers(array.raw_pointer()), count_(array.count()), len(array.length()), inner_device(array.inner_on_device()) {}
        __host__ __device__ size_t count() const { return count_; }
        __host__ __device__ size_t length() const { return len; }
        __host__ __device__ Slice<T> operator[](size_t index) const {
            return Slice<T>(pointers[index], len, inner_device, nullptr);
        }
    };

}