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
        Array<ConstSlice<T>> slices;
        bool inner_device_;
        size_t inner_device_index_;
        
        // Construction must be on host
        void initialize(const ConstSlice<T>* begin, size_t count) {
            if (count == 0) {
                slices = Array<ConstSlice<T>>(0, false);
                inner_device_ = false;
                inner_device_index_ = 0;
            } else {
                inner_device_ = false;
                inner_device_index_ = 0;
                for (size_t i = 0; i < count; i++) {
                    if (begin[i].on_device()) {
                        inner_device_ = true;
                        inner_device_index_ = begin[i].device_index();
                        break;
                    }
                }
                slices = Array<ConstSlice<T>>(count, false);
                for (size_t i = 0; i < count; i++) {
                    slices[i] = begin[i];
                    if (inner_device_ && !begin[i].on_device() && begin[i].raw_pointer()) {
                        throw std::runtime_error("[ConstSliceArray::ConstSliceArray] All slices must be on host or all on device");
                    }
                    if (inner_device_ && begin[i].on_device() && begin[i].device_index() != inner_device_index_) {
                        throw std::runtime_error("[ConstSliceArray::ConstSliceArray] All slices must be on the same device");
                    }
                }
            }
        }

    public:
        MemoryPoolHandle pool() const { return slices.pool(); }
        bool on_device() const { return slices.on_device(); }
        bool inner_on_device() const { return inner_device_; }
        size_t device_index() const { return slices.device_index(); }
        size_t inner_device_index() const { return inner_device_index_; }
        const ConstSlice<T>* raw_pointer() const {
            return slices.raw_pointer(); 
        }

        // How many slices are there in the collection
        size_t size() const { return slices.size(); }

        ConstSliceArray(): slices(), inner_device_(false), inner_device_index_(0) {}

        ConstSliceArray(const ConstSlice<T>* begin, size_t count) {
            initialize(begin, count);
        }
        ConstSliceArray(const ConstSliceVec<T>& vec): 
            ConstSliceArray(vec.data(), vec.size()) {}

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            slices.to_device_inplace(pool);
        }

        // The slices returned by this function does not have memory pool
        ConstSlice<T> operator[](size_t index) const {
            if (slices.on_device()) {
                throw std::runtime_error("[ConstSliceArray::operator[]] Cannot access slices on device");
            }
            return slices[index];
        }

        template <typename U>
        bool device_compatible(const U& other) const {
            if (slices.on_device()) {
                throw std::runtime_error("[ConstSliceArray::device_compatible] Cannot access slices on device");
            }
            if (other.on_device() && !inner_device_) {
                // all slices must be nullptr
                for (size_t i = 0; i < slices.size(); i++) {
                    if (slices[i].raw_pointer()) {
                        return false;
                    }
                }
            }
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
        Array<Slice<T>> slices;
        bool inner_device_;
        size_t inner_device_index_;
        // Construction must be on host.
        void initialize(const Slice<T>* begin, size_t count) {
            if (count == 0) {
                slices = Array<Slice<T>>(0, false);
                inner_device_ = false;
                inner_device_index_ = 0;
            } else {
                inner_device_ = false;
                inner_device_index_ = 0;
                for (size_t i = 0; i < count; i++) {
                    if (begin[i].on_device()) {
                        inner_device_ = true;
                        inner_device_index_ = begin[i].device_index();
                        break;
                    }
                }
                slices = Array<Slice<T>>(count, false);
                for (size_t i = 0; i < count; i++) {
                    slices[i] = begin[i];
                    if (inner_device_ && !begin[i].on_device() && begin[i].raw_pointer()) {
                        throw std::runtime_error("[ConstSliceArray::ConstSliceArray] All slices must be on host or all on device");
                    }
                    if (inner_device_ && begin[i].on_device() && begin[i].device_index() != inner_device_index_) {
                        throw std::runtime_error("[ConstSliceArray::ConstSliceArray] All slices must be on the same device");
                    }
                }
            }
        }
    public:
        MemoryPoolHandle pool() const { return slices.pool(); }
        bool on_device() const { return slices.on_device(); }
        bool inner_on_device() const { return inner_device_; }
        size_t device_index() const { return slices.device_index(); }
        size_t inner_device_index() const { return inner_device_index_; }
        const Slice<T>* raw_pointer() const { return slices.raw_pointer(); }

        // How many slices are there in the collection
        size_t size() const { return slices.size(); }

        SliceArray(): slices(), inner_device_(false), inner_device_index_(0) {}

        SliceArray(const Slice<T>* begin, size_t count) {
            initialize(begin, count);
        }
        SliceArray(const SliceVec<T>& vec): 
            SliceArray(vec.data(), vec.size()) {}

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            slices.to_device_inplace(pool);
        }

        // The slices returned by this function does not have memory pool
        Slice<T> operator[](size_t index) const {
            if (slices.on_device()) {
                throw std::runtime_error("[SliceArray::operator[]] Cannot access slices on device");
            }
            return slices[index];
        }

        template <typename U>
        bool device_compatible(const U& other) const {
            if (slices.on_device()) {
                throw std::runtime_error("[SliceArray::device_compatible] Cannot access slices on device");
            }
            if (other.on_device() && !inner_device_) {
                // all slices must be nullptr
                for (size_t i = 0; i < slices.size(); i++) {
                    if (slices[i].raw_pointer()) {
                        return false;
                    }
                }
            }
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
        const ConstSlice<T>* pointers;
        size_t count_;
        bool inner_device;
    public:
        ConstSliceArrayRef(const ConstSliceArray<T>& array):
            pointers(array.raw_pointer()), count_(array.size()), inner_device(array.inner_on_device()) {}
        __host__ __device__ size_t size() const { return count_; }
        __host__ __device__ ConstSlice<T> operator[](size_t index) const {
            return pointers[index];
        }
    };

    template <typename T>
    class SliceArrayRef {
        const Slice<T>* pointers;
        size_t count_;
        bool inner_device;
    public:
        SliceArrayRef(const SliceArray<T>& array):
            pointers(array.raw_pointer()), count_(array.size()), inner_device(array.inner_on_device()) {}
        __host__ __device__ size_t size() const { return count_; }
        __host__ __device__ Slice<T> operator[](size_t index) const {
            return pointers[index];
        }
    };

    

    template <typename T, typename U>
    utils::ConstSliceArray<T> construct_batch(const utils::ConstSliceVec<T>& vec, const MemoryPoolHandle& pool, const U& comp_ref) {
        utils::ConstSliceArray<T> arr(vec);
        if (!arr.device_compatible(comp_ref)) {
            throw std::runtime_error("[construct_batch] All inputs must be on the same device as comp_ref");
        }
        if (pool->get_device() != comp_ref.device_index()) {
            throw std::runtime_error("[construct_batch] All inputs must be on the same device as the pool");
        }
        arr.to_device_inplace(pool);
        return arr;
    }

    template <typename T, typename U>
    utils::SliceArray<T> construct_batch(const utils::SliceVec<T>& vec, const MemoryPoolHandle& pool, const U& comp_ref) {
        utils::SliceArray<T> arr(vec);
        if (!arr.device_compatible(comp_ref)) {
            throw std::runtime_error("[construct_batch] All inputs must be on the same device as comp_ref");
        }
        if (pool->get_device() != comp_ref.device_index()) {
            throw std::runtime_error("[construct_batch] All inputs must be on the same device as the pool");
        }
        arr.to_device_inplace(pool);
        return arr;
    }


    template <typename T>
    inline utils::ConstSliceVec<T> rcollect_as_const(const utils::SliceVec<T>& vec) {
        std::vector<utils::ConstSlice<T>> result;
        result.reserve(vec.size());
        for (const utils::Slice<T>& item : vec) {
            result.push_back(item.as_const());
        }
        return result;
    }

    template <typename T>
    inline std::vector<const T*> pcollect_const_pointer(const std::vector<T*>& vec) {
        std::vector<const T*> result;
        result.reserve(vec.size());
        for (T* item : vec) {
            result.push_back(item);
        }
        return result;
    }

    template <typename T>
    std::vector<T> clone(const std::vector<T>& vec) {
        std::vector<T> result;
        result.reserve(vec.size());
        for (const T& item : vec) {
            result.push_back(item.clone());
        }
        return result;
    }

    template <typename T>
    std::vector<T> pclone(const std::vector<T*>& vec) {
        std::vector<T> result;
        result.reserve(vec.size());
        for (T* item : vec) {
            result.push_back(item->clone());
        }
        return result;
    }

    template <typename T>
    std::vector<T*> collect_pointer(std::vector<T>& vec) {
        std::vector<T*> result;
        result.reserve(vec.size());
        for (T& item : vec) {
            result.push_back(&item);
        }
        return result;
    }

    template <typename T>
    std::vector<const T*> collect_const_pointer(const std::vector<T>& vec) {
        std::vector<const T*> result;
        result.reserve(vec.size());
        for (const T& item : vec) {
            result.push_back(&item);
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::ConstSlice<U>> pcollect_const_reference(const std::vector<const T*>& vec) {
        std::vector<utils::ConstSlice<U>> result;
        result.reserve(vec.size());
        for (const T* item : vec) {
            result.push_back(item->const_reference());
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::ConstSlice<U>> rcollect_const_reference(const std::vector<T>& vec) {
        std::vector<utils::ConstSlice<U>> result;
        result.reserve(vec.size());
        for (const T& item : vec) {
            result.push_back(item.const_reference());
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::Slice<U>> pcollect_reference(const std::vector<T*>& vec) {
        std::vector<utils::Slice<U>> result;
        result.reserve(vec.size());
        for (T* item : vec) {
            result.push_back(item->reference());
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::Slice<U>> rcollect_reference(std::vector<T>& vec) {
        std::vector<utils::Slice<U>> result;
        result.reserve(vec.size());
        for (T& item : vec) {
            result.push_back(item.reference());
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::ConstSlice<T>> pcollect_const_slice(const std::vector<const T*> vec, size_t begin, size_t end) {
        std::vector<utils::ConstSlice<T>> result;
        result.reserve(vec.size());
        for (T* item : vec) {
            result.push_back(item->const_slice(begin, end));
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::ConstSlice<T>> rcollect_const_slice(const std::vector<T>& vec, size_t begin, size_t end) {
        std::vector<utils::ConstSlice<T>> result;
        result.reserve(vec.size());
        for (T& item : vec) {
            result.push_back(item.const_slice(begin, end));
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::Slice<T>> pcollect_slice(std::vector<T*> vec, size_t begin, size_t end) {
        std::vector<utils::Slice<T>> result;
        result.reserve(vec.size());
        for (T* item : vec) {
            result.push_back(item->slice(begin, end));
        }
        return result;
    }

    template <typename T, typename U = uint64_t>
    std::vector<utils::Slice<T>> rcollect_slice(std::vector<T>& vec, size_t begin, size_t end) {
        std::vector<utils::Slice<T>> result;
        result.reserve(vec.size());
        for (T& item : vec) {
            result.push_back(item.slice(begin, end));
        }
        return result;
    }

}