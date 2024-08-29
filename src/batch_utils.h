#pragma once

#include "utils/box.h"
#include "utils/box_batch.h"
#include "ciphertext.h"

namespace troy::batch_utils {

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
        for (T& item : vec) {
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
    
    std::vector<utils::ConstSlice<uint64_t>> pcollect_const_poly(const std::vector<const Ciphertext*>& vec, size_t poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext* item : vec) {
            result.push_back(item->const_poly(poly_id));
        }
        return result;
    }

    std::vector<utils::ConstSlice<uint64_t>> rcollect_const_poly(const std::vector<Ciphertext>& vec, size_t poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext& item : vec) {
            result.push_back(item.const_poly(poly_id));
        }
        return result;
    }

    std::vector<utils::Slice<uint64_t>> pcollect_poly(const std::vector<Ciphertext*>& vec, size_t poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext* item : vec) {
            result.push_back(item->poly(poly_id));
        }
        return result;
    }

    std::vector<utils::Slice<uint64_t>> rcollect_poly(std::vector<Ciphertext>& vec, size_t poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext& item : vec) {
            result.push_back(item.poly(poly_id));
        }
        return result;
    }

    std::vector<utils::ConstSlice<uint64_t>> pcollect_const_polys(const std::vector<const Ciphertext*>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext* item : vec) {
            result.push_back(item->const_polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

    std::vector<utils::ConstSlice<uint64_t>> rcollect_const_polys(const std::vector<Ciphertext>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext& item : vec) {
            result.push_back(item.const_polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

    std::vector<utils::Slice<uint64_t>> pcollect_polys(const std::vector<Ciphertext*>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext* item : vec) {
            result.push_back(item->polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

    std::vector<utils::Slice<uint64_t>> rcollect_polys(std::vector<Ciphertext>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext& item : vec) {
            result.push_back(item.polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

}