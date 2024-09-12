#pragma once

#include "utils/box.h"
#include "utils/box_batch.h"
#include "ciphertext.h"
#include "plaintext.h"

namespace troy::batch_utils {

    using utils::ConstSliceArray;
    using utils::SliceArray;
    using utils::construct_batch;
    using utils::rcollect_as_const;
    using utils::clone;
    using utils::pclone;
    using utils::collect_pointer;
    using utils::collect_const_pointer;
    using utils::pcollect_const_reference;
    using utils::rcollect_const_reference;
    using utils::pcollect_reference;
    using utils::rcollect_reference;
    using utils::pcollect_const_slice;
    using utils::rcollect_const_slice;
    using utils::pcollect_slice;
    using utils::rcollect_slice;
    using utils::pcollect_const_pointer;
    
    inline std::vector<utils::ConstSlice<uint64_t>> pcollect_const_poly(const std::vector<const Ciphertext*>& vec, size_t poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext* item : vec) {
            result.push_back(item->const_poly(poly_id));
        }
        return result;
    }

    inline std::vector<utils::ConstSlice<uint64_t>> rcollect_const_poly(const std::vector<Ciphertext>& vec, size_t poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext& item : vec) {
            result.push_back(item.const_poly(poly_id));
        }
        return result;
    }

    inline std::vector<utils::ConstSlice<uint64_t>> pcollect_const_poly(const std::vector<const Plaintext*>& vec) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Plaintext* item : vec) {
            result.push_back(item->const_poly());
        }
        return result;
    }

    inline std::vector<utils::ConstSlice<uint64_t>> rcollect_const_poly(std::vector<Plaintext>& vec) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Plaintext& item : vec) {
            result.push_back(item.const_poly());
        }
        return result;
    }

    inline std::vector<utils::Slice<uint64_t>> pcollect_poly(const std::vector<Ciphertext*>& vec, size_t poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext* item : vec) {
            result.push_back(item->poly(poly_id));
        }
        return result;
    }

    inline std::vector<utils::Slice<uint64_t>> rcollect_poly(std::vector<Ciphertext>& vec, size_t poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext& item : vec) {
            result.push_back(item.poly(poly_id));
        }
        return result;
    }
    
    inline std::vector<utils::Slice<uint64_t>> pcollect_poly(const std::vector<Plaintext*>& vec) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Plaintext* item : vec) {
            result.push_back(item->poly());
        }
        return result;
    }

    inline std::vector<utils::Slice<uint64_t>> rcollect_poly(std::vector<Plaintext>& vec) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Plaintext& item : vec) {
            result.push_back(item.poly());
        }
        return result;
    }

    inline std::vector<utils::ConstSlice<uint64_t>> pcollect_const_polys(const std::vector<const Ciphertext*>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext* item : vec) {
            result.push_back(item->const_polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

    inline std::vector<utils::ConstSlice<uint64_t>> rcollect_const_polys(const std::vector<Ciphertext>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::ConstSlice<uint64_t>> result;
        result.reserve(vec.size());
        for (const Ciphertext& item : vec) {
            result.push_back(item.const_polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

    inline std::vector<utils::Slice<uint64_t>> pcollect_polys(const std::vector<Ciphertext*>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext* item : vec) {
            result.push_back(item->polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

    inline std::vector<utils::Slice<uint64_t>> rcollect_polys(std::vector<Ciphertext>& vec, size_t lower_poly_id, size_t upper_poly_id) {
        std::vector<utils::Slice<uint64_t>> result;
        result.reserve(vec.size());
        for (Ciphertext& item : vec) {
            result.push_back(item.polys(lower_poly_id, upper_poly_id));
        }
        return result;
    }

}