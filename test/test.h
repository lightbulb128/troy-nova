#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include "../src/utils/box.h"
#include <gtest/gtest.h>

using namespace troy;
using std::complex;
using std::vector;

// #define FAIL std::cout << "Assertion failed at " << __FILE__ << ":" << __LINE__ << std::endl
// #define EXPECT_EQ(a, b) if (a != b) { FAIL; return true; }
// #define EXPECT_TRUE(a) if (!a) { FAIL; return true; }
// #define EXPECT_FALSE(a) if (a) { FAIL; return true; }

// #define RUN_TEST(namespace, func) \
//     if (namespace::func()) { \
//         std::cout << "Test " << #namespace << "::" << #func << " failed" << std::endl; \
//         return 1; \
//     } else { \
//         std::cout << "Test " << #namespace << "::" << #func << " passed" << std::endl; \
//     }

#define KERNEL_EXPECT_EQ(a, b) \
    if (a != b) { \
        test_result[kernel_index] = false; \
    }

#define KERNEL_EXPECT_TRUE(a) \
    if (!a) { \
        test_result[kernel_index] = false; \
    }

#define KERNEL_EXPECT_FALSE(a) \
    if (a) { \
        test_result[kernel_index] = false; \
    }

#define RETURN_EQ(a, b) \
    if (a != b) { \
        return false; \
    }

inline bool all_is_true(const utils::Array<bool>& v) {
    size_t n = v.size();
    for (size_t i = 0; i < n; i++) {
        if (!v[i]) {
            return false;
        }
    }
    return true;
}

template<typename T>
inline bool same_vector(utils::ConstSlice<T> a, utils::ConstSlice<T> b) {
    if (a.size() != b.size()) {
        return false;
    }
    if (a.on_device()) {
        utils::Array<T> a_host = utils::Array<T>::create_and_copy_from_slice(a);
        a_host.to_host_inplace();
        return same_vector(a_host.const_reference(), b);
    }
    if (b.on_device()) {
        utils::Array<T> b_host = utils::Array<T>::create_and_copy_from_slice(b);
        b_host.to_host_inplace();
        return same_vector(a, b_host.const_reference());
    }
    size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

template<typename T>
inline bool same_vector(const std::vector<T>& a, utils::ConstSlice<T> b) {
    return same_vector(
        utils::ConstSlice(a.data(), a.size(), false, nullptr),
        b
    );
}

template<typename T>
inline bool same_vector(utils::ConstSlice<T> a, const std::vector<T>& b) {
    return same_vector(
        a,
        utils::ConstSlice(b.data(), b.size(), false, nullptr)
    );
}

template<typename T>
inline bool same_vector(const std::vector<T>& a, const std::vector<T>& b) {
    return same_vector(
        utils::ConstSlice(a.data(), a.size(), false, nullptr),
        utils::ConstSlice(b.data(), b.size(), false, nullptr)
    );
}

template<typename T> void print_vector(const std::vector<T>& v, bool end_line = true) {
    size_t n = v.size();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        std::cout << v[i]; 
        if (i != n - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
    if (end_line) {
        std::cout << std::endl;
    }
}

inline bool near_vector(const vector<complex<double>> &a, const vector<complex<double>> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i].real() - b[i].real()) > 0.5) {
            return false;
        }
        if (std::abs(a[i].imag() - b[i].imag()) > 0.5) {
            return false;
        }
    }
    return true;
}

inline bool near_vector(const vector<double> &a, const vector<double> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > 0.5) {
            return false;
        }
    }
    return true;
}

inline bool near_vector(const vector<int64_t> &a, const vector<double> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > 0.5) {
            return false;
        }
    }
    return true;
}

inline vector<uint64_t> random_uint64_vector(size_t count, uint64_t mod) {
    vector<uint64_t> vec(count);
    for (size_t i = 0; i < count; i++) {
        vec[i] = rand() % mod;
    }
    return vec;
}

inline vector<complex<double>> random_complex64_vector(size_t count, double component_max = 10.0) {
    vector<complex<double>> vec(count);
    for (size_t i = 0; i < count; i++) {
        vec[i] = complex<double>(
            (double)rand() / RAND_MAX * 2 * component_max - component_max,
            (double)rand() / RAND_MAX * 2 * component_max - component_max
        );
    }
    return vec;
}

inline vector<double> random_double_vector(size_t count, double max) {
    vector<double> vec(count);
    for (size_t i = 0; i < count; i++) {
        vec[i] = (double)rand() / RAND_MAX * max;
    }
    return vec;
}

#define SKIP_WHEN_NO_CUDA_DEVICE {               \
    int count = troy::utils::device_count();     \
    if (count == 0) {                            \
        GTEST_SKIP_("No CUDA device found");     \
    }                                            \
}

