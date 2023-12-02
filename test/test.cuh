#pragma once
#include <iostream>
#include <vector>
#include "../src/utils/box.cuh"

using namespace troy;

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
        utils::ConstSlice(a.data(), a.size(), false),
        b
    );
}

template<typename T>
inline bool same_vector(utils::ConstSlice<T> a, const std::vector<T>& b) {
    return same_vector(
        a,
        utils::ConstSlice(b.data(), b.size(), false)
    );
}

template<typename T>
inline bool same_vector(const std::vector<T>& a, const std::vector<T>& b) {
    return same_vector(
        utils::ConstSlice(a.data(), a.size(), false),
        utils::ConstSlice(b.data(), b.size(), false)
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