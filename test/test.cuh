#pragma once
#include <iostream>
#include <vector>
#include "../src/utils/box.cuh"

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

inline bool all_is_true(const troy::utils::Array<bool>& v) {
    size_t n = v.size();
    for (size_t i = 0; i < n; i++) {
        if (!v[i]) {
            return false;
        }
    }
    return true;
}