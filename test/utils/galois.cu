#include <gtest/gtest.h>
#include "../test.h"
#include "../../src/utils/galois.h"

using namespace std;
using namespace troy;
using namespace troy::utils;

namespace galois_tool {
    
    bool same_array_vector(ConstSlice<uint64_t> arr, const vector<uint64_t>& vec) {
        if (arr.size() != vec.size()) return false;
        for (size_t i = 0; i < arr.size(); i++) {
            if (arr[i] != vec[i]) return false;
        }
        return true;
    }


    TEST(GaloisToolTest, GetElements) {
        size_t coeff_count_power = 3;
        GaloisTool tool(coeff_count_power);
        EXPECT_EQ(tool.get_element_from_step(0), 15);
        EXPECT_EQ(tool.get_element_from_step(1), 3);
        EXPECT_EQ(tool.get_element_from_step(-3), 3);
        EXPECT_EQ(tool.get_element_from_step(2), 9);
        EXPECT_EQ(tool.get_element_from_step(-2), 9);
        EXPECT_EQ(tool.get_element_from_step(3), 11);
        EXPECT_EQ(tool.get_element_from_step(-1), 11);

        vector<int> elements{0, 1, -3, 2, -2, 3, -1};
        vector<size_t> expected{15, 3, 3, 9, 9, 11, 11};
        vector<size_t> actual = tool.get_elements_from_steps(elements);
        EXPECT_EQ(actual, expected);

        vector<size_t> all{15, 3, 11, 9, 9};
        EXPECT_EQ(tool.get_elements_all(), all);

        EXPECT_EQ(tool.get_index_from_element(15), 7);
        EXPECT_EQ(tool.get_index_from_element(3), 1);
        EXPECT_EQ(tool.get_index_from_element(11), 5);
        EXPECT_EQ(tool.get_index_from_element(9), 4);

        MemoryPool::Destroy();
    }

    bool test_body_apply(bool device) {
        size_t logn = 3;
        size_t n = 1 << logn;
        GaloisTool tool(logn);
        if (device) tool.to_device_inplace();

        Array<uint64_t> input = Array<uint64_t>::from_vector({0,1,2,3,4,5,6,7});
        Array<uint64_t> output(n, false);
        if (device) {
            input.to_device_inplace(); output.to_device_inplace();
        }
        Box<Modulus> modulus(new Modulus(17), false);
        if (device) {
            modulus.to_device_inplace();
        }
        tool.apply(input.const_reference(), 3, modulus.as_const_pointer(), output.reference());
        if (device) {
            output.to_host_inplace();
        }
        if (!same_array_vector(output.const_reference(), {0,14,6,1,13,7,2,12})) return false;

        if (device) {
            output.to_device_inplace();
        }
        tool.apply_ntt(input.const_reference(), 3, output.reference());
        if (device) {
            output.to_host_inplace();
        }
        if (!same_array_vector(output.const_reference(), {4,5,7,6,1,0,2,3})) return false;

        return true;
    }

    TEST(GaloisToolTest, HostApply) {
        EXPECT_TRUE(test_body_apply(false));
    }

    TEST(GaloisToolTest, DeviceApply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        EXPECT_TRUE(test_body_apply(true));
        MemoryPool::Destroy();
    }

}