#include <gtest/gtest.h>
#include "../test.cuh"
#include "../../src/utils/rns_tool.cuh"
#include <vector>

using namespace troy;
using namespace troy::utils;
using std::vector;

namespace rns_tool {

    bool same_array_vector(ConstSlice<uint64_t> arr, const vector<uint64_t>& vec) {
        if (arr.size() != vec.size()) return false;
        for (int i = 0; i < arr.size(); i++) {
            if (arr[i] != vec[i]) return false;
        }
        return true;
    }

    Array<Modulus> to_moduli(vector<uint64_t> m) {
        Array<Modulus> moduli(m.size(), false);
        for (int i = 0; i < m.size(); i++) {
            moduli[i] = Modulus(m[i]);
        }
        return moduli;
    }

    RNSBase to_rns_base(vector<uint64_t> m) {
        auto moduli = to_moduli(m);
        return RNSBase(moduli.const_reference());
    }

    bool test_body_divide_and_round_q_last_inplace(bool device) {
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({13, 7}), plain_t);
            if (device) rns_tool.to_device_inplace();
            size_t base_q_size = rns_tool.base_q().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_q_size, 0));
                if (device) input.to_device_inplace();
                rns_tool.divide_and_round_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 2), vector<uint64_t>(2, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2});
                if (device) input.to_device_inplace();
                rns_tool.divide_and_round_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 2), vector<uint64_t>({0, 0}))) return false;
            }
            
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({12, 11, 4, 3});
                if (device) input.to_device_inplace();
                rns_tool.divide_and_round_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 2), vector<uint64_t>({4, 3}))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({6, 2, 5, 1});
                if (device) input.to_device_inplace();
                rns_tool.divide_and_round_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 2), vector<uint64_t>({3, 2}))) return false;
            }
        }
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3, 5, 7, 11}), plain_t);
            if (device) rns_tool.to_device_inplace();
            size_t base_q_size = rns_tool.base_q().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_q_size, 0));
                if (device) input.to_device_inplace();
                rns_tool.divide_and_round_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 6), vector<uint64_t>(6, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2, 1, 2, 1, 2});
                if (device) input.to_device_inplace();
                rns_tool.divide_and_round_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 6), vector<uint64_t>(6, 0))) return false;
            }
            
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({0, 1, 0, 0, 4, 0, 5, 4});
                if (device) input.to_device_inplace();
                rns_tool.divide_and_round_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if ((3 + 2 - input[0]) % 3 > 1) return false;
                if ((3 + 0 - input[1]) % 3 > 1) return false;
                if ((5 + 0 - input[2]) % 5 > 1) return false;
                if ((5 + 1 - input[3]) % 5 > 1) return false;
                if ((7 + 5 - input[4]) % 7 > 1) return false;
                if ((7 + 6 - input[5]) % 7 > 1) return false;
            }
        }

        return true;

    }

    TEST(RNSToolTest, HostDivideAndRoundQLastInplace) {
        ASSERT_TRUE(test_body_divide_and_round_q_last_inplace(false));
    }

    TEST(RNSToolTest, DeviceDivideAndRoundQLastInplace) {
        ASSERT_TRUE(test_body_divide_and_round_q_last_inplace(true));
    }

}