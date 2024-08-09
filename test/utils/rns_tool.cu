#include <gtest/gtest.h>
#include "../test.h"
#include "../../src/utils/rns_tool.h"
#include <vector>

using namespace troy;
using namespace troy::utils;
using std::vector;

namespace rns_tool {

    bool same_array_vector(ConstSlice<uint64_t> arr, const vector<uint64_t>& vec) {
        if (arr.size() != vec.size()) return false;
        for (size_t i = 0; i < arr.size(); i++) {
            if (arr[i] != vec[i]) return false;
        }
        return true;
    }

    Array<Modulus> to_moduli(vector<uint64_t> m) {
        Array<Modulus> moduli(m.size(), false);
        for (size_t i = 0; i < m.size(); i++) {
            moduli[i] = Modulus(m[i]);
        }
        return moduli;
    }

    RNSBase to_rns_base(vector<uint64_t> m) {
        auto moduli = to_moduli(m);
        return RNSBase(moduli.const_reference());
    }



    bool test_body_fast_b_conv_sk(bool device) {
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3}), plain_t);
            if (device) rns_tool.to_device_inplace();
            size_t base_q_size = rns_tool.base_q().size();
            size_t base_Bsk_size = rns_tool.base_Bsk().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_Bsk_size, 0));
                Array<uint64_t> output(poly_modulus_degree * base_q_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_b_conv_sk(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), vector<uint64_t>(poly_modulus_degree * base_q_size, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2});
                Array<uint64_t> output(poly_modulus_degree * base_q_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_b_conv_sk(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2})) return false;
            }
            
        }
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3, 5}), plain_t);
            if (device) rns_tool.to_device_inplace();
            size_t base_q_size = rns_tool.base_q().size();
            size_t base_Bsk_size = rns_tool.base_Bsk().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2, 1, 2});
                Array<uint64_t> output(poly_modulus_degree * base_q_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_b_conv_sk(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2, 1, 2})) return false;
            }
        }

        return true;

    }

    TEST(RNSToolTest, HostFastBConvSK) {
        ASSERT_TRUE(test_body_fast_b_conv_sk(false));
    }

    TEST(RNSToolTest, DeviceFastBConvSK) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        ASSERT_TRUE(test_body_fast_b_conv_sk(true));
        MemoryPool::Destroy();
    }

    
    bool test_body_montgomery_reduction(bool device) {
        // This function assumes the input is in base Bsk U {m_tilde}. If the input is
        // |[c*m_tilde]_q + qu|_m for m in Bsk U {m_tilde}, then the output is c' in Bsk
        // such that c' = c mod q. In other words, this function cancels the extra multiples
        // of q in the Bsk U {m_tilde} representation. The functions works correctly for
        // sufficiently small values of u.
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3}), plain_t);
            uint64_t m_tilde_value = rns_tool.m_tilde()->value();
            uint64_t q0_value = rns_tool.base_q().base()[0].value();
            if (device) rns_tool.to_device_inplace();
            size_t base_Bsk_m_tilde_size = rns_tool.base_Bsk_m_tilde().size();
            size_t base_Bsk_size = rns_tool.base_Bsk().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_Bsk_m_tilde_size, 0));
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.sm_mrq(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), vector<uint64_t>(poly_modulus_degree * base_Bsk_size, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({m_tilde_value, 2 * m_tilde_value, m_tilde_value, 2 * m_tilde_value, 0, 0});
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.sm_mrq(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2, 1, 2})) return false;
            }
            
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({q0_value, q0_value, q0_value, q0_value, q0_value, q0_value});
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.sm_mrq(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {0, 0, 0, 0})) return false;
            }
            
        }
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3, 5}), plain_t);
            uint64_t m_tilde_value = rns_tool.m_tilde()->value();
            if (device) rns_tool.to_device_inplace();
            size_t base_Bsk_m_tilde_size = rns_tool.base_Bsk_m_tilde().size();
            size_t base_Bsk_size = rns_tool.base_Bsk().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({
                    m_tilde_value, 2 * m_tilde_value, 
                    m_tilde_value, 2 * m_tilde_value, 
                    m_tilde_value, 2 * m_tilde_value, 
                    0, 0
                });
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.sm_mrq(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2, 1, 2, 1, 2})) return false;
            }
            
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({
                    15, 30, 15, 30, 15, 30, 15, 30
                });
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.sm_mrq(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {0, 0, 0, 0, 0, 0})) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({
                    2 * m_tilde_value + 15, 
                    2 * m_tilde_value + 30, 
                    2 * m_tilde_value + 15, 
                    2 * m_tilde_value + 30, 
                    2 * m_tilde_value + 15, 
                    2 * m_tilde_value + 30, 
                    2 * m_tilde_value + 15, 
                    2 * m_tilde_value + 30
                });
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.sm_mrq(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {2, 2, 2, 2, 2, 2})) return false;
            }
        }

        return true;

    }

    TEST(RNSToolTest, HostMontgomeryReduction) {
        ASSERT_TRUE(test_body_montgomery_reduction(false));
    }

    TEST(RNSToolTest, DeviceMontgomeryReduction) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        ASSERT_TRUE(test_body_montgomery_reduction(true));
        MemoryPool::Destroy();
    }

    
    bool test_body_fast_floor(bool device) {
        // This function assumes the input is in base q U Bsk. It outputs an approximation of
        // the value divided by q floored in base Bsk. The approximation has absolute value up
        // to k-1, where k is the number of primes in the base q.
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3}), plain_t);
            size_t base_q_size = rns_tool.base_q().size();
            size_t base_Bsk_size = rns_tool.base_Bsk().size();
            if (device) rns_tool.to_device_inplace();
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * (base_q_size + base_Bsk_size), 0));
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_floor(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), vector<uint64_t>(poly_modulus_degree * base_Bsk_size, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({15, 3, 15, 3, 15, 3});
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_floor(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {5, 1, 5, 1})) return false;
            }
            
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({17, 4, 17, 4, 17, 4});
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_floor(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {5, 1, 5, 1})) return false;
            }
        }
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3, 5}), plain_t);
            size_t base_q_size = rns_tool.base_q().size();
            size_t base_Bsk_size = rns_tool.base_Bsk().size();
            if (device) rns_tool.to_device_inplace();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * (base_q_size + base_Bsk_size), 0));
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_floor(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), vector<uint64_t>(poly_modulus_degree * base_Bsk_size, 0))) return false;
            }
            
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({
                    15, 30, 15, 30, 15, 30, 15, 30, 15, 30
                });
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_floor(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2, 1, 2, 1, 2})) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({
                    21, 32, 21, 32, 21, 32, 21, 32, 21, 32
                });
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_floor(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (std::abs((int64_t(1) - int64_t(output[0]))) > 1) return false;
                if (std::abs((int64_t(2) - int64_t(output[1]))) > 1) return false;
                if (std::abs((int64_t(1) - int64_t(output[2]))) > 1) return false;
                if (std::abs((int64_t(2) - int64_t(output[3]))) > 1) return false;
                if (std::abs((int64_t(1) - int64_t(output[4]))) > 1) return false;
                if (std::abs((int64_t(2) - int64_t(output[5]))) > 1) return false;
            }
        }

        return true;

    }

    TEST(RNSToolTest, HostFastFloor) {
        ASSERT_TRUE(test_body_fast_floor(false));
    }

    TEST(RNSToolTest, DeviceFastFloor) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        ASSERT_TRUE(test_body_fast_floor(true));
        MemoryPool::Destroy();
    }


    bool test_body_fast_b_conv_m_tilde(bool device) {
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3}), plain_t);
            RNSTool rns_tool_cpu = rns_tool.clone();
            size_t base_q_size = rns_tool.base_q().size();
            size_t base_Bsk_m_tilde_size = rns_tool.base_Bsk_m_tilde().size();
            if (device) rns_tool.to_device_inplace();
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_q_size, 0));
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_m_tilde_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_b_conv_m_tilde(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), vector<uint64_t>(poly_modulus_degree * base_Bsk_m_tilde_size, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2});
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_m_tilde_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_b_conv_m_tilde(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                uint64_t temp = rns_tool.m_tilde_value() % 3;
                uint64_t temp2 = (2 * rns_tool.m_tilde_value()) % 3;
                if ((temp  % rns_tool_cpu.base_Bsk_m_tilde().base()[0].value()) != output[0]) return false;
                if ((temp2 % rns_tool_cpu.base_Bsk_m_tilde().base()[0].value()) != output[1]) return false;
                if ((temp  % rns_tool_cpu.base_Bsk_m_tilde().base()[1].value()) != output[2]) return false;
                if ((temp2 % rns_tool_cpu.base_Bsk_m_tilde().base()[1].value()) != output[3]) return false;
                if ((temp  % rns_tool_cpu.base_Bsk_m_tilde().base()[2].value()) != output[4]) return false;
                if ((temp2 % rns_tool_cpu.base_Bsk_m_tilde().base()[2].value()) != output[5]) return false;
            }
        }
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(0);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({3, 5}), plain_t);
            RNSTool rns_tool_cpu = rns_tool.clone();
            size_t base_q_size = rns_tool.base_q().size();
            size_t base_Bsk_m_tilde_size = rns_tool.base_Bsk_m_tilde().size();
            if (device) rns_tool.to_device_inplace();
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_q_size, 0));
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_m_tilde_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_b_conv_m_tilde(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), vector<uint64_t>(poly_modulus_degree * base_Bsk_m_tilde_size, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 1, 2, 2});
                Array<uint64_t> output(poly_modulus_degree * base_Bsk_m_tilde_size, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.fast_b_conv_m_tilde(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                uint64_t m_tilde = rns_tool.m_tilde_value();
                uint64_t temp = ((2 * m_tilde) % 3) * 5 + ((4 * m_tilde) % 5) * 3;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[0].value()) != output[0]) return false;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[0].value()) != output[1]) return false;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[1].value()) != output[2]) return false;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[1].value()) != output[3]) return false;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[2].value()) != output[4]) return false;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[2].value()) != output[5]) return false;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[3].value()) != output[6]) return false;
                if ((temp % rns_tool_cpu.base_Bsk_m_tilde().base()[3].value()) != output[7]) return false;
            }
        }

        return true;

    }

    TEST(RNSToolTest, HostFastBConvMTilde) {
        ASSERT_TRUE(test_body_fast_b_conv_m_tilde(false));
    }

    TEST(RNSToolTest, DeviceFastBConvMTilde) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        ASSERT_TRUE(test_body_fast_b_conv_m_tilde(true));
        MemoryPool::Destroy();
    }

    bool test_body_exact_scale_and_round(bool device) {
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(3);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({5, 7}), plain_t);
            RNSTool rns_tool_cpu = rns_tool.clone();
            if (device) rns_tool.to_device_inplace();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({29, 65, 29, 65});
                Array<uint64_t> output(poly_modulus_degree, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.decrypt_scale_and_round(input.const_reference(), poly_modulus_degree, output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {2, 0})) return false;
            }
        }

        return true;

    }

    TEST(RNSToolTest, HostExactScaleAndRound) {
        ASSERT_TRUE(test_body_exact_scale_and_round(false));
    }

    TEST(RNSToolTest, DeviceExactScaleAndRound) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        ASSERT_TRUE(test_body_exact_scale_and_round(true));
        MemoryPool::Destroy();
    }
    
    
    bool test_body_mod_t_and_divide_q_last_inplace(bool device) {
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(3);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({13, 7}), plain_t);
            if (device) rns_tool.to_device_inplace();
            size_t base_q_size = rns_tool.base_q().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_q_size, 0));
                if (device) input.to_device_inplace();
                rns_tool.mod_t_and_divide_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 2), vector<uint64_t>(2, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2});
                if (device) input.to_device_inplace();
                rns_tool.mod_t_and_divide_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 2), vector<uint64_t>({11, 12}))) return false;
            }
            
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({12, 11, 4, 3});
                if (device) input.to_device_inplace();
                rns_tool.mod_t_and_divide_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 2), vector<uint64_t>({1, 3}))) return false;
            }
        }
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(3);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({5, 7, 11}), plain_t);
            if (device) rns_tool.to_device_inplace();
            size_t base_q_size = rns_tool.base_q().size();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector(vector<uint64_t>(poly_modulus_degree * base_q_size, 0));
                if (device) input.to_device_inplace();
                rns_tool.mod_t_and_divide_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 4), vector<uint64_t>(4, 0))) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2, 1, 2});
                if (device) input.to_device_inplace();
                rns_tool.mod_t_and_divide_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 4), {4, 3, 6, 5})) return false;
            }

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({0, 1, 0, 0, 4, 0});
                if (device) input.to_device_inplace();
                rns_tool.mod_t_and_divide_q_last_inplace(input.reference());
                if (device) input.to_host_inplace();
                if (!same_array_vector(input.const_slice(0, 4), {0, 1, 5, 0})) return false;
            }
            
        }

        return true;

    }

    TEST(RNSToolTest, HostModTAndDivideQLastInplace) {
        ASSERT_TRUE(test_body_mod_t_and_divide_q_last_inplace(false));
    }

    TEST(RNSToolTest, DeviceModTAndDivideQLastInplace) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        ASSERT_TRUE(test_body_mod_t_and_divide_q_last_inplace(true));
        MemoryPool::Destroy();
    }

    bool test_body_decrypt_mod_t(bool device) {
        
        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(3);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({13, 7}), plain_t);
            RNSTool rns_tool_cpu = rns_tool.clone();
            if (device) rns_tool.to_device_inplace();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({0, 0, 0, 0});
                Array<uint64_t> output(poly_modulus_degree, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.decrypt_mod_t(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {0, 0})) return false;
            }
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2});
                Array<uint64_t> output(poly_modulus_degree, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.decrypt_mod_t(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2})) return false;
            }
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({12, 11, 4, 3});
                Array<uint64_t> output(poly_modulus_degree, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.decrypt_mod_t(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 0})) return false;
            }
        }

        {
            size_t poly_modulus_degree = 2;
            Modulus plain_t(3);
            RNSTool rns_tool(poly_modulus_degree, to_rns_base({5, 7, 11}), plain_t);
            RNSTool rns_tool_cpu = rns_tool.clone();
            if (device) rns_tool.to_device_inplace();

            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({0, 0, 0, 0, 0, 0});
                Array<uint64_t> output(poly_modulus_degree, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.decrypt_mod_t(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {0, 0})) return false;
            }
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({1, 2, 1, 2, 1, 2});
                Array<uint64_t> output(poly_modulus_degree, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.decrypt_mod_t(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2})) return false;
            }
            {
                Array<uint64_t> input = Array<uint64_t>::from_vector({0, 1, 0, 0, 4, 0});
                Array<uint64_t> output(poly_modulus_degree, false);
                if (device) input.to_device_inplace();
                if (device) output.to_device_inplace();
                rns_tool.decrypt_mod_t(input.const_reference(), output.reference());
                if (device) input.to_host_inplace();
                if (device) output.to_host_inplace();
                if (!same_array_vector(output.const_reference(), {1, 2})) return false;
            }
        }

        return true;

    }

    TEST(RNSToolTest, HostDecryptModT) {
        ASSERT_TRUE(test_body_decrypt_mod_t(false));
    }

    TEST(RNSToolTest, DeviceDecryptModT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        ASSERT_TRUE(test_body_decrypt_mod_t(true));
        MemoryPool::Destroy();
    }

}