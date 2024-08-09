#include <gtest/gtest.h>
#include "../test.h"
#include "../../src/utils/basics.h"
#include "../../src/utils/box.h"

using namespace troy::utils;

namespace basics {

    TEST(Basics, HostOnDevice) {
        EXPECT_FALSE(on_device());
    }

    __global__
    void kernel_on_device(Slice<bool> result) {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        result[index] = on_device();
    }

    TEST(Basics, DeviceOnDevice) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        MemoryPoolHandle pool = MemoryPool::GlobalPool();
        Array<bool> r(16, true, pool);
        utils::set_device(r.device_index());
        kernel_on_device<<<4, 4>>>(r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    TEST(Basics, HostBits) {

        bool device = false;

        EXPECT_EQ(get_significant_bit_count(0), 0);
        EXPECT_EQ(get_significant_bit_count(3), 2);
        EXPECT_EQ(get_significant_bit_count(0x6666), 15);

        {
            uint64_t sl_data[2]{1, 1};
            ConstSlice<uint64_t> sl(sl_data, 2, device, nullptr);
            EXPECT_EQ(get_significant_bit_count_uint(sl), 65);
            EXPECT_EQ(get_significant_uint64_count_uint(sl), 2);
            EXPECT_EQ(get_nonzero_uint64_count_uint(sl), 2);
        }
        
        {
            uint64_t sl_data[5]{1, 1, 31, 0, 0};
            ConstSlice<uint64_t> sl(sl_data, 5, device, nullptr);
            EXPECT_EQ(get_significant_bit_count_uint(sl), 128 + 5);
            EXPECT_EQ(get_significant_uint64_count_uint(sl), 3);
            EXPECT_EQ(get_nonzero_uint64_count_uint(sl), 3);
        }

        {
            uint64_t sl_data[5]{1, 1, 31, 0, 3};
            ConstSlice<uint64_t> sl(sl_data, 5, device, nullptr);
            EXPECT_EQ(get_significant_bit_count_uint(sl), 256 + 2);
            EXPECT_EQ(get_significant_uint64_count_uint(sl), 5);
            EXPECT_EQ(get_nonzero_uint64_count_uint(sl), 4);
        }

        EXPECT_EQ(get_power_of_two(0), -1);
        EXPECT_EQ(get_power_of_two(16), 4);
        EXPECT_EQ(get_power_of_two(0x6666), -1);
    }

    __global__ void kernel_bits(Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;
        
        bool device = true;

        KERNEL_EXPECT_EQ(get_significant_bit_count(0), 0);
        KERNEL_EXPECT_EQ(get_significant_bit_count(3), 2);
        KERNEL_EXPECT_EQ(get_significant_bit_count(0x6666), 15);

        {
            uint64_t sl_data[2]{1, 1};
            ConstSlice<uint64_t> sl(sl_data, 2, device, nullptr);
            KERNEL_EXPECT_EQ(get_significant_bit_count_uint(sl), 65);
            KERNEL_EXPECT_EQ(get_significant_uint64_count_uint(sl), 2);
            KERNEL_EXPECT_EQ(get_nonzero_uint64_count_uint(sl), 2);
        }
        
        {
            uint64_t sl_data[5]{1, 1, 31, 0, 0};
            ConstSlice<uint64_t> sl(sl_data, 5, device, nullptr);
            KERNEL_EXPECT_EQ(get_significant_bit_count_uint(sl), 128 + 5);
            KERNEL_EXPECT_EQ(get_significant_uint64_count_uint(sl), 3);
            KERNEL_EXPECT_EQ(get_nonzero_uint64_count_uint(sl), 3);
        }

        {
            uint64_t sl_data[5]{1, 1, 31, 0, 3};
            ConstSlice<uint64_t> sl(sl_data, 5, device, nullptr);
            KERNEL_EXPECT_EQ(get_significant_bit_count_uint(sl), 256 + 2);
            KERNEL_EXPECT_EQ(get_significant_uint64_count_uint(sl), 5);
            KERNEL_EXPECT_EQ(get_nonzero_uint64_count_uint(sl), 4);
        }

        KERNEL_EXPECT_EQ(get_power_of_two(0), -1);
        KERNEL_EXPECT_EQ(get_power_of_two(16), 4);
        KERNEL_EXPECT_EQ(get_power_of_two(0x6666), -1);

    }

    TEST(Basics, DeviceBits) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_bits<<<4, 4>>>(r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__ 
    bool test_add_uint64_carry(uint64_t x, uint64_t y, uint8_t carry, uint64_t out, uint8_t out_carry) {
        uint64_t result;
        return out_carry == add_uint64_carry(x, y, carry, result)
            && result == out;
    }

    __host__ __device__
    bool test_add_uint64(uint64_t x, uint64_t y, uint64_t out, uint8_t out_carry) {
        uint64_t result;
        return out_carry == add_uint64(x, y, result)
            && result == out;
    }

    __host__ __device__
    bool test_add_uint128(ConstSlice<uint64_t> o1, ConstSlice<uint64_t> o2, ConstSlice<uint64_t> o3, uint8_t carry) {
        uint64_t result[2];
        Slice<uint64_t> result_slice(result, 2, on_device(), nullptr);
        uint8_t c = add_uint128(o1, o2, result_slice);
        return c == carry && result_slice[0] == o3[0] && result_slice[1] == o3[1];
    }

    __global__ void kernel_add(Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;
        
        bool device = true;

        KERNEL_EXPECT_TRUE(test_add_uint64_carry(1, 1, 0, 2, 0));
        KERNEL_EXPECT_TRUE(test_add_uint64_carry(1, 1, 1, 3, 0));
        KERNEL_EXPECT_TRUE(test_add_uint64_carry(0xffffffffffffffff, 0x1, 0, 0, 1));
        KERNEL_EXPECT_TRUE(test_add_uint64_carry(0xffffffffffffffff, 0x1, 1, 1, 1));
        KERNEL_EXPECT_TRUE(test_add_uint64_carry(0xffffffffffffffff, 0xffffffffffffffff, 1, 0xffffffffffffffff, 1));

        KERNEL_EXPECT_TRUE(test_add_uint64(1, 1, 2, 0));
        KERNEL_EXPECT_TRUE(test_add_uint64(0xffffffffffffffff, 0x1, 0, 1));

        uint64_t o1[2]; ConstSlice<uint64_t> o1_slice(o1, 2, device, nullptr);
        uint64_t o2[2]; ConstSlice<uint64_t> o2_slice(o2, 2, device, nullptr);
        uint64_t o3[2]; ConstSlice<uint64_t> o3_slice(o3, 2, device, nullptr);
        
        o1[0] = 0x3418e9072c3a0a61; o1[1] = 0xca5ec19b9e101da3;
        o2[0] = 0xae3db791415d70f3; o2[1] = 0xe8163d4482118bd;
        o3[0] = 0xe256a0986d977b54; o3[1] = 0xd8e0256fe6313660;
        KERNEL_EXPECT_TRUE(test_add_uint128(o1_slice, o2_slice, o3_slice, 0));

        o1[0] = 0x5065d944029b0242; o1[1] = 0xdd1e40b9f3532fc8;
        o2[0] = 0x61f9f5c87eafc04c; o2[1] = 0xafbb16475d48fbb5;
        o3[0] = 0xb25fcf0c814ac28e; o3[1] = 0x8cd95701509c2b7d;
        KERNEL_EXPECT_TRUE(test_add_uint128(o1_slice, o2_slice, o3_slice, 1));

    }

    TEST(Basics, HostAdd) {

        bool device = false;

        EXPECT_TRUE(test_add_uint64_carry(1, 1, 0, 2, 0));
        EXPECT_TRUE(test_add_uint64_carry(1, 1, 1, 3, 0));
        EXPECT_TRUE(test_add_uint64_carry(0xffffffffffffffff, 0x1, 0, 0, 1));
        EXPECT_TRUE(test_add_uint64_carry(0xffffffffffffffff, 0x1, 1, 1, 1));
        EXPECT_TRUE(test_add_uint64_carry(0xffffffffffffffff, 0xffffffffffffffff, 1, 0xffffffffffffffff, 1));

        EXPECT_TRUE(test_add_uint64(1, 1, 2, 0));
        EXPECT_TRUE(test_add_uint64(0xffffffffffffffff, 0x1, 0, 1));

        uint64_t o1[2]; ConstSlice<uint64_t> o1_slice(o1, 2, device, nullptr);
        uint64_t o2[2]; ConstSlice<uint64_t> o2_slice(o2, 2, device, nullptr);
        uint64_t o3[2]; ConstSlice<uint64_t> o3_slice(o3, 2, device, nullptr);
        
        o1[0] = 0x3418e9072c3a0a61; o1[1] = 0xca5ec19b9e101da3;
        o2[0] = 0xae3db791415d70f3; o2[1] = 0xe8163d4482118bd;
        o3[0] = 0xe256a0986d977b54; o3[1] = 0xd8e0256fe6313660;
        EXPECT_TRUE(test_add_uint128(o1_slice, o2_slice, o3_slice, 0));

        o1[0] = 0x5065d944029b0242; o1[1] = 0xdd1e40b9f3532fc8;
        o2[0] = 0x61f9f5c87eafc04c; o2[1] = 0xafbb16475d48fbb5;
        o3[0] = 0xb25fcf0c814ac28e; o3[1] = 0x8cd95701509c2b7d;
        EXPECT_TRUE(test_add_uint128(o1_slice, o2_slice, o3_slice, 1));
    }

    TEST(Basics, DeviceAdd) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_add<<<4, 4>>>(r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__ 
    bool test_multiply_uint64_uint64(uint64_t x, uint64_t y, uint64_t r0, uint64_t r1) {
        uint64_t result[2];
        Slice<uint64_t> result_slice(result, 2, on_device(), nullptr);
        multiply_uint64_uint64(x, y, result_slice);
        return result_slice[0] == r0 && result_slice[1] == r1;
    }

    __host__
    bool test_multiply_uint_uint64(ConstSlice<uint64_t> x, uint64_t y, size_t rlen, ConstSlice<uint64_t> r) {
        Array<uint64_t> result(rlen, on_device(), nullptr);
        multiply_uint_uint64(x, y, result.reference());
        for (size_t i = 0; i < rlen; i++) {
            if (result[i] != r[i]) {
                return false;
            }
        }
        return true;
    }

    __host__
    bool test_multiply_uint(ConstSlice<uint64_t> x, ConstSlice<uint64_t> y, size_t rlen, ConstSlice<uint64_t> r) {
        Array<uint64_t> result(rlen, on_device(), nullptr);
        multiply_uint(x, y, result.reference());
        for (size_t i = 0; i < rlen; i++) {
            if (result[i] != r[i]) {
                return false;
            }
        }
        return true;
    }

    TEST(Basics, HostMultiply) {

        bool device = false;

        EXPECT_TRUE(test_multiply_uint64_uint64(0, 0, 0, 0));
        EXPECT_TRUE(test_multiply_uint64_uint64(4, 4, 16, 0));
        EXPECT_TRUE(test_multiply_uint64_uint64(0xdeadbeefdeadbeef, 0x1234567890abcdef, 0xd3b89abeffbfa421, 0xfd5bdeee3ceb48c));
    
        {
            uint64_t x[0]; 
            uint64_t y = 0; 
            size_t r = 0;
            uint64_t result[0];
            EXPECT_TRUE(test_multiply_uint_uint64(ConstSlice<uint64_t>(x, 0, device, nullptr), y, r, ConstSlice<uint64_t>(result, 0, device, nullptr)));
        }

        {
            uint64_t x[0]; 
            uint64_t y = 1; 
            size_t r = 0;
            uint64_t result[0];
            EXPECT_TRUE(test_multiply_uint_uint64(ConstSlice<uint64_t>(x, 0, device, nullptr), y, r, ConstSlice<uint64_t>(result, 0, device, nullptr)));
        }

        {
            uint64_t x[0]; 
            uint64_t y = 0xbead005946621c1c; 
            size_t r = 1;
            uint64_t result[1]{0};
            EXPECT_TRUE(test_multiply_uint_uint64(ConstSlice<uint64_t>(x, 0, device, nullptr), y, r, ConstSlice<uint64_t>(result, 1, device, nullptr)));
        }

        {
            uint64_t x[1]; x[0] = 0x38603a368dc7161c;
            uint64_t y = 0x300b86532dbe7240;
            size_t r = 1;
            uint64_t result[1]; result[0] = 0x1db6e47f6e65ff00;
            EXPECT_TRUE(test_multiply_uint_uint64(ConstSlice<uint64_t>(x, 1, device, nullptr), y, r, ConstSlice<uint64_t>(result, 1, device, nullptr)));
        }

        {
            uint64_t x[1]; x[0] = 0x38603a368dc7161c;
            uint64_t y = 0x300b86532dbe7240;
            size_t r = 2;
            uint64_t result[1]; result[0] = 0x1db6e47f6e65ff00; result[1] = 0xa9494a16aabb469;
            EXPECT_TRUE(test_multiply_uint_uint64(ConstSlice<uint64_t>(x, 1, device, nullptr), y, r, ConstSlice<uint64_t>(result, 2, device, nullptr)));
        }

        {
            uint64_t x[2]; x[0] = 0xab0bc09f7b288a5e; x[1] = 0x1613bdbc5066de5c;
            uint64_t y = 0x611bbb8ef414913d;
            size_t r = 3;
            uint64_t result[3]{0x9b38e7f2b6603666, 0xe96b9f5536fba9a, 0x85fdf261cebd933};
            EXPECT_TRUE(test_multiply_uint_uint64(ConstSlice<uint64_t>(x, 2, device, nullptr), y, r, ConstSlice<uint64_t>(result, 3, device, nullptr)));
        }

        {
            uint64_t x[0];
            uint64_t y[0];
            size_t r = 0;
            uint64_t result[0];
            EXPECT_TRUE(test_multiply_uint(ConstSlice<uint64_t>(x, 0, device, nullptr), ConstSlice<uint64_t>(y, 0, device, nullptr), r, ConstSlice<uint64_t>(result, 0, device, nullptr)));
        }

        {
            uint64_t x[0];
            uint64_t y[1]{0xbead005946621c1c};
            size_t r = 1;
            uint64_t result[1]{0};
            EXPECT_TRUE(test_multiply_uint(ConstSlice<uint64_t>(x, 0, device, nullptr), ConstSlice<uint64_t>(y, 1, device, nullptr), r, ConstSlice<uint64_t>(result, 1, device, nullptr)));
        }

        {
            uint64_t x[3]{0x2ab4f6ef5c8d6205, 0xfb49f1a6128fbd46, 0x66b72c7f86d79dd8};
            uint64_t y[3]{0xf6639b8f1e77ba65, 0xeda2107393685f21, 0xd7df5e486c4f352d};
            size_t r = 3;
            uint64_t result[3]{0x2e2db4ae63524df9, 0x2b55e17efb94b806, 0xc3b4577b011a8cf4};
            EXPECT_TRUE(test_multiply_uint(ConstSlice<uint64_t>(x, 3, device, nullptr), ConstSlice<uint64_t>(y, 3, device, nullptr), r, ConstSlice<uint64_t>(result, 3, device, nullptr)));
        }

        {
            uint64_t x[3]{0x2ab4f6ef5c8d6205, 0xfb49f1a6128fbd46, 0x66b72c7f86d79dd8};
            uint64_t y[3]{0xf6639b8f1e77ba65, 0xeda2107393685f21, 0xd7df5e486c4f352d};
            size_t r = 6;
            uint64_t result[6]{0x2e2db4ae63524df9, 0x2b55e17efb94b806, 0xc3b4577b011a8cf4, 0xa3e9fd16fdb71a0a, 0xb5a777d46f14340d, 0x569d75c32ea5f167};
            EXPECT_TRUE(test_multiply_uint(ConstSlice<uint64_t>(x, 3, device, nullptr), ConstSlice<uint64_t>(y, 3, device, nullptr), r, ConstSlice<uint64_t>(result, 6, device, nullptr)));
        }
    }

    __global__ void kernel_multiply(Slice<bool> test_result) {
        
        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_multiply_uint64_uint64(0, 0, 0, 0));
        KERNEL_EXPECT_TRUE(test_multiply_uint64_uint64(4, 4, 16, 0));
        KERNEL_EXPECT_TRUE(test_multiply_uint64_uint64(0xdeadbeefdeadbeef, 0x1234567890abcdef, 0xd3b89abeffbfa421, 0xfd5bdeee3ceb48c));

    }

    TEST(Basics, DeviceMultiply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_multiply<<<4, 4>>>(r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__
    bool test_divide_uint_inplace(ConstSlice<uint64_t> numerator, ConstSlice<uint64_t> denominator, ConstSlice<uint64_t> quotient, ConstSlice<uint64_t> remainder) {
        Array<uint64_t> n(numerator.size(), on_device(), nullptr);
        n.copy_from_slice(numerator);
        Array<uint64_t> d(denominator.size(), on_device(), nullptr);
        d.copy_from_slice(denominator);
        Array<uint64_t> q(numerator.size(), on_device(), nullptr);
        divide_uint_inplace(n.reference(), d.const_reference(), q.reference());
        for (size_t i = 0; i < remainder.size(); i++) {
            if (n[i] != remainder[i]) {
                return false;
            }
        }
        for (size_t i = 0; i < quotient.size(); i++) {
            if (q[i] != quotient[i]) {
                return false;
            }
        }
        return true;
    }

    TEST(Basics, HostDivide) {

        bool device = false;

        {
            uint64_t numerator[1]{1};
            uint64_t denominator[1]{1};
            uint64_t quotient[1]{1};
            uint64_t remainder[1]{0};
            EXPECT_TRUE(test_divide_uint_inplace(ConstSlice<uint64_t>(numerator, 1, device, nullptr), ConstSlice<uint64_t>(denominator, 1, device, nullptr), ConstSlice<uint64_t>(quotient, 1, device, nullptr), ConstSlice<uint64_t>(remainder, 1, device, nullptr)));
        }

        {
            uint64_t numerator[1]{16};
            uint64_t denominator[1]{1};
            uint64_t quotient[1]{16};
            uint64_t remainder[1]{0};
            EXPECT_TRUE(test_divide_uint_inplace(ConstSlice<uint64_t>(numerator, 1, device, nullptr), ConstSlice<uint64_t>(denominator, 1, device, nullptr), ConstSlice<uint64_t>(quotient, 1, device, nullptr), ConstSlice<uint64_t>(remainder, 1, device, nullptr)));
        }
        
        {
            uint64_t numerator[1]{16};
            uint64_t denominator[1]{5};
            uint64_t quotient[1]{3};
            uint64_t remainder[1]{1};
            EXPECT_TRUE(test_divide_uint_inplace(ConstSlice<uint64_t>(numerator, 1, device, nullptr), ConstSlice<uint64_t>(denominator, 1, device, nullptr), ConstSlice<uint64_t>(quotient, 1, device, nullptr), ConstSlice<uint64_t>(remainder, 1, device, nullptr)));
        }

        {
            uint64_t numerator[2]{0x7d6112ec7f1902b, 0x72870865d354e6f1};
            uint64_t denominator[2]{0xfe9aaaf7d7b4};
            uint64_t quotient[2]{0xc51b92990d851bd2, 0x7327};
            uint64_t remainder[1]{0xeafb305ea283};
            EXPECT_TRUE(test_divide_uint_inplace(ConstSlice<uint64_t>(numerator, 2, device, nullptr), ConstSlice<uint64_t>(denominator, 2, device, nullptr), ConstSlice<uint64_t>(quotient, 2, device, nullptr), ConstSlice<uint64_t>(remainder, 1, device, nullptr)));
        }

        {
            uint64_t numerator[4]{0x153e235f0fd3f123, 0x596ab8b0c3c7b048, 0xd00750c13822be9d, 0xde061e96884b8a96};
            uint64_t denominator[2]{0xe239675985a60044, 0xfc9a15316};
            uint64_t quotient[3]{0xf5c6b5a03514f4a3, 0xe31f8f9a2e9b27c6, 0xe102bb3};
            uint64_t remainder[2]{0x89fdf476a590f5d7, 0x8c48bb986};
            EXPECT_TRUE(test_divide_uint_inplace(ConstSlice<uint64_t>(numerator, 4, device, nullptr), ConstSlice<uint64_t>(denominator, 2, device, nullptr), ConstSlice<uint64_t>(quotient, 3, device, nullptr), ConstSlice<uint64_t>(remainder, 2, device, nullptr)));
        }
        
        {
            uint64_t numerator[1]{0x377aaf500976c};
            uint64_t denominator[2]{0x24df8dd80ab231e4, 0x7844224f4};
            uint64_t quotient[0]{};
            uint64_t remainder[1]{0x377aaf500976c};
            EXPECT_TRUE(test_divide_uint_inplace(ConstSlice<uint64_t>(numerator, 1, device, nullptr), ConstSlice<uint64_t>(denominator, 2, device, nullptr), ConstSlice<uint64_t>(quotient, 0, device, nullptr), ConstSlice<uint64_t>(remainder, 1, device, nullptr)));
        }
    }

}

// int main() {
//     utils::set_device(0);
//     RUN_TEST(basics, host_on_device);
//     RUN_TEST(basics, device_on_device);
//     return 0;
// }