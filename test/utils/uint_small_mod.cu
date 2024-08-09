#include <gtest/gtest.h>
#include "../../src/utils/uint_small_mod.h"
#include "../test.h"

using namespace troy::utils;
using namespace troy;

namespace uint_small_mod {

    __host__ __device__
    bool test_increment_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(1, increment_uint64_mod(0, mod));
            RETURN_EQ(0, increment_uint64_mod(1, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(1, increment_uint64_mod(0, mod));
            RETURN_EQ(2, increment_uint64_mod(1, mod));
            RETURN_EQ(0, increment_uint64_mod(0xffff, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(1, increment_uint64_mod(0, mod));
            RETURN_EQ(0, increment_uint64_mod(2305843009211596800ULL, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostIncrementUint64Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0x10000);
        modulus[2] = Modulus(2305843009211596801ULL);
        EXPECT_TRUE(test_increment_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_increment_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_increment_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceIncrementUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0x10000);
        modulus[2] = Modulus(2305843009211596801ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_increment_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_decrement_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(0, decrement_uint64_mod(1, mod));
            RETURN_EQ(1, decrement_uint64_mod(0, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(0, decrement_uint64_mod(1, mod));
            RETURN_EQ(1, decrement_uint64_mod(2, mod));
            RETURN_EQ(0xffff, decrement_uint64_mod(0, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(0, decrement_uint64_mod(1, mod));
            RETURN_EQ(2305843009211596800ULL, decrement_uint64_mod(0, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostDecrementUint64Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0x10000);
        modulus[2] = Modulus(2305843009211596801ULL);
        EXPECT_TRUE(test_decrement_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_decrement_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_decrement_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceDecrementUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0x10000);
        modulus[2] = Modulus(2305843009211596801ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_decrement_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_negate_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(0, negate_uint64_mod(0, mod));
            RETURN_EQ(1, negate_uint64_mod(1, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(0, negate_uint64_mod(0, mod));
            RETURN_EQ(0xfffe, negate_uint64_mod(1, mod));
            RETURN_EQ(0x1, negate_uint64_mod(0xfffe, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(0, negate_uint64_mod(0, mod));
            RETURN_EQ(0xffff, negate_uint64_mod(1, mod));
            RETURN_EQ(0x1, negate_uint64_mod(0xffff, mod));
        } {
            const Modulus& mod = moduli[3];
            RETURN_EQ(0, negate_uint64_mod(0, mod));
            RETURN_EQ(2305843009211596800ULL, negate_uint64_mod(1, mod));
        }
        return true;
    }
    
    TEST(UintSmallMod, HostNegateUint64Mod) {
        Array<Modulus> modulus(4, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0xFFFFULL);
        modulus[2] = Modulus(0x10000);
        modulus[3] = Modulus(2305843009211596801ULL);
        EXPECT_TRUE(test_negate_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_negate_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_negate_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceNegateUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(4, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0xFFFFULL);
        modulus[2] = Modulus(0x10000);
        modulus[3] = Modulus(2305843009211596801ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_negate_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_divide2_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(0, divide2_uint64_mod(0, mod));
            RETURN_EQ(2, divide2_uint64_mod(1, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(11, divide2_uint64_mod(5, mod));
            RETURN_EQ(4, divide2_uint64_mod(8, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(0x800000000000000ULL, divide2_uint64_mod(1, mod));
            RETURN_EQ(0x800000000000001ULL, divide2_uint64_mod(3, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostDivide2Uint64Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(3);
        modulus[1] = Modulus(17);
        modulus[2] = Modulus(0xFFFFFFFFFFFFFFFULL);
        EXPECT_TRUE(test_divide2_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_divide2_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_divide2_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceDivide2Uint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(3);
        modulus[1] = Modulus(17);
        modulus[2] = Modulus(0xFFFFFFFFFFFFFFFULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_divide2_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());        utils::stream_sync();
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_add_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(0, add_uint64_mod(0, 0, mod));
            RETURN_EQ(1, add_uint64_mod(0, 1, mod));
            RETURN_EQ(1, add_uint64_mod(1, 0, mod));
            RETURN_EQ(0, add_uint64_mod(1, 1, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(0, add_uint64_mod(0, 0, mod));
            RETURN_EQ(1, add_uint64_mod(0, 1, mod));
            RETURN_EQ(1, add_uint64_mod(1, 0, mod));
            RETURN_EQ(2, add_uint64_mod(1, 1, mod));
            RETURN_EQ(4, add_uint64_mod(7, 7, mod));
            RETURN_EQ(3, add_uint64_mod(6, 7, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(0, add_uint64_mod(0, 0, mod));
            RETURN_EQ(1, add_uint64_mod(0, 1, mod));
            RETURN_EQ(1, add_uint64_mod(1, 0, mod));
            RETURN_EQ(2, add_uint64_mod(1, 1, mod));
            RETURN_EQ(0ULL, add_uint64_mod(1152921504605798400ULL, 1152921504605798401ULL, mod));
            RETURN_EQ(1ULL, add_uint64_mod(1152921504605798401ULL, 1152921504605798401ULL, mod));
            RETURN_EQ(2305843009211596799ULL, add_uint64_mod(2305843009211596800ULL, 2305843009211596800ULL, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostAddUint64Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(10);
        modulus[2] = Modulus(2305843009211596801ULL);
        EXPECT_TRUE(test_add_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_add_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_add_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceAddUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(10);
        modulus[2] = Modulus(2305843009211596801ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_add_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();

        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_sub_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(0ULL, sub_uint64_mod(0, 0, mod));
            RETURN_EQ(1ULL, sub_uint64_mod(0, 1, mod));
            RETURN_EQ(1ULL, sub_uint64_mod(1, 0, mod));
            RETURN_EQ(0ULL, sub_uint64_mod(1, 1, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(0ULL, sub_uint64_mod(0, 0, mod));
            RETURN_EQ(9ULL, sub_uint64_mod(0, 1, mod));
            RETURN_EQ(1ULL, sub_uint64_mod(1, 0, mod));
            RETURN_EQ(0ULL, sub_uint64_mod(1, 1, mod));
            RETURN_EQ(0ULL, sub_uint64_mod(7, 7, mod));
            RETURN_EQ(9ULL, sub_uint64_mod(6, 7, mod));
            RETURN_EQ(1ULL, sub_uint64_mod(7, 6, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(0ULL, sub_uint64_mod(0, 0, mod));
            RETURN_EQ(2305843009211596800ULL, sub_uint64_mod(0, 1, mod));
            RETURN_EQ(1ULL, sub_uint64_mod(1, 0, mod));
            RETURN_EQ(0ULL, sub_uint64_mod(1, 1, mod));
            RETURN_EQ(2305843009211596800ULL, sub_uint64_mod(1152921504605798400ULL, 1152921504605798401ULL, mod));
            RETURN_EQ(1ULL, sub_uint64_mod(1152921504605798401ULL, 1152921504605798400ULL, mod));
            RETURN_EQ(0ULL, sub_uint64_mod(1152921504605798401ULL, 1152921504605798401ULL, mod));
            RETURN_EQ(0ULL, sub_uint64_mod(2305843009211596800ULL, 2305843009211596800ULL, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostSubUint64Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(10);
        modulus[2] = Modulus(2305843009211596801ULL);
        EXPECT_TRUE(test_sub_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_sub_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_sub_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceSubUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(10);
        modulus[2] = Modulus(2305843009211596801ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_sub_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_barrett_reduce_uint128_mod(ConstSlice<Modulus> moduli) {
        uint64_t input[2]; ConstSlice<uint64_t> input_slice(input, 2, on_device(), nullptr);
        {
            const Modulus& mod = moduli[0];
            input[0] = 0;
            input[1] = 0;
            RETURN_EQ(0ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 1;
            input[1] = 0;
            RETURN_EQ(1ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 0xFFFFFFFFFFFFFFFFULL;
            input[1] = 0xFFFFFFFFFFFFFFFFULL;
            RETURN_EQ(1ULL, barrett_reduce_uint128(input_slice, mod));
        } {
            const Modulus& mod = moduli[1];
            input[0] = 0;
            input[1] = 0;
            RETURN_EQ(0ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 1;
            input[1] = 0;
            RETURN_EQ(1ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 123;
            input[1] = 456;
            RETURN_EQ(0ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 0xFFFFFFFFFFFFFFFFULL;
            input[1] = 0xFFFFFFFFFFFFFFFFULL;
            RETURN_EQ(0ULL, barrett_reduce_uint128(input_slice, mod));
        } {
            const Modulus& mod = moduli[2];
            input[0] = 0;
            input[1] = 0;
            RETURN_EQ(0ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 1;
            input[1] = 0;
            RETURN_EQ(1ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 123;
            input[1] = 456;
            RETURN_EQ(8722750765283ULL, barrett_reduce_uint128(input_slice, mod));
            input[0] = 24242424242424;
            input[1] = 79797979797979;
            RETURN_EQ(1010101010101ULL, barrett_reduce_uint128(input_slice, mod));
        }
        return true;
    }


    TEST(UintSmallMod, HostBarrettReduceUint128Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(3);
        modulus[2] = Modulus(13131313131313ULL);
        EXPECT_TRUE(test_barrett_reduce_uint128_mod(modulus.const_reference()));
    }

    __global__ void kernel_barrett_reduce_uint128_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_barrett_reduce_uint128_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceBarrettReduceUint128Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(3);
        modulus[2] = Modulus(13131313131313ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_barrett_reduce_uint128_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_multiply_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(0ULL, multiply_uint64_mod(0, 0, mod));
            RETURN_EQ(0ULL, multiply_uint64_mod(0, 1, mod));
            RETURN_EQ(0ULL, multiply_uint64_mod(1, 0, mod));
            RETURN_EQ(1ULL, multiply_uint64_mod(1, 1, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(0ULL, multiply_uint64_mod(0, 0, mod));
            RETURN_EQ(0ULL, multiply_uint64_mod(0, 1, mod));
            RETURN_EQ(0ULL, multiply_uint64_mod(1, 0, mod));
            RETURN_EQ(1ULL, multiply_uint64_mod(1, 1, mod));
            RETURN_EQ(9ULL, multiply_uint64_mod(7, 7, mod));
            RETURN_EQ(2ULL, multiply_uint64_mod(6, 7, mod));
            RETURN_EQ(2ULL, multiply_uint64_mod(7, 6, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(0ULL, multiply_uint64_mod(0, 0, mod));
            RETURN_EQ(0ULL, multiply_uint64_mod(0, 1, mod));
            RETURN_EQ(0ULL, multiply_uint64_mod(1, 0, mod));
            RETURN_EQ(1ULL, multiply_uint64_mod(1, 1, mod));
            RETURN_EQ(576460752302899200ULL, multiply_uint64_mod(1152921504605798400ULL, 1152921504605798401ULL, mod));
            RETURN_EQ(576460752302899200ULL, multiply_uint64_mod(1152921504605798401ULL, 1152921504605798400ULL, mod));
            RETURN_EQ(1729382256908697601ULL, multiply_uint64_mod(1152921504605798401ULL, 1152921504605798401ULL, mod));
            RETURN_EQ(1ULL, multiply_uint64_mod(2305843009211596800ULL, 2305843009211596800ULL, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostMultiplyUint64Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(10);
        modulus[2] = Modulus(2305843009211596801ULL);
        EXPECT_TRUE(test_multiply_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_multiply_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_multiply_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceMultiplyUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(10);
        modulus[2] = Modulus(2305843009211596801ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_multiply_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }



    __host__ __device__
    bool test_multiply_add_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(0ULL, multiply_add_uint64_mod(0, 0, 0, mod));
            RETURN_EQ(0ULL, multiply_add_uint64_mod(1, 0, 0, mod));
            RETURN_EQ(0ULL, multiply_add_uint64_mod(0, 1, 0, mod));
            RETURN_EQ(1ULL, multiply_add_uint64_mod(0, 0, 1, mod));
            RETURN_EQ(3ULL, multiply_add_uint64_mod(3, 4, 5, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(0ULL, multiply_add_uint64_mod(0, 0, 0, mod));
            RETURN_EQ(0ULL, multiply_add_uint64_mod(1, 0, 0, mod));
            RETURN_EQ(0ULL, multiply_add_uint64_mod(0, 1, 0, mod));
            RETURN_EQ(1ULL, multiply_add_uint64_mod(0, 0, 1, mod));
            RETURN_EQ(0ULL, multiply_add_uint64_mod(mod.value() - 1, mod.value() - 1, mod.value() - 1, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostMultiplyAddUint64Mod) {
        Array<Modulus> modulus(2, false, nullptr);
        modulus[0] = Modulus(7);
        modulus[1] = Modulus(0x1FFFFFFFFFFFFFFFULL);
        EXPECT_TRUE(test_multiply_add_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_multiply_add_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_multiply_add_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceMultiplyAddUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(2, false, nullptr);
        modulus[0] = Modulus(7);
        modulus[1] = Modulus(0x1FFFFFFFFFFFFFFFULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_multiply_add_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

    __host__ __device__
    bool test_modulo_uint(ConstSlice<Modulus> moduli) {
        {
            uint64_t value[3]; Slice<uint64_t> value_slice(value, 3, on_device(), nullptr);
            const Modulus& mod = moduli[0];
            value[0] = 0;
            value[1] = 0;
            value[2] = 0;
            modulo_uint_inplace(value_slice, mod);
            RETURN_EQ(0ULL, value[0]);
            RETURN_EQ(0ULL, value[1]);
            RETURN_EQ(0ULL, value[2]);
            value[0] = 1;
            value[1] = 0;
            value[2] = 0;
            modulo_uint_inplace(value_slice, mod);
            RETURN_EQ(1ULL, value[0]);
            RETURN_EQ(0ULL, value[1]);
            RETURN_EQ(0ULL, value[2]);
            value[0] = 2;
            value[1] = 0;
            value[2] = 0;
            modulo_uint_inplace(value_slice, mod);
            RETURN_EQ(0ULL, value[0]);
            RETURN_EQ(0ULL, value[1]);
            RETURN_EQ(0ULL, value[2]);
            value[0] = 3;
            value[1] = 0;
            value[2] = 0;
            modulo_uint_inplace(value_slice, mod);
            RETURN_EQ(1ULL, value[0]);
            RETURN_EQ(0ULL, value[1]);
            RETURN_EQ(0ULL, value[2]);
        } {
            uint64_t value[3]; Slice<uint64_t> value_slice(value, 3, on_device(), nullptr);
            const Modulus& mod = moduli[1];
            value[0] = 9585656442714717620ul;
            value[1] = 1817697005049051848;
            value[2] = 0;
            modulo_uint_inplace(value_slice, mod);
            RETURN_EQ(65143ULL, value[0]);
            RETURN_EQ(0ULL, value[1]);
            RETURN_EQ(0ULL, value[2]);
        } {
            uint64_t value[3]; Slice<uint64_t> value_slice(value, 3, on_device(), nullptr);
            const Modulus& mod = moduli[2];
            value[0] = 9585656442714717620ul;
            value[1] = 1817697005049051848;
            value[2] = 0;
            modulo_uint_inplace(value_slice, mod);
            RETURN_EQ(0xDB4ULL, value[0]);
            RETURN_EQ(0ULL, value[1]);
            RETURN_EQ(0ULL, value[2]);
        } {
            uint64_t value[4]; Slice<uint64_t> value_slice(value, 4, on_device(), nullptr);
            const Modulus& mod = moduli[3];
            value[0] = 9585656442714717620ul;
            value[1] = 1817697005049051848;
            value[2] = 14447416709120365380ul;
            value[3] = 67450014862939159;
            modulo_uint_inplace(value_slice, mod);
            RETURN_EQ(124510066632001ULL, value[0]);
            RETURN_EQ(0ULL, value[1]);
            RETURN_EQ(0ULL, value[2]);
            RETURN_EQ(0ULL, value[3]);
        }
        return true;
    }


    TEST(UintSmallMod, HostModuloUintMod) {
        Array<Modulus> modulus(4, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0xFFFF);
        modulus[2] = Modulus(0x1000);
        modulus[3] = Modulus(0xFFFFFFFFC001ULL);
        EXPECT_TRUE(test_modulo_uint(modulus.const_reference()));
    }

    __global__ void kernel_modulo_uint_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_modulo_uint(modulus));

    }
    
    TEST(UintSmallMod, DeviceModuloUintMod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(4, false, nullptr);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(0xFFFF);
        modulus[2] = Modulus(0x1000);
        modulus[3] = Modulus(0xFFFFFFFFC001ULL);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_modulo_uint_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }


    __host__ __device__
    bool test_exponentiate_uint64_mod(ConstSlice<Modulus> moduli) {
        {
            const Modulus& mod = moduli[0];
            RETURN_EQ(1ULL, exponentiate_uint64_mod(1, 0, mod));
            RETURN_EQ(1ULL, exponentiate_uint64_mod(1, 0xFFFFFFFFFFFFFFFFULL, mod));
            RETURN_EQ(3ULL, exponentiate_uint64_mod(2, 0xFFFFFFFFFFFFFFFFULL, mod));
        } {
            const Modulus& mod = moduli[1];
            RETURN_EQ(0ULL, exponentiate_uint64_mod(2, 60, mod));
            RETURN_EQ(0x800000000000000ULL, exponentiate_uint64_mod(2, 59, mod));
        } {
            const Modulus& mod = moduli[2];
            RETURN_EQ(39418477653ULL, exponentiate_uint64_mod(2424242424, 16, mod));
        }
        return true;
    }

    TEST(UintSmallMod, HostExponentiateUint64Mod) {
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(5);
        modulus[1] = Modulus(0x1000000000000000ULL);
        modulus[2] = Modulus(131313131313);
        EXPECT_TRUE(test_exponentiate_uint64_mod(modulus.const_reference()));
    }

    __global__ void kernel_exponentiate_uint64_mod(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        KERNEL_EXPECT_TRUE(test_exponentiate_uint64_mod(modulus));

    }
    
    TEST(UintSmallMod, DeviceExponentiateUint64Mod) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(3, false, nullptr);
        modulus[0] = Modulus(5);
        modulus[1] = Modulus(0x1000000000000000ULL);
        modulus[2] = Modulus(131313131313);
        Array<Modulus> device_modulus = modulus.to_device(MemoryPool::GlobalPool());

        Array<bool> r(16, true, MemoryPool::GlobalPool()); 
        utils::set_device(r.device_index());
        kernel_exponentiate_uint64_mod<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        MemoryPool::Destroy();
    }

}