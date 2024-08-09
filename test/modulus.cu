#include <gtest/gtest.h>
#include "../src/modulus.h"
#include "test.h"

using namespace troy;
using namespace troy::utils;

namespace modulus {

    TEST(Modulus, HostCreateModulus) {

        Modulus mod(0);
        EXPECT_EQ(mod.is_zero(), true);
        EXPECT_EQ(mod.value(), 0);
        EXPECT_EQ(mod.bit_count(), 0);
        EXPECT_EQ(mod.uint64_count(), 1);
        EXPECT_EQ(mod.const_ratio()[0], 0);
        EXPECT_EQ(mod.const_ratio()[1], 0);
        EXPECT_EQ(mod.const_ratio()[2], 0);
        EXPECT_FALSE(mod.is_prime());

        mod = Modulus(3);
        EXPECT_EQ(mod.is_zero(), false);
        EXPECT_EQ(mod.value(), 3);
        EXPECT_EQ(mod.bit_count(), 2);
        EXPECT_EQ(mod.uint64_count(), 1);
        EXPECT_EQ(mod.const_ratio()[0], 6148914691236517205ULL);
        EXPECT_EQ(mod.const_ratio()[1], 6148914691236517205ULL);
        EXPECT_EQ(mod.const_ratio()[2], 1);
        EXPECT_TRUE(mod.is_prime());

        mod = Modulus(0xF00000F00000F);
        EXPECT_EQ(mod.is_zero(), false);
        EXPECT_EQ(mod.value(), 0xF00000F00000F);
        EXPECT_EQ(mod.bit_count(), 52);
        EXPECT_EQ(mod.uint64_count(), 1);
        EXPECT_EQ(mod.const_ratio()[0], 1224979098644774929ULL);
        EXPECT_EQ(mod.const_ratio()[1], 4369ULL);
        EXPECT_EQ(mod.const_ratio()[2], 281470698520321ULL);
        EXPECT_FALSE(mod.is_prime());

        mod = Modulus(0xF00000F000079);
        EXPECT_EQ(mod.is_zero(), false);
        EXPECT_EQ(mod.value(), 0xF00000F000079);
        EXPECT_EQ(mod.bit_count(), 52);
        EXPECT_EQ(mod.uint64_count(), 1);
        EXPECT_EQ(mod.const_ratio()[0], 1224979096621368355ULL);
        EXPECT_EQ(mod.const_ratio()[1], 4369ULL);
        EXPECT_EQ(mod.const_ratio()[2], 1144844808538997ULL);
        EXPECT_TRUE(mod.is_prime());

    }

    
    __global__ void kernel_create_modulus(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        const Modulus* mod = &modulus[0];
        KERNEL_EXPECT_EQ(mod->is_zero(), true);
        KERNEL_EXPECT_EQ(mod->value(), 0);
        KERNEL_EXPECT_EQ(mod->bit_count(), 0);
        KERNEL_EXPECT_EQ(mod->uint64_count(), 1);
        KERNEL_EXPECT_EQ(mod->const_ratio()[0], 0);
        KERNEL_EXPECT_EQ(mod->const_ratio()[1], 0);
        KERNEL_EXPECT_EQ(mod->const_ratio()[2], 0);
        KERNEL_EXPECT_FALSE(mod->is_prime());

        mod = &modulus[1];
        KERNEL_EXPECT_EQ(mod->is_zero(), false);
        KERNEL_EXPECT_EQ(mod->value(), 3);
        KERNEL_EXPECT_EQ(mod->bit_count(), 2);
        KERNEL_EXPECT_EQ(mod->uint64_count(), 1);
        KERNEL_EXPECT_EQ(mod->const_ratio()[0], 6148914691236517205ULL);
        KERNEL_EXPECT_EQ(mod->const_ratio()[1], 6148914691236517205ULL);
        KERNEL_EXPECT_EQ(mod->const_ratio()[2], 1);
        KERNEL_EXPECT_TRUE(mod->is_prime());

        mod = &modulus[2];
        KERNEL_EXPECT_EQ(mod->is_zero(), false);
        KERNEL_EXPECT_EQ(mod->value(), 0xF00000F00000F);
        KERNEL_EXPECT_EQ(mod->bit_count(), 52);
        KERNEL_EXPECT_EQ(mod->uint64_count(), 1);
        KERNEL_EXPECT_EQ(mod->const_ratio()[0], 1224979098644774929ULL);
        KERNEL_EXPECT_EQ(mod->const_ratio()[1], 4369ULL);
        KERNEL_EXPECT_EQ(mod->const_ratio()[2], 281470698520321ULL);
        KERNEL_EXPECT_FALSE(mod->is_prime());

        mod = &modulus[3];
        KERNEL_EXPECT_EQ(mod->is_zero(), false);
        KERNEL_EXPECT_EQ(mod->value(), 0xF00000F000079);
        KERNEL_EXPECT_EQ(mod->bit_count(), 52);
        KERNEL_EXPECT_EQ(mod->uint64_count(), 1);
        KERNEL_EXPECT_EQ(mod->const_ratio()[0], 1224979096621368355ULL);
        KERNEL_EXPECT_EQ(mod->const_ratio()[1], 4369ULL);
        KERNEL_EXPECT_EQ(mod->const_ratio()[2], 1144844808538997ULL);
        KERNEL_EXPECT_TRUE(mod->is_prime());

    }

    TEST(Modulus, DeviceCreateModulus) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(4, false);
        modulus[0] = Modulus(0);
        modulus[1] = Modulus(3);
        modulus[2] = Modulus(0xF00000F00000F);
        modulus[3] = Modulus(0xF00000F000079);
        Array<Modulus> device_modulus = modulus.to_device();

        Array<bool> r(16, true); 
        utils::set_device(r.device_index());
        kernel_create_modulus<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        cudaDeviceSynchronize();
        utils::MemoryPool::Destroy();
    }

    TEST(Modulus, HostReduce) {

        Modulus mod(2);
        EXPECT_EQ(0, mod.reduce(0));
        EXPECT_EQ(1, mod.reduce(1));
        EXPECT_EQ(0, mod.reduce(2));
        EXPECT_EQ(0, mod.reduce(0xF0F0F0));

        mod = Modulus(10);
        EXPECT_EQ(0, mod.reduce(0));
        EXPECT_EQ(1, mod.reduce(1));
        EXPECT_EQ(8, mod.reduce(8));
        EXPECT_EQ(7, mod.reduce(1234567));
        EXPECT_EQ(0, mod.reduce(12345670));

    }
    
    __global__ void kernel_reduce(ConstSlice<Modulus> modulus, Slice<bool> test_result) {

        size_t kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
        test_result[kernel_index] = true;

        const Modulus* mod = &modulus[0];
        KERNEL_EXPECT_EQ(0, mod->reduce(0));
        KERNEL_EXPECT_EQ(1, mod->reduce(1));
        KERNEL_EXPECT_EQ(0, mod->reduce(2));
        KERNEL_EXPECT_EQ(0, mod->reduce(0xF0F0F0));

        mod = &modulus[1];
        KERNEL_EXPECT_EQ(0, mod->reduce(0));
        KERNEL_EXPECT_EQ(1, mod->reduce(1));
        KERNEL_EXPECT_EQ(8, mod->reduce(8));
        KERNEL_EXPECT_EQ(7, mod->reduce(1234567));
        KERNEL_EXPECT_EQ(0, mod->reduce(12345670));

    }

    TEST(Modulus, DeviceReduce) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        Array<Modulus> modulus(2, false);
        modulus[0] = Modulus(2);
        modulus[1] = Modulus(10);
        Array<Modulus> device_modulus = modulus.to_device();

        Array<bool> r(16, true); 
        utils::set_device(r.device_index());
        kernel_reduce<<<4, 4>>>(device_modulus.const_reference(), r.reference());
        utils::stream_sync();
        Array<bool> h = r.to_host();
        EXPECT_TRUE(all_is_true(h));
        utils::MemoryPool::Destroy();
    }

}