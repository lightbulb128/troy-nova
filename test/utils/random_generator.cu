#include <gtest/gtest.h>
#include "../test.h"
#include "../../src/utils/random_generator.h"

using namespace troy;
using namespace troy::utils;

namespace random_generator {

    bool test_body_seeded_rng(bool device) {

        // same seed should get same results
        size_t produce_count = 100;
        
        RandomGenerator rng1(36);
        RandomGenerator rng2(36);
        
        Array<uint64_t> buffer1(produce_count, device);
        Array<uint64_t> buffer2(produce_count, device);

        rng1.fill_uint64s(buffer1.reference());
        rng2.fill_uint64s(buffer2.reference());

        if (device) {
            buffer1.to_host_inplace();
            buffer2.to_host_inplace();
        }
        
        // check results
        
        for (size_t i = 0; i < produce_count; i++) {
            if (buffer1[i] != buffer2[i]) {
                return false;
            }
        }

        // different seeds should get different results

        RandomGenerator rng3(13);
        RandomGenerator rng4(49);

        // reuse buffers
        if (device) {
            buffer1.to_device_inplace();
            buffer2.to_device_inplace();
        }

        rng3.fill_uint64s(buffer1.reference());
        rng4.fill_uint64s(buffer2.reference());

        if (device) {
            buffer1.to_host_inplace();
            buffer2.to_host_inplace();
        }

        // check results

        for (size_t i = 0; i < produce_count; i++) {
            if (buffer1[i] == buffer2[i]) {
                return false;
            }
        }

        return true;
    }

    TEST(RandomGeneratorTest, HostSeededRNG) {
        ASSERT_TRUE(test_body_seeded_rng(false));
        MemoryPool::Destroy();
    }

    TEST(RandomGeneratorTest, DeviceSeededRNG) {
        ASSERT_TRUE(test_body_seeded_rng(true));
        MemoryPool::Destroy();
    }

    bool test_body_ternary_or_centered_binomial(bool device, bool ternary) {

        size_t moduli_count = 3;
        Array<Modulus> moduli(moduli_count, false);
        moduli[0] = Modulus(12345);
        moduli[1] = Modulus(23456);
        moduli[2] = Modulus(56789);

        size_t n = 16;
        
        if (device) {
            moduli.to_device_inplace();
        }

        RandomGenerator rng(13);

        Array<uint64_t> buffer(n * moduli_count, device);

        if (ternary) {
            rng.sample_poly_ternary(buffer.reference(), n, moduli.const_reference());
        } else {
            rng.sample_poly_centered_binomial(buffer.reference(), n, moduli.const_reference());
        }

        if (device) {
            buffer.to_host_inplace();
            moduli.to_host_inplace();
        }

        // check consistency
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 1; i < moduli_count; i++) {
                int absolute = buffer[j];
                if (absolute > static_cast<int>(moduli[0].value() / 2)) absolute -= moduli[0].value();
                int absolute_j = buffer[j + i * n];
                if (absolute_j > static_cast<int>(moduli[0].value() / 2)) absolute_j -= moduli[i].value();
                if (absolute != absolute_j) {
                    return false;
                }
            }
        }

        return true;

    }

    TEST(RandomGeneratorTest, HostTernary) {
        ASSERT_TRUE(test_body_ternary_or_centered_binomial(false, true));
    }

    TEST(RandomGeneratorTest, DeviceTernary) {
        ASSERT_TRUE(test_body_ternary_or_centered_binomial(true, true));
        MemoryPool::Destroy();
    }

    TEST(RandomGeneratorTest, HostCenteredBinomial) {
        ASSERT_TRUE(test_body_ternary_or_centered_binomial(false, false));
    }

    TEST(RandomGeneratorTest, DeviceCenteredBinomial) {
        ASSERT_TRUE(test_body_ternary_or_centered_binomial(true, false));
        MemoryPool::Destroy();
    }

    
    bool test_body_uniform(bool device) {

        size_t moduli_count = 3;
        Array<Modulus> moduli(moduli_count, false);
        moduli[0] = Modulus(12345);
        moduli[1] = Modulus(23456);
        moduli[2] = Modulus(56789);

        size_t n = 16;
        
        if (device) {
            moduli.to_device_inplace();
        }

        RandomGenerator rng(13);

        Array<uint64_t> buffer(n * moduli_count, device);

        rng.sample_poly_uniform(buffer.reference(), n, moduli.const_reference());

        if (device) {
            buffer.to_host_inplace();
            moduli.to_host_inplace();
        }

        // check consistency
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < moduli_count; i++) {
                if (buffer[j + i * n] >= moduli[i].value()) {
                    return false;
                }
            }
        }

        return true;

    }

    TEST(RandomGeneratorTest, HostUniform) {
        ASSERT_TRUE(test_body_uniform(false));
    }

    TEST(RandomGeneratorTest, DeviceUniform) {
        ASSERT_TRUE(test_body_uniform(true));
        MemoryPool::Destroy();
    }

    TEST(RandomGeneratorTest, HostDeviceConsistency) {
        
        // same seed should get same results
        size_t produce_count = 123;
        
        RandomGenerator rng1(36);
        RandomGenerator rng2(36);
        
        Array<uint8_t> buffer1(produce_count, false);
        Array<uint8_t> buffer2(produce_count, true);

        rng1.fill_bytes(buffer1.reference());
        rng2.fill_bytes(buffer2.reference());

        buffer1.to_host_inplace();
        buffer2.to_host_inplace();

        // check results
        for (size_t i = 0; i < produce_count; i++) {
            ASSERT_EQ(buffer1[i], buffer2[i]);
        }

        size_t moduli_count = 3;
        Array<Modulus> moduli(moduli_count, false);
        moduli[0] = Modulus(12345);
        moduli[1] = Modulus(23456);
        moduli[2] = Modulus(56789);
        Array<Modulus> modulus_device = moduli.to_device();

        size_t n = 123;

        { // ternary
            Array<uint64_t> buffer_host(n * moduli_count, false);
            Array<uint64_t> buffer_device(n * moduli_count, true);

            rng1.sample_poly_ternary(buffer_host.reference(), n, moduli.const_reference());
            rng2.sample_poly_ternary(buffer_device.reference(), n, modulus_device.const_reference());

            buffer_device.to_host_inplace();
            for (size_t i = 0; i < n * moduli_count; i++) {
                ASSERT_EQ(buffer_host[i], buffer_device[i]);
            }
        }

        { // centered binomial
            Array<uint64_t> buffer_host(n * moduli_count, false);
            Array<uint64_t> buffer_device(n * moduli_count, true);

            rng1.sample_poly_centered_binomial(buffer_host.reference(), n, moduli.const_reference());
            rng2.sample_poly_centered_binomial(buffer_device.reference(), n, modulus_device.const_reference());

            buffer_device.to_host_inplace();
            for (size_t i = 0; i < n * moduli_count; i++) {
                ASSERT_EQ(buffer_host[i], buffer_device[i]);
            }
        }

        { // uniform
            Array<uint64_t> buffer_host(n * moduli_count, false);
            Array<uint64_t> buffer_device(n * moduli_count, true);

            rng1.sample_poly_uniform(buffer_host.reference(), n, moduli.const_reference());
            rng2.sample_poly_uniform(buffer_device.reference(), n, modulus_device.const_reference());

            buffer_device.to_host_inplace();
            for (size_t i = 0; i < n * moduli_count; i++) {
                ASSERT_EQ(buffer_host[i], buffer_device[i]);
            }

        }

    }


}