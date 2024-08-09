#include <gtest/gtest.h>
#include "../../src/utils/ntt.h"
#include "../test.h"

using namespace troy;
using namespace troy::utils;

namespace ntt {

    TEST(NTT, Basics) {

        size_t coeff_count_power = 1;
        Modulus modulus = get_prime(2 << coeff_count_power, 60);
        NTTTables tables(coeff_count_power, modulus);
        ASSERT_EQ(tables.coeff_count(), 2);
        ASSERT_EQ(tables.coeff_count_power(), 1);

        coeff_count_power = 2;
        modulus = get_prime(2 << coeff_count_power, 50);
        tables = NTTTables(coeff_count_power, modulus);
        ASSERT_EQ(tables.coeff_count(), 4);
        ASSERT_EQ(tables.coeff_count_power(), 2);

        coeff_count_power = 10;
        modulus = get_prime(2 << coeff_count_power, 40);
        tables = NTTTables(coeff_count_power, modulus);
        ASSERT_EQ(tables.coeff_count(), 1024);
        ASSERT_EQ(tables.coeff_count_power(), 10);

        MemoryPool::Destroy();
    }

    TEST(NTT, PrimitiveRoots) {
        size_t coeff_count_power = 1;
        Modulus modulus(0xffffffffffc0001);
        NTTTables tables(coeff_count_power, modulus);
        ASSERT_EQ(tables.root_powers()[0].operand, 1);
        ASSERT_EQ(tables.root_powers()[1].operand, 288794978602139552ul);

        uint64_t inv;
        utils::try_invert_uint64_mod(288794978602139552ul, modulus, inv);
        ASSERT_EQ(tables.inv_root_powers()[1].operand, inv);

        coeff_count_power = 2;
        tables = NTTTables(coeff_count_power, modulus);
        ASSERT_EQ(tables.root_powers()[0].operand, 1);
        ASSERT_EQ(tables.root_powers()[1].operand, 288794978602139552ul);
        ASSERT_EQ(tables.root_powers()[2].operand, 178930308976060547ul);
        ASSERT_EQ(tables.root_powers()[3].operand, 748001537669050592ul);
        MemoryPool::Destroy();
    }

    TEST(NTT, HostNegacyclicNTT) {

        size_t coeff_count_power = 1;
        Modulus modulus(0xffffffffffc0001);
        NTTTables tables(coeff_count_power, modulus);
        ConstSlice<NTTTables> table_slice(&tables, 1, false, nullptr);

        Array<uint64_t> poly(2, false, nullptr);
        poly[0] = 0; poly[1] = 0;
        ntt_inplace_p(poly.reference(), poly.size(), table_slice);
        EXPECT_EQ(poly[0], 0);
        EXPECT_EQ(poly[1], 0);

        poly[0] = 1; poly[1] = 0;
        ntt_inplace_p(poly.reference(), poly.size(), table_slice);
        EXPECT_EQ(poly[0], 1);
        EXPECT_EQ(poly[1], 1);

        poly[0] = 1; poly[1] = 1;
        ntt_inplace_p(poly.reference(), poly.size(), table_slice);
        EXPECT_EQ(poly[0], 288794978602139553);
        EXPECT_EQ(poly[1], 864126526004445282);
    }


    TEST(NTT, DeviceNegacyclicNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;

        size_t coeff_count_power = 1;
        Modulus modulus(0xffffffffffc0001);
        Box<NTTTables> tables(new NTTTables(coeff_count_power, modulus), false, nullptr);
        tables->to_device_inplace(); // this moves the arrays into device, but not the table itself
        Box<NTTTables> tables_device = tables.to_device(MemoryPool::GlobalPool());
        ConstSlice<NTTTables> table_slice = ConstSlice<NTTTables>::from_pointer(tables_device.as_const_pointer());

        Array<uint64_t> poly(2, false, nullptr);
        poly[0] = 0; poly[1] = 0;
        poly.to_device_inplace(MemoryPool::GlobalPool());
        ntt_inplace_p(poly.reference(), poly.size(), table_slice);
        poly.to_host_inplace();
        EXPECT_EQ(poly[0], 0);
        EXPECT_EQ(poly[1], 0);

        // poly[0] = 1; poly[1] = 0;
        // poly.to_device_inplace();
        // ntt_inplace_p(poly.reference(), poly.size(), table_slice);
        // poly.to_host_inplace();
        // EXPECT_EQ(poly[0], 1);
        // EXPECT_EQ(poly[1], 1);

        // poly[0] = 1; poly[1] = 1;
        // poly.to_device_inplace();
        // ntt_inplace_p(poly.reference(), poly.size(), table_slice);
        // poly.to_host_inplace();
        // EXPECT_EQ(poly[0], 288794978602139553);
        // EXPECT_EQ(poly[1], 864126526004445282);
        MemoryPool::Destroy();
    }

    TEST(NTT, HostInverseNegacyclicNTT) {

        size_t coeff_count_power = 5;
        size_t n = 1 << coeff_count_power;
        Modulus modulus(0xffffffffffc0001);
        NTTTables tables(coeff_count_power, modulus);
        ConstSlice<NTTTables> table_slice(&tables, 1, false, nullptr);

        Array<uint64_t> poly(n, false, nullptr);
        set_zero_uint(poly.reference());
        intt_inplace_p(poly.reference(), poly.size(), table_slice);
        for (size_t i = 0; i < n; i++) {
            EXPECT_EQ(poly[i], 0);
        }

        Array<uint64_t> original(n, false, nullptr);
        for (size_t i = 0; i < n; i++) {
            original[i] = i;
        }
        poly = original.clone(MemoryPool::GlobalPool());
        ntt_inplace_p(poly.reference(), n, table_slice);
        intt_inplace_p(poly.reference(), n, table_slice);
        for (size_t i = 0; i < n; i++) {
            EXPECT_EQ(poly[i], original[i]);
        }

    }

    TEST(NTT, DeviceInverseNegacyclicNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;

        size_t coeff_count_power = 5;
        size_t n = 1 << coeff_count_power;
        Modulus modulus(0xffffffffffc0001);
        
        Box<NTTTables> tables(new NTTTables(coeff_count_power, modulus), false, nullptr);
        tables->to_device_inplace(); // this moves the arrays into device, but not the table itself
        Box<NTTTables> tables_device = tables.to_device(MemoryPool::GlobalPool());
        ConstSlice<NTTTables> table_slice = ConstSlice<NTTTables>::from_pointer(tables_device.as_const_pointer());

        Array<uint64_t> poly(n, false, nullptr);
        set_zero_uint(poly.reference());
        poly.to_device_inplace(MemoryPool::GlobalPool());
        intt_inplace_p(poly.reference(), poly.size(), table_slice);
        poly.to_host_inplace();
        for (size_t i = 0; i < n; i++) {
            EXPECT_EQ(poly[i], 0);
        }

        Array<uint64_t> original(n, false, nullptr);
        for (size_t i = 0; i < n; i++) {
            original[i] = i;
        }
        poly = original.clone(MemoryPool::GlobalPool());
        poly.to_device_inplace(MemoryPool::GlobalPool());
        ntt_inplace_p(poly.reference(), n, table_slice);
        intt_inplace_p(poly.reference(), n, table_slice);
        poly.to_host_inplace();
        for (size_t i = 0; i < n; i++) {
            EXPECT_EQ(poly[i], original[i]);
        }

        MemoryPool::Destroy();
    }

}