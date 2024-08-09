#include "cuda_runtime.h"
#include <gtest/gtest.h>
#include "test.h"
#include "../src/ckks_encoder.h"

using namespace troy;
using troy::utils::Array;
using troy::utils::ConstSlice;
using troy::utils::Slice;
using std::vector;
using std::complex;

namespace ckks_encoder {

    void print_complex_vector(const vector<complex<double>> &a) {
        for (size_t i = 0; i < a.size(); i++) {
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }

    void print_double_vector(const vector<double> &a) {
        for (size_t i = 0; i < a.size(); i++) {
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }

    void print_uint64_vector(const vector<uint64_t> &a) {
        for (size_t i = 0; i < a.size(); i++) {
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }

    void test_simd(bool device) {

        size_t slots;
        EncryptionParameters parms(SchemeType::CKKS);
        HeContextPointer context;
        vector<complex<double>> values;
        
        slots = 1024;
        parms.set_poly_modulus_degree(slots << 1);
        parms.set_coeff_modulus(CoeffModulus::create(slots << 1, {40, 40, 40, 40}).const_reference());
        context = HeContext::create(parms, true, SecurityLevel::Nil);
        CKKSEncoder encoder = CKKSEncoder(context);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }
        double delta = std::pow(2.0, 16.0);

        values.resize(slots);
        for (size_t i = 0; i < slots; i++) {
            values[i] = complex<double>(0, 0);
        }
        Plaintext plain = encoder.encode_complex64_simd_new(values, std::nullopt, delta);
        ASSERT_EQ(plain.parms_id(), context->first_parms_id());
        vector<complex<double>> result = encoder.decode_complex64_simd_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        int bound = 16;
        for (size_t i = 0; i < slots; i++) {
            values[i] = complex<double>(rand() % bound, rand() % bound);
        }
        plain = encoder.encode_complex64_simd_new(values, std::nullopt, delta);
        result = encoder.decode_complex64_simd_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        for (size_t i = 0; i < slots; i++) {
            values[i] = complex<double>(rand() % bound, rand() % bound);
        }
        ParmsID second_parms_id = context->first_context_data().value()->next_context_data().value()->parms_id();
        plain = encoder.encode_complex64_simd_new(values, second_parms_id, delta);
        ASSERT_EQ(plain.parms_id(), second_parms_id);
        result = encoder.decode_complex64_simd_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        values.resize(10);
        for (size_t i = 0; i < 10; i++) {
            values[i] = complex<double>(rand() % bound, rand() % bound);
        }
        plain = encoder.encode_complex64_simd_new(values, std::nullopt, delta);
        result = encoder.decode_complex64_simd_new(plain);
        values.resize(slots);
        ASSERT_TRUE(near_vector(values, result));


        slots = 1024;
        parms.set_poly_modulus_degree(slots << 1);
        parms.set_coeff_modulus(CoeffModulus::create(slots << 1, {30, 30, 30, 30, 30}).const_reference());
        delta = std::pow(2.0, 30.0);
        context = HeContext::create(parms, false, SecurityLevel::Nil);
        encoder = CKKSEncoder(context);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }

        bound = 1 << 30;
        values.resize(slots);
        for (size_t i = 0; i < slots; i++) {
            values[i] = complex<double>(rand() % bound, rand() % bound);
        }
        plain = encoder.encode_complex64_simd_new(values, std::nullopt, delta);
        result = encoder.decode_complex64_simd_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        complex<double> value = complex<double>(rand() % bound, rand() % bound);
        for (size_t i = 0; i < slots; i++) {
            values[i] = value;
        }
        plain = encoder.encode_complex64_single_new(value, std::nullopt, delta);
        result = encoder.decode_complex64_simd_new(plain);
        ASSERT_TRUE(near_vector(values, result));

    }

    TEST(CKKSEncoderTest, HostSimd) {
        test_simd(false);
    }

    TEST(CKKSEncoderTest, DeviceSimd) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_simd(true);
        utils::MemoryPool::Destroy();
    }

    void test_double_polynomial(bool device) {

        size_t slots;
        EncryptionParameters parms(SchemeType::CKKS);
        HeContextPointer context;
        vector<double> values;
        
        slots = 1024;
        parms.set_poly_modulus_degree(slots << 1);
        parms.set_coeff_modulus(CoeffModulus::create(slots << 1, {40, 40, 40, 40}).const_reference());
        context = HeContext::create(parms, true, SecurityLevel::Nil);
        CKKSEncoder encoder = CKKSEncoder(context);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }
        double delta = std::pow(2.0, 16.0);

        values.resize(slots * 2);
        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = 0;
        }
        Plaintext plain = encoder.encode_float64_polynomial_new(values, std::nullopt, delta);
        ASSERT_EQ(plain.parms_id(), context->first_parms_id());
        vector<double> result = encoder.decode_float64_polynomial_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        int bound = 16;
        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = rand() % (2 * bound) - bound;
        }
        plain = encoder.encode_float64_polynomial_new(values, std::nullopt, delta);
        // std::cout << "plain: "; print_uint64_vector(plain.data().to_vector());
        result = encoder.decode_float64_polynomial_new(plain);
        // std::cout << "result: "; print_double_vector(result);
        ASSERT_TRUE(near_vector(values, result));

        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = rand() % (2 * bound) - bound;
        }
        ParmsID second_parms_id = context->first_context_data().value()->next_context_data().value()->parms_id();
        plain = encoder.encode_float64_polynomial_new(values, second_parms_id, delta);
        ASSERT_EQ(plain.parms_id(), second_parms_id);
        result = encoder.decode_float64_polynomial_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        values.resize(10);
        for (size_t i = 0; i < 10; i++) {
            values[i] = rand() % (2 * bound) - bound;
        }
        plain = encoder.encode_float64_polynomial_new(values, std::nullopt, delta);
        result = encoder.decode_float64_polynomial_new(plain);
        values.resize(slots * 2);
        ASSERT_TRUE(near_vector(values, result));
        
        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = 0;
        }
        values[0] = rand() % (2 * bound) - bound;
        plain = encoder.encode_float64_single_new(values[0], std::nullopt, delta);
        result = encoder.decode_float64_polynomial_new(plain);
        ASSERT_TRUE(near_vector(values, result));
    }

    TEST(CKKSEncoderTest, HostDoublePolynomial) {
        test_double_polynomial(false);
    }

    TEST(CKKSEncoderTest, DeviceDoublePolynomial) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_double_polynomial(true);
        utils::MemoryPool::Destroy();
    }

    void test_integer_polynomial(bool device) {

        size_t slots;
        EncryptionParameters parms(SchemeType::CKKS);
        HeContextPointer context;
        vector<int64_t> values;
        
        slots = 1024;
        parms.set_poly_modulus_degree(slots << 1);
        parms.set_coeff_modulus(CoeffModulus::create(slots << 1, {40, 40, 40, 40}).const_reference());
        context = HeContext::create(parms, true, SecurityLevel::Nil);
        CKKSEncoder encoder = CKKSEncoder(context);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }

        values.resize(slots * 2);
        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = 0;
        }
        Plaintext plain = encoder.encode_integer64_polynomial_new(values, std::nullopt);
        ASSERT_EQ(plain.parms_id(), context->first_parms_id());
        vector<double> result = encoder.decode_float64_polynomial_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        int bound = 16;
        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = rand() % (2 * bound) - bound;
        }
        plain = encoder.encode_integer64_polynomial_new(values, std::nullopt);
        result = encoder.decode_float64_polynomial_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = rand() % (2 * bound) - bound;
        }
        ParmsID second_parms_id = context->first_context_data().value()->next_context_data().value()->parms_id();
        plain = encoder.encode_integer64_polynomial_new(values, second_parms_id);
        ASSERT_EQ(plain.parms_id(), second_parms_id);
        result = encoder.decode_float64_polynomial_new(plain);
        ASSERT_TRUE(near_vector(values, result));

        values.resize(10);
        for (size_t i = 0; i < 10; i++) {
            values[i] = rand() % (2 * bound) - bound;
        }
        plain = encoder.encode_integer64_polynomial_new(values, std::nullopt);
        result = encoder.decode_float64_polynomial_new(plain);
        values.resize(slots * 2);
        ASSERT_TRUE(near_vector(values, result));
        
        for (size_t i = 0; i < slots * 2; i++) {
            values[i] = 0;
        }
        values[0] = rand() % (2 * bound) - bound;
        plain = encoder.encode_integer64_single_new(values[0], std::nullopt);
        result = encoder.decode_float64_polynomial_new(plain);
        ASSERT_TRUE(near_vector(values, result));
    }

    TEST(CKKSEncoderTest, HostIntegerPolynomial) {
        test_integer_polynomial(false);
    }

    TEST(CKKSEncoderTest, DeviceIntegerPolynomial) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_integer_polynomial(true);
        utils::MemoryPool::Destroy();
    }

}