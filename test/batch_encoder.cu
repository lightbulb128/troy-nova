#include <gtest/gtest.h>
#include "test.cuh"
#include "../src/batch_encoder.cuh"

using namespace troy;

namespace batch_encoder {

    static const size_t DEGREE = 1024;

    void test_unbatch_uint_vector(bool device) {

        EncryptionParameters parms(SchemeType::BFV);
        parms.set_poly_modulus_degree(DEGREE);
        parms.set_coeff_modulus(CoeffModulus::create(DEGREE, {60}).const_reference());
        Modulus t = PlainModulus::batching(DEGREE, 20);
        parms.set_plain_modulus(t);

        HeContextPointer context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_TRUE(context->first_context_data().value()->qualifiers().using_batching);

        BatchEncoder encoder(context);

        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }

        std::vector<uint64_t> plain_vec(encoder.slot_count());
        for (size_t i = 0; i < encoder.slot_count(); i++) {
            plain_vec[i] = i % t.value();
        }
        Plaintext plain = encoder.encode_new(plain_vec);
        auto decoded_vec = encoder.decode_new(plain);
        ASSERT_TRUE(same_vector(plain_vec, decoded_vec));

        for (size_t i = 0; i < encoder.slot_count(); i++) {
            plain_vec[i] = 5;
        }
        plain = encoder.encode_new(plain_vec);
        std::vector<uint64_t> encoded_vec(encoder.slot_count());
        encoded_vec[0] = 5;
        ASSERT_TRUE(same_vector(plain.poly().as_const(), encoded_vec));
        decoded_vec = encoder.decode_new(plain);
        ASSERT_TRUE(same_vector(plain_vec, decoded_vec));

        plain_vec.resize(20);
        for (size_t i = 0; i < 20; i++) {
            plain_vec[i] = i;
        }
        plain = encoder.encode_new(plain_vec);
        decoded_vec = encoder.decode_new(plain);
        ASSERT_TRUE(same_vector(plain_vec, utils::ConstSlice(decoded_vec.data(), 20, false)));
        ASSERT_TRUE(same_vector(utils::ConstSlice(decoded_vec.data() + 20, 44, false), std::vector<uint64_t>(44, 0)));
    }

    TEST(BatchEncoderTest, HostUnbatchUintVector) {
        test_unbatch_uint_vector(false);
    }

    TEST(BatchEncoderTest, DeviceUnbatchUintVector) {
        test_unbatch_uint_vector(true);
        utils::MemoryPool::Destroy();
    }

    void test_polynomial(bool device) {

        EncryptionParameters parms(SchemeType::BFV);
        parms.set_poly_modulus_degree(DEGREE);
        parms.set_coeff_modulus(CoeffModulus::create(DEGREE, {60}).const_reference());
        parms.set_plain_modulus(256);

        HeContextPointer context = HeContext::create(parms, false, SecurityLevel::Nil);
        ASSERT_FALSE(context->first_context_data().value()->qualifiers().using_batching);

        BatchEncoder encoder(context);

        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }

        std::vector<uint64_t> plain_vec(encoder.slot_count());
        for (size_t i = 0; i < encoder.slot_count(); i++) {
            plain_vec[i] = i % 256;
        }
        Plaintext plain = encoder.encode_polynomial_new(plain_vec);
        auto decoded_vec = encoder.decode_polynomial_new(plain);
        ASSERT_TRUE(same_vector(plain_vec, decoded_vec));

        plain_vec.resize(20);
        for (size_t i = 0; i < 20; i++) {
            plain_vec[i] = i;
        }
        plain = encoder.encode_polynomial_new(plain_vec);
        ASSERT_EQ(plain.coeff_count(), 20);
        decoded_vec = encoder.decode_polynomial_new(plain);
        ASSERT_TRUE(same_vector(plain_vec, decoded_vec));
    }

    TEST(BatchEncoderTest, HostPolynomial) {
        test_polynomial(false);
    }

    TEST(BatchEncoderTest, DevicePolynomial) {
        test_polynomial(true);
        utils::MemoryPool::Destroy();
    }

}