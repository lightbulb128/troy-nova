#include <gtest/gtest.h>
#include "test.h"
#include "../src/batch_encoder.h"

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
        ASSERT_TRUE(same_vector(plain_vec, utils::ConstSlice(decoded_vec.data(), 20, false, nullptr)));
        ASSERT_TRUE(same_vector(utils::ConstSlice(decoded_vec.data() + 20, 44, false, nullptr), std::vector<uint64_t>(44, 0)));
    }

    TEST(BatchEncoderTest, HostUnbatchUintVector) {
        test_unbatch_uint_vector(false);
    }

    TEST(BatchEncoderTest, DeviceUnbatchUintVector) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_unbatch_uint_vector(true);
        utils::MemoryPool::Destroy();
    }

    void test_polynomial(bool device, bool is_bgv) {

        EncryptionParameters parms(!is_bgv ? SchemeType::BFV : SchemeType::BGV);
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

    TEST(BatchEncoderTest, HostBFVPolynomial) {
        test_polynomial(false, false);
    }

    TEST(BatchEncoderTest, DeviceBFVPolynomial) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_polynomial(true, false);
        utils::MemoryPool::Destroy();
    }

    TEST(BatchEncoderTest, HostBGVPolynomial) {
        test_polynomial(false, true);
    }

    TEST(BatchEncoderTest, DeviceBGVPolynomial) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_polynomial(true, true);
        utils::MemoryPool::Destroy();
    }

    void test_scale_up_down(bool device) {

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
        Plaintext scale_up = encoder.scale_up_new(plain, std::nullopt);
        Plaintext scale_down = encoder.scale_down_new(scale_up);

        auto original_vec = plain.data().to_vector();
        auto scale_down_vec = scale_down.data().to_vector();
        ASSERT_TRUE(same_vector(original_vec, scale_down_vec));

        // partial
        size_t n = encoder.slot_count();
        size_t cc = n / 3;
        {
            std::vector<uint64_t> plain_vec(cc);
            for (size_t i = 0; i < cc; i++) {
                plain_vec[i] = i % t.value();
            }
            Plaintext plain = encoder.encode_polynomial_new(plain_vec);
            ASSERT_EQ(plain.coeff_count(), cc);
            Plaintext scale_up = encoder.scale_up_new(plain, std::nullopt);
            ASSERT_EQ(scale_up.coeff_count(), cc);
            ASSERT_EQ(scale_up.poly_modulus_degree(), n);
            ASSERT_EQ(scale_up.data().size(), scale_up.coeff_modulus_size() * cc);
            Plaintext scale_down = encoder.scale_down_new(scale_up);
            ASSERT_EQ(scale_down.coeff_count(), cc);
            ASSERT_EQ(scale_down.poly_modulus_degree(), n);
            auto scale_down_vec = scale_down.data().to_vector();
            auto original_vec = plain.data().to_vector();
            ASSERT_TRUE(same_vector(original_vec, scale_down_vec));
        }
    }

    TEST(BatchEncoderTest, HostScaleUpDown) {
        test_scale_up_down(false);
    }

    TEST(BatchEncoderTest, DeviceScaleUpDown) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_scale_up_down(true);
        utils::MemoryPool::Destroy();
    }


    void test_centralize_decentralize(bool device) {

        EncryptionParameters parms(SchemeType::BGV);
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
        Plaintext scale_up = encoder.centralize_new(plain, std::nullopt);
        Plaintext scale_down = encoder.decentralize_new(scale_up);

        auto original_vec = plain.data().to_vector();
        auto scale_down_vec = scale_down.data().to_vector();
        ASSERT_TRUE(same_vector(original_vec, scale_down_vec));

        // partial
        size_t n = encoder.slot_count();
        size_t cc = n / 3;
        {
            std::vector<uint64_t> plain_vec(cc);
            for (size_t i = 0; i < cc; i++) {
                plain_vec[i] = i % t.value();
            }
            Plaintext plain = encoder.encode_polynomial_new(plain_vec);
            ASSERT_EQ(plain.coeff_count(), cc);
            Plaintext scale_up = encoder.centralize_new(plain, std::nullopt);
            ASSERT_EQ(scale_up.coeff_count(), cc);
            ASSERT_EQ(scale_up.poly_modulus_degree(), n);
            ASSERT_EQ(scale_up.data().size(), scale_up.coeff_modulus_size() * cc);
            Plaintext scale_down = encoder.decentralize_new(scale_up);
            ASSERT_EQ(scale_down.coeff_count(), cc);
            ASSERT_EQ(scale_down.poly_modulus_degree(), n);
            auto scale_down_vec = scale_down.data().to_vector();
            auto original_vec = plain.data().to_vector();
            ASSERT_TRUE(same_vector(original_vec, scale_down_vec));
        }
    }

    TEST(BatchEncoderTest, HostCentralizeDecentralize) {
        test_centralize_decentralize(false);
    }

    TEST(BatchEncoderTest, DeviceCentralizeDecentralize) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_centralize_decentralize(true);
        utils::MemoryPool::Destroy();
    }



}