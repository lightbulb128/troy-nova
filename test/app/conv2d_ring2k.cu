#include <gtest/gtest.h>
#include <sstream>
#include "../test_adv.h"
#include "../../src/app/conv2d.h"
#include "../test.h"


namespace conv2d_ring2k {

    using namespace troy;
    using namespace troy::linear;
    using std::stringstream;
    using std::vector;
    using uint128_t = __uint128_t;

    template <typename T>
    vector<T> random_sampler(size_t size, size_t bit_length) {
        vector<T> result(size);
        T mask = bit_length == (sizeof(T) * 8) ? (static_cast<T>(-1)) : (static_cast<T>(1) << bit_length) - 1;
        for (size_t i = 0; i < size; i++) {
            result[i] = rand() & mask;
        }
        return result;
    }

    template <>
    vector<uint128_t> random_sampler<uint128_t>(size_t size, size_t bit_length) {
        vector<uint128_t> result(size);
        uint128_t mask = bit_length == (sizeof(uint128_t) * 8) ? (static_cast<uint128_t>(-1)) : (static_cast<uint128_t>(1) << bit_length) - 1;
        for (size_t i = 0; i < size; i++) {
            result[i] = ((static_cast<uint128_t>(rand()) << 64) | rand()) & mask;
        }
        return result;
    }

    template <typename T>
    bool same_vector(const vector<T>& a, const vector<T>& b) {
        if (b.size() != a.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    template <typename T>
    void test_conv2d(
        bool device,
        size_t t_bits, size_t poly_degree,
        std::vector<size_t> q_bits,
        size_t bs, size_t ic, size_t oc, size_t ih, size_t iw, size_t kh, size_t kw,
        bool mod_switch_to_next
    ) {
        EncryptionParameters params(SchemeType::BFV);
        params.set_poly_modulus_degree(poly_degree);
        params.set_plain_modulus(1 << 20); // this does not matter
        params.set_coeff_modulus(CoeffModulus::create(poly_degree, q_bits));
        uint128_t t_mask = (t_bits == 128) ? static_cast<uint128_t>(-1) : ((static_cast<uint128_t>(1) << t_bits) - 1);

        HeContextPointer context = HeContext::create(params, true, SecurityLevel::Nil, 0x123);
        PolynomialEncoderRing2k<T> encoder(context, t_bits);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }

        size_t oh = ih - kh + 1;
        size_t ow = iw - kw + 1;
        
        vector<T> x = random_sampler<T>(bs * ic * ih * iw, t_bits);
        vector<T> w = random_sampler<T>(oc * ic * kh * kw, t_bits);
        vector<T> s = random_sampler<T>(bs * oc * oh * ow, t_bits);
        Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw,
            poly_degree, MatmulObjective::EncryptLeft);

        KeyGenerator keygen(context);
        PublicKey public_key = keygen.create_public_key(false);
        Encryptor encryptor(context); encryptor.set_public_key(public_key);
        encryptor.set_secret_key(keygen.secret_key());
        Evaluator evaluator(context);
        Decryptor decryptor(context, keygen.secret_key());
        
        Cipher2d x_encrypted = helper.encrypt_inputs_ring2k(encryptor, encoder, x.data(), std::nullopt);
        Plain2d w_encoded = helper.encode_weights_ring2k(encoder, w.data(), std::nullopt);
        Plain2d s_encoded = helper.encode_outputs_ring2k(encoder, s.data(), std::nullopt);


        stringstream x_serialized; 
        x_encrypted.save(x_serialized, context);
        x_encrypted = Cipher2d::load_new(x_serialized, context);

        Cipher2d y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<T> y_decrypted = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);   

        vector<T> y_truth(bs * oc * oh * ow, 0);
        for (size_t b = 0; b < bs; b++) {
            for (size_t o = 0; o < oc; o++) {
                for (size_t i = 0; i < oh; i++) {
                    for (size_t j = 0; j < ow; j++) {
                        T& y_current = y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j];
                        for (size_t c = 0; c < ic; c++) {
                            for (size_t p = 0; p < kh; p++) {
                                for (size_t q = 0; q < kw; q++) {
                                    y_current +=
                                        x[b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q)] *
                                        w[o * ic * kh * kw + c * kh * kw + p * kw + q];
                                }
                            }
                        }
                        y_current += s[b * oc * oh * ow + o * oh * ow + i * ow + j];
                        y_current &= t_mask;
                    }
                }
            }
        }

        ASSERT_TRUE(same_vector(y_truth, y_decrypted));
    }


    template <typename T>
    void test_conv2d_reverse(
        bool device,
        size_t t_bits, size_t poly_degree,
        std::vector<size_t> q_bits,
        size_t bs, size_t ic, size_t oc, size_t ih, size_t iw, size_t kh, size_t kw,
        bool mod_switch_to_next
    ) {
        EncryptionParameters params(SchemeType::BFV);
        params.set_poly_modulus_degree(poly_degree);
        params.set_plain_modulus(1 << 20); // this does not matter
        params.set_coeff_modulus(CoeffModulus::create(poly_degree, q_bits));
        uint128_t t_mask = (t_bits == 128) ? static_cast<uint128_t>(-1) : ((static_cast<uint128_t>(1) << t_bits) - 1);

        HeContextPointer context = HeContext::create(params, true, SecurityLevel::Nil, 0x123);
        PolynomialEncoderRing2k<T> encoder(context, t_bits);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }

        size_t oh = ih - kh + 1;
        size_t ow = iw - kw + 1;
        
        vector<T> x = random_sampler<T>(bs * ic * ih * iw, t_bits);
        vector<T> w = random_sampler<T>(oc * ic * kh * kw, t_bits);
        vector<T> s = random_sampler<T>(bs * oc * oh * ow, t_bits);
        Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw,
            poly_degree, MatmulObjective::EncryptRight);

        KeyGenerator keygen(context);
        PublicKey public_key = keygen.create_public_key(false);
        Encryptor encryptor(context); encryptor.set_public_key(public_key);
        encryptor.set_secret_key(keygen.secret_key());
        Evaluator evaluator(context);
        Decryptor decryptor(context, keygen.secret_key());
        
        Plain2d x_encoded = helper.encode_inputs_ring2k(encoder, x.data(), std::nullopt);
        Cipher2d w_encrypted = helper.encrypt_weights_ring2k(encryptor, encoder, w.data(), std::nullopt);
        Plain2d s_encoded = helper.encode_outputs_ring2k(encoder, s.data(), std::nullopt);


        stringstream w_serialized;
        w_encrypted.save(w_serialized, context);
        w_encrypted = Cipher2d::load_new(w_serialized, context);

        Cipher2d y_encrypted = helper.conv2d_reverse(evaluator, x_encoded, w_encrypted);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<T> y_decrypted = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);   

        vector<T> y_truth(bs * oc * oh * ow, 0);
        for (size_t b = 0; b < bs; b++) {
            for (size_t o = 0; o < oc; o++) {
                for (size_t i = 0; i < oh; i++) {
                    for (size_t j = 0; j < ow; j++) {
                        T& y_current = y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j];
                        for (size_t c = 0; c < ic; c++) {
                            for (size_t p = 0; p < kh; p++) {
                                for (size_t q = 0; q < kw; q++) {
                                    y_current +=
                                        x[b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q)] *
                                        w[o * ic * kh * kw + c * kh * kw + p * kw + q];
                                }
                            }
                        }
                        y_current += s[b * oc * oh * ow + o * oh * ow + i * ow + j];
                        y_current &= t_mask;
                    }
                }
            }
        }

        ASSERT_TRUE(same_vector(y_truth, y_decrypted));
    }

    void ring32_test_suite(bool device) {
        test_conv2d<uint32_t>(device, 17, 4096, {60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint32_t>(device, 20, 4096, {60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint32_t>(device, 32, 4096, {60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint32_t>(device, 32, 4096, {60, 60, 60}, 2, 3, 10, 56, 56, 10, 10, false);
    }

    void ring64_test_suite(bool device) {
        test_conv2d<uint64_t>(device, 33, 4096, {60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint64_t>(device, 50, 4096, {60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 2, 3, 10, 56, 56, 10, 10, false);
    }

    void ring128_test_suite(bool device) {
        test_conv2d<uint128_t>(device,  65, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint128_t>(device, 100, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 10, 56, 56, 10, 10, false);
    }

    
    void ring32_test_reverse_suite(bool device) {
        test_conv2d_reverse<uint32_t>(device, 17, 4096, {60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint32_t>(device, 20, 4096, {60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint32_t>(device, 32, 4096, {60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint32_t>(device, 32, 4096, {60, 60, 60}, 2, 3, 10, 56, 56, 10, 10, false);
    }

    void ring64_test_reverse_suite(bool device) {
        test_conv2d_reverse<uint64_t>(device, 33, 4096, {60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint64_t>(device, 50, 4096, {60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 2, 3, 10, 56, 56, 10, 10, false);
    }

    void ring128_test_reverse_suite(bool device) {
        test_conv2d_reverse<uint128_t>(device,  65, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint128_t>(device, 100, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, HostRing32Conv2d) { 
        ring32_test_suite(false); 
    }
    TEST(Conv2dTest, DeviceRing32Conv2d) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring32_test_suite(true); 
    }
    TEST(Conv2dTest, HostRing64Conv2d) { 
        ring64_test_suite(false); 
    }
    TEST(Conv2dTest, DeviceRing64Conv2d) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring64_test_suite(true); 
    }
    TEST(Conv2dTest, HostRing128Conv2d) { 
        ring128_test_suite(false); 
    }
    TEST(Conv2dTest, DeviceRing128Conv2d) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring128_test_suite(true); 
    }
    TEST(Conv2dTest, HostRing32Conv2dReverse) { 
        ring32_test_reverse_suite(false); 
    }
    TEST(Conv2dTest, DeviceRing32Conv2dReverse) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring32_test_reverse_suite(true); 
    }
    TEST(Conv2dTest, HostRing64Conv2dReverse) { 
        ring64_test_reverse_suite(false); 
    }
    TEST(Conv2dTest, DeviceRing64Conv2dReverse) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring64_test_reverse_suite(true); 
    }
    TEST(Conv2dTest, HostRing128Conv2dReverse) { 
        ring128_test_reverse_suite(false); 
    }
    TEST(Conv2dTest, DeviceRing128Conv2dReverse) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring128_test_reverse_suite(true); 
    }
}