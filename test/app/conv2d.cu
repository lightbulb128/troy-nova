#include <gtest/gtest.h>
#include <sstream>
#include "../test_adv.h"
#include "../../src/app/conv2d.h"
#include "../test.h"


namespace conv2d {

    using namespace troy;
    using namespace troy::linear;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;
    using std::stringstream;
    using std::vector;

    inline uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t t) {
        __uint128_t c = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
        return static_cast<uint64_t>(c % static_cast<__uint128_t>(t));
    }

    inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t t) {
        if (a + b >= t) {
            return a + b - t;
        } else {
            return a + b;
        }
    }

    inline void add_mod_inplace(uint64_t& a, uint64_t b, uint64_t t) {
        a = add_mod(a, b, t);
    }

    void test_conv2d(
        const GeneralHeContext& context, 
        size_t bs, size_t ic, size_t oc, size_t ih, size_t iw, size_t kh, size_t kw,
        bool mod_switch_to_next
    ) {
        SchemeType scheme = context.params_host().scheme();
        if (scheme != SchemeType::BFV && scheme != SchemeType::BGV) {
            throw std::runtime_error("[test_matmul] Unsupported scheme");
        }
        uint64_t t = context.t();
        size_t oh = ih - kh + 1;
        size_t ow = iw - kw + 1;
        
        GeneralVector x = context.random_polynomial(bs * ic * ih * iw);
        GeneralVector w = context.random_polynomial(oc * ic * kh * kw);
        GeneralVector s = context.random_polynomial(bs * oc * oh * ow);
        Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw,
            context.params_host().poly_modulus_degree(), MatmulObjective::EncryptLeft);

        HeContextPointer he = context.context();
        const BatchEncoder& encoder = context.encoder().batch();
        const Encryptor& encryptor = context.encryptor();
        const Evaluator& evaluator = context.evaluator();
        const Decryptor& decryptor = context.decryptor();
        
        Cipher2d x_encrypted = helper.encrypt_inputs_uint64s(encryptor, encoder, x.integers().data());
        Plain2d w_encoded = helper.encode_weights_uint64s(encoder, w.integers().data());
        Plain2d s_encoded = helper.encode_outputs_uint64s(encoder, s.integers().data());

        stringstream x_serialized;
        x_encrypted.save(x_serialized, he);
        x_encrypted = Cipher2d::load_new(x_serialized, he);

        Cipher2d y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded);

        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<uint64_t> y_decrypted = helper.decrypt_outputs_uint64s(encoder, decryptor, y_encrypted);   

        vector<uint64_t> y_truth(bs * oc * oh * ow, 0);
        for (size_t b = 0; b < bs; b++) {
            for (size_t o = 0; o < oc; o++) {
                for (size_t i = 0; i < oh; i++) {
                    for (size_t j = 0; j < ow; j++) {
                        for (size_t c = 0; c < ic; c++) {
                            for (size_t p = 0; p < kh; p++) {
                                for (size_t q = 0; q < kw; q++) {
                                    add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                                        multiply_mod(
                                            x.integers()[b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q)],
                                            w.integers()[o * ic * kh * kw + c * kh * kw + p * kw + q],
                                            t
                                        ), t);
                                }
                            }
                        }
                        add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                            s.integers()[b * oc * oh * ow + o * oh * ow + i * ow + j], t);
                    }
                }
            }
        }

        GeneralVector decrypted(std::move(y_decrypted), false);
        GeneralVector truthv(std::move(y_truth), false);

        // std::cerr << "Truth:     " << truthv << std::endl;
        // std::cerr << "Decrypted: " << decrypted << std::endl;
        
        ASSERT_TRUE(truthv.near_equal(decrypted, 0));
    }

    void test_conv2d_reverse(
        const GeneralHeContext& context, 
        size_t bs, size_t ic, size_t oc, size_t ih, size_t iw, size_t kh, size_t kw,
        bool mod_switch_to_next
    ) {
        SchemeType scheme = context.params_host().scheme();
        if (scheme != SchemeType::BFV && scheme != SchemeType::BGV) {
            throw std::runtime_error("[test_matmul] Unsupported scheme");
        }
        uint64_t t = context.t();
        size_t oh = ih - kh + 1;
        size_t ow = iw - kw + 1;
        
        GeneralVector x = context.random_polynomial(bs * ic * ih * iw);
        GeneralVector w = context.random_polynomial(oc * ic * kh * kw);
        GeneralVector s = context.random_polynomial(bs * oc * oh * ow);
        Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw,
            context.params_host().poly_modulus_degree(), MatmulObjective::EncryptRight);

        HeContextPointer he = context.context();
        const BatchEncoder& encoder = context.encoder().batch();
        const Encryptor& encryptor = context.encryptor();
        const Evaluator& evaluator = context.evaluator();
        const Decryptor& decryptor = context.decryptor();
        
        Plain2d x_encoded = helper.encode_inputs_uint64s(encoder, x.integers().data());
        Cipher2d w_encrypted = helper.encrypt_weights_uint64s(encryptor, encoder, w.integers().data());
        Plain2d s_encoded = helper.encode_outputs_uint64s(encoder, s.integers().data());

        stringstream w_serialized;
        w_encrypted.save(w_serialized, he);
        w_encrypted = Cipher2d::load_new(w_serialized, he);

        Cipher2d y_encrypted = helper.conv2d_reverse(evaluator, x_encoded, w_encrypted);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<uint64_t> y_decrypted = helper.decrypt_outputs_uint64s(encoder, decryptor, y_encrypted);   

        vector<uint64_t> y_truth(bs * oc * oh * ow, 0);
        for (size_t b = 0; b < bs; b++) {
            for (size_t o = 0; o < oc; o++) {
                for (size_t i = 0; i < oh; i++) {
                    for (size_t j = 0; j < ow; j++) {
                        for (size_t c = 0; c < ic; c++) {
                            for (size_t p = 0; p < kh; p++) {
                                for (size_t q = 0; q < kw; q++) {
                                    add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                                        multiply_mod(
                                            x.integers()[b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q)],
                                            w.integers()[o * ic * kh * kw + c * kh * kw + p * kw + q],
                                            t
                                        ), t);
                                }
                            }
                        }
                        add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                            s.integers()[b * oc * oh * ow + o * oh * ow + i * ow + j], t);
                    }
                }
            }
        }

        GeneralVector decrypted(std::move(y_decrypted), false);
        GeneralVector truthv(std::move(y_truth), false);

        // std::cerr << "Truth:     " << truthv << std::endl;
        // std::cerr << "Decrypted: " << decrypted << std::endl;
        
        ASSERT_TRUE(truthv.near_equal(decrypted, 0));
    }



    TEST(Conv2dTest, HostBFVConv2d) {
        GeneralHeContext ghe(false, SchemeType::BFV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d(ghe, 2, 2, 6, 7, 9, 3, 5, false);
        // test_conv2d(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, DeviceBFVConv2d) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }
    
    TEST(Conv2dTest, HostBGVConv2d) {
        GeneralHeContext ghe(false, SchemeType::BGV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, DeviceBGVConv2d) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, HostBFVConv2dReverse) {
        GeneralHeContext ghe(false, SchemeType::BFV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d_reverse(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, DeviceBFVConv2dReverse) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d_reverse(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }
    
    TEST(Conv2dTest, HostBGVConv2dReverse) {
        GeneralHeContext ghe(false, SchemeType::BGV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d_reverse(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, DeviceBGVConv2dReverse) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 1024, 40, { 60, 40, 40, 60 }, true, 0x123, 0);
        srand(0);
        test_conv2d_reverse(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }



}