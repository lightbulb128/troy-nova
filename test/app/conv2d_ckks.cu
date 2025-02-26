#include <gtest/gtest.h>
#include <sstream>
#include "../test_adv.h"
#include "../../src/app/conv2d.h"
#include "../test.h"


namespace conv2d_ckks {

    using namespace troy;
    using namespace troy::linear;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;
    using std::stringstream;
    using std::vector;

    void test_conv2d(
        const GeneralHeContext& context, 
        size_t bs, size_t ic, size_t oc, size_t ih, size_t iw, size_t kh, size_t kw,
        bool mod_switch_to_next
    ) {
        SchemeType scheme = context.params_host().scheme();
        if (scheme != SchemeType::CKKS) {
            throw std::runtime_error("[test_matmul] Unsupported scheme");
        }
        double scale = context.scale();
        size_t oh = ih - kh + 1;
        size_t ow = iw - kw + 1;
        
        GeneralVector x = context.random_polynomial(bs * ic * ih * iw);
        GeneralVector w = context.random_polynomial(oc * ic * kh * kw);
        GeneralVector s = context.random_polynomial(bs * oc * oh * ow);
        Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw,
            context.params_host().poly_modulus_degree(), MatmulObjective::EncryptLeft);

        HeContextPointer he = context.context();
        const CKKSEncoder& encoder = context.encoder().ckks();
        const Encryptor& encryptor = context.encryptor();
        const Evaluator& evaluator = context.evaluator();
        const Decryptor& decryptor = context.decryptor();
        
        Cipher2d x_encrypted = helper.encrypt_inputs_doubles(encryptor, encoder, x.doubles().data(), std::nullopt, scale);
        Plain2d w_encoded = helper.encode_weights_doubles(encoder, w.doubles().data(), std::nullopt, scale);
        Plain2d s_encoded = helper.encode_outputs_doubles(encoder, s.doubles().data(), std::nullopt, scale * scale);

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

        vector<double> y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);   

        vector<double> y_truth(bs * oc * oh * ow, 0);
        for (size_t b = 0; b < bs; b++) {
            for (size_t o = 0; o < oc; o++) {
                for (size_t i = 0; i < oh; i++) {
                    for (size_t j = 0; j < ow; j++) {
                        for (size_t c = 0; c < ic; c++) {
                            for (size_t p = 0; p < kh; p++) {
                                for (size_t q = 0; q < kw; q++) {
                                    y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j] +=
                                        x.doubles()[b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q)] *
                                        w.doubles()[o * ic * kh * kw + c * kh * kw + p * kw + q];
                                }
                            }
                        }
                        y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j] += s.doubles()[b * oc * oh * ow + o * oh * ow + i * ow + j];
                    }
                }
            }
        }

        GeneralVector decrypted(std::move(y_decrypted));
        GeneralVector truthv(std::move(y_truth));

        // std::cerr << "Truth:     " << truthv << std::endl;
        // std::cerr << "Decrypted: " << decrypted << std::endl;
        
        ASSERT_TRUE(truthv.near_equal(decrypted, context.tolerance()));
    }


    void test_conv2d_reverse(
        const GeneralHeContext& context, 
        size_t bs, size_t ic, size_t oc, size_t ih, size_t iw, size_t kh, size_t kw,
        bool mod_switch_to_next
    ) {
        SchemeType scheme = context.params_host().scheme();
        if (scheme != SchemeType::CKKS) {
            throw std::runtime_error("[test_matmul] Unsupported scheme");
        }
        double scale = context.scale();
        size_t oh = ih - kh + 1;
        size_t ow = iw - kw + 1;
        
        GeneralVector x = context.random_polynomial(bs * ic * ih * iw);
        GeneralVector w = context.random_polynomial(oc * ic * kh * kw);
        GeneralVector s = context.random_polynomial(bs * oc * oh * ow);
        Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw,
            context.params_host().poly_modulus_degree(), MatmulObjective::EncryptRight);

        HeContextPointer he = context.context();
        const CKKSEncoder& encoder = context.encoder().ckks();
        const Encryptor& encryptor = context.encryptor();
        const Evaluator& evaluator = context.evaluator();
        const Decryptor& decryptor = context.decryptor();
        
        Plain2d x_encoded = helper.encode_inputs_doubles(encoder, x.doubles().data(), std::nullopt, scale);
        Cipher2d w_encrypted = helper.encrypt_weights_doubles(encryptor, encoder, w.doubles().data(), std::nullopt, scale);
        Plain2d s_encoded = helper.encode_outputs_doubles(encoder, s.doubles().data(), std::nullopt, scale * scale);

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

        vector<double> y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);   

        vector<double> y_truth(bs * oc * oh * ow, 0);
        for (size_t b = 0; b < bs; b++) {
            for (size_t o = 0; o < oc; o++) {
                for (size_t i = 0; i < oh; i++) {
                    for (size_t j = 0; j < ow; j++) {
                        for (size_t c = 0; c < ic; c++) {
                            for (size_t p = 0; p < kh; p++) {
                                for (size_t q = 0; q < kw; q++) {
                                    y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j] +=
                                        x.doubles()[b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q)] *
                                        w.doubles()[o * ic * kh * kw + c * kh * kw + p * kw + q];
                                }
                            }
                        }
                        y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j] += s.doubles()[b * oc * oh * ow + o * oh * ow + i * ow + j];
                    }
                }
            }
        }

        GeneralVector decrypted(std::move(y_decrypted));
        GeneralVector truthv(std::move(y_truth));

        // std::cerr << "Truth:     " << truthv << std::endl;
        // std::cerr << "Decrypted: " << decrypted << std::endl;
        
        ASSERT_TRUE(truthv.near_equal(decrypted, context.tolerance()));
    }



    TEST(Conv2dTest, HostCKKSConv2d) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_conv2d(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, DeviceCKKSConv2d) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_conv2d(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, HostCKKSConv2dReverse) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_conv2d_reverse(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

    TEST(Conv2dTest, DeviceCKKSConv2dReverse) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_conv2d_reverse(ghe, 2, 3, 6, 7, 9, 3, 5, false);
        test_conv2d_reverse(ghe, 2, 3, 10, 56, 56, 10, 10, false);
    }

}