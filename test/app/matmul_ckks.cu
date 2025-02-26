#include <gtest/gtest.h>
#include <sstream>
#include "../test_adv.h"
#include "../../src/app/matmul.h"
#include "../test.h"

namespace matmul_ckks {

    using namespace troy;
    using namespace troy::linear;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;
    using std::stringstream;
    using std::vector;

    void test_matmul(const GeneralHeContext& context, size_t m, size_t r, size_t n, bool pack_lwe, bool mod_switch_to_next) {
        SchemeType scheme = context.params_host().scheme();
        if (scheme != SchemeType::CKKS) {
            throw std::runtime_error("[test_matmul] Unsupported scheme");
        }
        double scale = context.scale();
        
        GeneralVector x = context.random_polynomial(m * r);
        GeneralVector w = context.random_polynomial(r * n);
        GeneralVector s = context.random_polynomial(m * n);
        MatmulHelper helper(m, r, n, context.params_host().poly_modulus_degree(), MatmulObjective::EncryptLeft, pack_lwe);

        HeContextPointer he = context.context();
        const CKKSEncoder& encoder = context.encoder().ckks();
        const Encryptor& encryptor = context.encryptor();
        const Evaluator& evaluator = context.evaluator();
        const Decryptor& decryptor = context.decryptor();
        GaloisKeys automorphism_key;
        if (pack_lwe) {
            automorphism_key = context.key_generator().create_automorphism_keys(false);
        }
        
        Cipher2d x_encrypted = helper.encrypt_inputs_doubles(encryptor, encoder, x.doubles().data(), std::nullopt, scale);
        Plain2d w_encoded = helper.encode_weights_doubles(encoder, w.doubles().data(), std::nullopt, scale);
        Plain2d s_encoded = helper.encode_outputs_doubles(encoder, s.doubles().data(), std::nullopt, scale * scale);

        stringstream x_serialized;
        x_encrypted.save(x_serialized, he);
        x_encrypted = Cipher2d::load_new(x_serialized, he);

        Cipher2d y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }
        if (pack_lwe) {
            y_encrypted = helper.pack_outputs(evaluator, automorphism_key, y_encrypted);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<double> y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);   

        vector<double> y_truth(m * n, 0);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < r; k++) {
                    y_truth[i * n + j] += x.doubles()[i * r + k] * w.doubles()[k * n + j];
                }
                y_truth[i * n + j] += s.doubles()[i * n + j];
            }
        }

        GeneralVector decrypted(std::move(y_decrypted));
        GeneralVector truthv(std::move(y_truth));

        // std::cerr << "Truth:     " << truthv << std::endl;
        // std::cerr << "Decrypted: " << decrypted << std::endl;
        
        ASSERT_TRUE(truthv.near_equal(decrypted, context.tolerance()));
    }

    
    void test_matmul_reverse(const GeneralHeContext& context, size_t m, size_t r, size_t n, bool pack_lwe, bool mod_switch_to_next) {
        SchemeType scheme = context.params_host().scheme();
        if (scheme != SchemeType::CKKS) {
            throw std::runtime_error("[test_matmul] Unsupported scheme");
        }
        double scale = context.scale();
        
        GeneralVector x = context.random_polynomial(m * r);
        GeneralVector w = context.random_polynomial(r * n);
        GeneralVector s = context.random_polynomial(m * n);
        MatmulHelper helper(m, r, n, context.params_host().poly_modulus_degree(), MatmulObjective::EncryptRight, pack_lwe);

        HeContextPointer he = context.context();
        const CKKSEncoder& encoder = context.encoder().ckks();
        const Encryptor& encryptor = context.encryptor();
        const Evaluator& evaluator = context.evaluator();
        const Decryptor& decryptor = context.decryptor();
        GaloisKeys automorphism_key;
        if (pack_lwe) {
            automorphism_key = context.key_generator().create_automorphism_keys(false);
        }
        
        Plain2d x_encoded = helper.encode_inputs_doubles(encoder, x.doubles().data(), std::nullopt, scale);
        Cipher2d w_encrypted = helper.encrypt_weights_doubles(encryptor, encoder, w.doubles().data(), std::nullopt, scale);
        Plain2d s_encoded = helper.encode_outputs_doubles(encoder, s.doubles().data(), std::nullopt, scale * scale);

        stringstream w_serialized;
        w_encrypted.save(w_serialized, he);
        w_encrypted = Cipher2d::load_new(w_serialized, he);

        Cipher2d y_encrypted = helper.matmul_reverse(evaluator, x_encoded, w_encrypted);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }
        if (pack_lwe) {
            y_encrypted = helper.pack_outputs(evaluator, automorphism_key, y_encrypted);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<double> y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);   

        vector<double> y_truth(m * n, 0);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < r; k++) {
                    y_truth[i * n + j] += x.doubles()[i * r + k] * w.doubles()[k * n + j];
                }
                y_truth[i * n + j] += s.doubles()[i * n + j];
            }
        }

        GeneralVector decrypted(std::move(y_decrypted));
        GeneralVector truthv(std::move(y_truth));

        // std::cerr << "Truth:     " << truthv << std::endl;
        // std::cerr << "Decrypted: " << decrypted << std::endl;
        
        ASSERT_TRUE(truthv.near_equal(decrypted, context.tolerance()));
    }


    TEST(MatmulTest, HostCKKSMatmul) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_matmul(ghe, 4, 5, 6, false, false);
        test_matmul(ghe, 64, 128, 256, false, false);
        test_matmul(ghe, 4, 5, 6, true, false);
        test_matmul(ghe, 64, 128, 256, true, false);
    }

    TEST(MatmulTest, DeviceCKKSMatmul) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_matmul(ghe, 4, 5, 6, false, false);
        test_matmul(ghe, 64, 128, 256, false, false);
        // test_matmul(ghe, 400, 500, 600, false, false); // very slow!
        test_matmul(ghe, 4, 5, 6, true, false);
        test_matmul(ghe, 64, 128, 256, true, false);
        // test_matmul(ghe, 400, 500, 600, true, false); // very slow!
    }
    
    TEST(MatmulTest, HostCKKSMatmulReverse) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_matmul_reverse(ghe, 4, 5, 6, false, false);
        test_matmul_reverse(ghe, 64, 128, 256, false, false);
        test_matmul_reverse(ghe, 4, 5, 6, true, false);
        test_matmul_reverse(ghe, 64, 128, 256, true, false);
    }

    TEST(MatmulTest, DeviceCKKSMatmulReverse) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 1024, 0, { 60, 40, 40, 60 }, true, 0x123, 2, (double)(1<<20), 1e-2);
        srand(0);
        test_matmul_reverse(ghe, 4, 5, 6, false, false);
        test_matmul_reverse(ghe, 64, 128, 256, false, false);
        // test_matmul(ghe, 400, 500, 600, false, false); // very slow!
        test_matmul_reverse(ghe, 4, 5, 6, true, false);
        test_matmul_reverse(ghe, 64, 128, 256, true, false);
        // test_matmul(ghe, 400, 500, 600, true, false); // very slow!
    }

}