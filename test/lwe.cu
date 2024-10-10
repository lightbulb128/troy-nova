#include <gtest/gtest.h>
#include "test.h"
#include "test_adv.h"
#include "../src/lwe_ciphertext.h"

namespace lwe {    
    
    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;

    void test_extract_lwe(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_polynomial_full();
        Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);

        vector<size_t> terms = {0, 1, 3, 7};
        for (size_t term : terms) {
            auto extracted = context.evaluator().extract_lwe_new(encrypted, term);
            auto assembled = context.evaluator().assemble_lwe_new(extracted);
            if (context.params_host().scheme() == SchemeType::CKKS || context.params_host().scheme() == SchemeType::BGV) {
                context.evaluator().transform_to_ntt_inplace(assembled);
            }
            auto decrypted = context.decryptor().decrypt_new(assembled);
            auto decoded = context.encoder().decode_polynomial(decrypted);
            ASSERT_TRUE(message.element(term).near_equal(decoded.element(0), tolerance));
        }
    }

    TEST(LweTest, HostBFVExtractLWE) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_extract_lwe(ghe);
    }
    TEST(LweTest, HostBGVExtractLWE) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_extract_lwe(ghe);
    }
    TEST(LweTest, HostCKKSExtractLWE) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_extract_lwe(ghe);
    }
    TEST(LweTest, DeviceBFVExtractLWE) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_extract_lwe(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceBGVExtractLWE) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_extract_lwe(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceCKKSExtractLWE) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_extract_lwe(ghe);
        utils::MemoryPool::Destroy();
    }
    
    void test_pack_lwes(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        ASSERT_TRUE(context.params_host().poly_modulus_degree() == 32);

        GaloisKeys automorphism_key = context.key_generator().create_automorphism_keys(false);

        // pack 32 lwes
        GeneralVector message = context.random_polynomial_full();
        Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        std::vector<LWECiphertext> extracted(32);
        for (size_t i = 0; i < 32; i++) {
            extracted[i] = context.evaluator().extract_lwe_new(encrypted, i);
        }

        Ciphertext assembled = context.evaluator().pack_lwe_ciphertexts_new(extracted, automorphism_key);
        Plaintext decrypted = context.decryptor().decrypt_new(assembled);
        GeneralVector decoded = context.encoder().decode_polynomial(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));

        // pack 7 lwes
        for (size_t i = 0; i < 32; i++) {
            if (i % 4 == 0 && i / 4 < 7) continue;
            if (message.is_integers()) message.integers()[i] = 0;
            else message.doubles()[i] = 0;
        }
        extracted.resize(7);
        for (size_t i = 0; i < 7; i++) {
            extracted[i] = context.evaluator().extract_lwe_new(encrypted, i * 4);
        }
        assembled = context.evaluator().pack_lwe_ciphertexts_new(extracted, automorphism_key);
        decrypted = context.decryptor().decrypt_new(assembled);
        decoded = context.encoder().decode_polynomial(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));
    }
    
    TEST(LweTest, HostBFVPackLWEs) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes(ghe);
    }
    TEST(LweTest, HostBGVPackLWEs) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes(ghe);
    }
    TEST(LweTest, HostCKKSPackLWEs) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_lwes(ghe);
    }
    TEST(LweTest, DeviceBFVPackLWEs) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceBGVPackLWEs) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceCKKSPackLWEs) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_lwes(ghe);
        utils::MemoryPool::Destroy();
    }


    void test_pack_rlwes(const GeneralHeContext& context) {
        uint64_t t = context.t();
        std::cerr << "t = " << t << std::endl;
        double scale = context.scale();
        double tolerance = context.tolerance();

        const size_t poly_modulus_degree = context.params_host().poly_modulus_degree();
        ASSERT_TRUE(context.params_host().poly_modulus_degree() == 32);

        GaloisKeys automorphism_key = context.key_generator().create_automorphism_keys(false);

        // pack 7 lwes with shift -3, input interval 8, output interval 1
        auto test_setting = [&](size_t n, size_t input_interval, size_t output_interval, int shift_){
            ASSERT_TRUE(n <= input_interval / output_interval);
            ASSERT_TRUE(shift_ <= 0 && static_cast<size_t>(-shift_) < input_interval);
            size_t shift = 2 * poly_modulus_degree + shift_;
            auto message = context.batch_random_polynomial_full(n);
            auto encoded = context.encoder().batch_encode_polynomial(message, std::nullopt, scale);
            auto encrypted = context.batch_encrypt_asymmetric(encoded);
            GeneralVector truth = GeneralVector::zeros_like(message[0], message[0].size());
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < poly_modulus_degree; j += input_interval) {
                    truth.set(i * output_interval + j, message[i].element(j - shift_));
                }
            }

            Ciphertext assembled = context.evaluator().pack_rlwe_ciphertexts_new(batch_utils::collect_const_pointer(encrypted),  automorphism_key, shift, input_interval, output_interval); 
            Plaintext decrypted = context.decryptor().decrypt_new(assembled);
            GeneralVector decoded = context.encoder().decode_polynomial(decrypted);
            ASSERT_TRUE(truth.near_equal(decoded, tolerance));
        };

        test_setting(32, 32, 1, 0);
        test_setting(16, 16, 1, 0);
        test_setting(3, 8, 2, -3);
    }

    TEST(LweTest, HostBFVPackRLWEs) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes(ghe);
    }
    TEST(LweTest, HostBGVPackRLWEs) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes(ghe);
    }
    TEST(LweTest, HostCKKSPackRLWEs) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_rlwes(ghe);
    }
    TEST(LweTest, DeviceBFVPackRLWEs) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceBGVPackRLWEs) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceCKKSPackRLWEs) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_rlwes(ghe);
        utils::MemoryPool::Destroy();
    }

}