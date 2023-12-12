#include <gtest/gtest.h>
#include "test.cuh"
#include "test_adv.cuh"
#include "../src/lwe_ciphertext.cuh"

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
            if (context.params_host().scheme() == SchemeType::CKKS) {
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
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_extract_lwe(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceBGVExtractLWE) {
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_extract_lwe(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceCKKSExtractLWE) {
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
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceBGVPackLWEs) {
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceCKKSPackLWEs) {
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_lwes(ghe);
        utils::MemoryPool::Destroy();
    }

}