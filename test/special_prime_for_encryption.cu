#include <gtest/gtest.h>
#include "test.cuh"
#include "test_adv.cuh"
#include "../src/key_generator.cuh"
#include "../src/encryptor.cuh"
#include "../src/decryptor.cuh"
#include "../src/evaluator.cuh"

namespace special_prime_for_encryption {

    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;

    void test_encrypt(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);

        Plaintext decrypted = context.decryptor().decrypt_new(encrypted);
        GeneralVector decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));
    }

    TEST(SpecialPrimeForEncryptionTest, HostBFVEncrypt) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            0, 0, 1e-4, false, true);
        test_encrypt(ghe);
    }

    TEST(SpecialPrimeForEncryptionTest, HostBGVEncrypt) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            0, 0, 1e-4, false, true);
        test_encrypt(ghe);
    }

    TEST(SpecialPrimeForEncryptionTest, HostCKKSEncrypt) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            10, 1ull<<20, 1e-2, false, true);
        test_encrypt(ghe);
    }

    TEST(SpecialPrimeForEncryptionTest, DeviceBFVEncrypt) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            0, 0, 1e-4, false, true);
        test_encrypt(ghe);
    }

    TEST(SpecialPrimeForEncryptionTest, DeviceBGVEncrypt) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            0, 0, 1e-4, false, true);
        test_encrypt(ghe);
    }

    TEST(SpecialPrimeForEncryptionTest, DeviceCKKSEncrypt) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            10, 1ull<<20, 1e-2, false, true);
        test_encrypt(ghe);
    }
    
}