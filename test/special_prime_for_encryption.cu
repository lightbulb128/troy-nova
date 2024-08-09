#include <gtest/gtest.h>
#include "test.h"
#include "test_adv.h"
#include "../src/key_generator.h"
#include "../src/encryptor.h"
#include "../src/decryptor.h"
#include "../src/evaluator.h"

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
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            0, 0, 1e-4, false, true);
        test_encrypt(ghe);
    }

    TEST(SpecialPrimeForEncryptionTest, DeviceBGVEncrypt) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            0, 0, 1e-4, false, true);
        test_encrypt(ghe);
    }

    TEST(SpecialPrimeForEncryptionTest, DeviceCKKSEncrypt) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 20, {60, 40, 40, 60}, true, 0x123, 
            10, 1ull<<20, 1e-2, false, true);
        test_encrypt(ghe);
    }
    

    void test_multiply(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        std::cerr << context.params_host() << std::endl;

        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
        Ciphertext multiplied = context.evaluator().multiply_new(encrypted1, encrypted2);
        Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.mul(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        // test multiply then add
        GeneralVector message3 = context.random_simd_full();
        Plaintext encoded3 = context.encoder().encode_simd(message3, std::nullopt, multiplied.scale());
        Ciphertext encrypted3 = context.encryptor().encrypt_asymmetric_new(encoded3);
        Ciphertext added = context.evaluator().add_new(multiplied, encrypted3);
        decrypted = context.decryptor().decrypt_new(added);
        result = context.encoder().decode_simd(decrypted);
        truth = truth.add(message3, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));
    }

    TEST(SpecialPrimeForEncryptionTest, HostBFVMultiply) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
    }
    TEST(SpecialPrimeForEncryptionTest, HostBGVMultiply) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
    }
    TEST(SpecialPrimeForEncryptionTest, HostCKKSMultiply) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply(ghe);
    }
    TEST(SpecialPrimeForEncryptionTest, DeviceBFVMultiply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SpecialPrimeForEncryptionTest, DeviceBGVMultiply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SpecialPrimeForEncryptionTest, DeviceCKKSMultiply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply(ghe);
        utils::MemoryPool::Destroy();
    }


    void test_relinearize(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        RelinKeys relin_keys = context.key_generator().create_relin_keys(false, 3);
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
        Ciphertext multiplied = context.evaluator().multiply_new(encrypted1, encrypted2);
        Ciphertext relined = context.evaluator().relinearize_new(multiplied, relin_keys);
        Plaintext decrypted = context.decryptor().decrypt_new(relined);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.mul(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        GeneralVector message3 = context.random_simd_full();
        Plaintext encoded3 = context.encoder().encode_simd(message3, std::nullopt, relined.scale());
        Ciphertext encrypted3 = context.encryptor().encrypt_asymmetric_new(encoded3);
        multiplied = context.evaluator().multiply_new(multiplied, encrypted3);
        relined = context.evaluator().relinearize_new(multiplied, relin_keys);
        decrypted = context.decryptor().decrypt_new(relined);
        result = context.encoder().decode_simd(decrypted);
        truth = truth.mul(message3, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

    }

    TEST(SpecialPrimeForEncryptionTest, HostBFVRelinearize) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_relinearize(ghe);
    }
    TEST(SpecialPrimeForEncryptionTest, HostBGVRelinearize) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_relinearize(ghe);
    }
    TEST(SpecialPrimeForEncryptionTest, HostCKKSRelinearize) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_relinearize(ghe);
    }
    TEST(SpecialPrimeForEncryptionTest, DeviceBFVRelinearize) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_relinearize(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SpecialPrimeForEncryptionTest, DeviceBGVRelinearize) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_relinearize(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SpecialPrimeForEncryptionTest, DeviceCKKSRelinearize) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_relinearize(ghe);
        utils::MemoryPool::Destroy();
    }


}