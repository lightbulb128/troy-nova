#include <gtest/gtest.h>
#include "test.cuh"
#include "test_adv.cuh"
#include "../src/key_generator.cuh"
#include "../src/encryptor.cuh"
#include "../src/decryptor.cuh"
#include "../src/evaluator.cuh"

namespace evaluator {

    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;

    void test_negate(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();
        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        encrypted.scale() = scale;
        Ciphertext negated = context.evaluator().negate_new(encrypted);
        Plaintext decrypted = context.decryptor().decrypt_new(negated);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(result.negate(t), tolerance));
    }

    TEST(EvaluatorTest, HostBFVNegate) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
    }
    TEST(EvaluatorTest, HostBGVNegate) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
    }
    TEST(EvaluatorTest, HostCKKSNegate) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_negate(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVNegate) {
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
    }
    TEST(EvaluatorTest, DeviceBGVNegate) {
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
    }
    TEST(EvaluatorTest, DeviceCKKSNegate) {
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_negate(ghe);
    }

    void test_add_subtract(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
        Ciphertext added = context.evaluator().add_new(encrypted1, encrypted2);
        Plaintext decrypted = context.decryptor().decrypt_new(added);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.add(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        Ciphertext subtracted = context.evaluator().sub_new(encrypted1, encrypted2);
        decrypted = context.decryptor().decrypt_new(subtracted);
        result = context.encoder().decode_simd(decrypted);
        truth = message1.sub(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

    }

    TEST(EvaluatorTest, HostBFVAdd) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorTest, HostBGVAdd) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorTest, HostCKKSAdd) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVAdd) {
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorTest, DeviceBGVAdd) {
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorTest, DeviceCKKSAdd) {
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract(ghe);
    }

    void test_multiply(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

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
    

    TEST(EvaluatorTest, HostBFVMultiply) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
    }
    TEST(EvaluatorTest, HostBGVMultiply) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
    }
    TEST(EvaluatorTest, HostCKKSMultiply) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 40 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVMultiply) {
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
    }
    TEST(EvaluatorTest, DeviceBGVMultiply) {
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply(ghe);
    }
    TEST(EvaluatorTest, DeviceCKKSMultiply) {
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 40 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply(ghe);
    }

}