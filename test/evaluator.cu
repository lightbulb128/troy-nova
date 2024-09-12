#include "cuda_runtime.h"
#include <gtest/gtest.h>
#include "test.h"
#include "test_adv.h"
#include "../src/key_generator.h"
#include "../src/encryptor.h"
#include "../src/decryptor.h"
#include "../src/evaluator.h"

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

        { // new
            Ciphertext negated = context.evaluator().negate_new(encrypted);
            Plaintext decrypted = context.decryptor().decrypt_new(negated);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(message.near_equal(result.negate(t), tolerance));
        }
        { // assign
            Ciphertext negated; context.evaluator().negate(encrypted, negated);
            Plaintext decrypted = context.decryptor().decrypt_new(negated);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(message.near_equal(result.negate(t), tolerance));
        }
        { // inplace
            Ciphertext negated = encrypted.clone();
            context.evaluator().negate_inplace(negated);
            Plaintext decrypted = context.decryptor().decrypt_new(negated);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(message.near_equal(result.negate(t), tolerance));
        }
    }

    TEST(EvaluatorTest, HostBFVNegate) {
        {
            GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_negate(ghe);
        }
        {
            GeneralHeContext ghe(false, SchemeType::BFV, 32, 35, { 30, 30, 30, 30 }, false, 0x123, 0);
            test_negate(ghe);
        }
    }
    TEST(EvaluatorTest, HostBGVNegate) {
        {
            GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_negate(ghe);
        }
        {
            GeneralHeContext ghe(false, SchemeType::BGV, 32, 35, { 30, 30, 30, 30 }, false, 0x123, 0);
            test_negate(ghe);
        }
    }
    TEST(EvaluatorTest, HostCKKSNegate) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_negate(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVNegate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        {
            GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_negate(ghe);
        }
        {
            GeneralHeContext ghe(true, SchemeType::BFV, 32, 35, { 30, 30, 30 }, false, 0x123, 0);
            test_negate(ghe);
        }
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVNegate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        {
            GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_negate(ghe);
        }
        {
            GeneralHeContext ghe(true, SchemeType::BGV, 32, 35, { 30, 30, 30 }, false, 0x123, 0);
            test_negate(ghe);
        }
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSNegate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_negate(ghe);
        utils::MemoryPool::Destroy();
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

        { // new
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
        { // assign
            Ciphertext added; context.evaluator().add(encrypted1, encrypted2, added);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message1.add(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));

            Ciphertext subtracted; context.evaluator().sub(encrypted1, encrypted2, subtracted);
            decrypted = context.decryptor().decrypt_new(subtracted);
            result = context.encoder().decode_simd(decrypted);
            truth = message1.sub(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
        { // inplace
            Ciphertext added = encrypted1.clone();
            context.evaluator().add_inplace(added, encrypted2);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message1.add(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));

            Ciphertext subtracted = encrypted1.clone();
            context.evaluator().sub_inplace(subtracted, encrypted2);
            decrypted = context.decryptor().decrypt_new(subtracted);
            result = context.encoder().decode_simd(decrypted);
            truth = message1.sub(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
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
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVAdd) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSAdd) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_add_subtract_ntt(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
        context.evaluator().transform_to_ntt_inplace(encrypted1);
        context.evaluator().transform_to_ntt_inplace(encrypted2);
        Ciphertext added = context.evaluator().add_new(encrypted1, encrypted2);
        context.evaluator().transform_from_ntt_inplace(added);
        Plaintext decrypted = context.decryptor().decrypt_new(added);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.add(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        Ciphertext subtracted = context.evaluator().sub_new(encrypted1, encrypted2);
        context.evaluator().transform_from_ntt_inplace(subtracted);
        decrypted = context.decryptor().decrypt_new(subtracted);
        result = context.encoder().decode_simd(decrypted);
        truth = message1.sub(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostBFVAddNTT) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_ntt(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVAddNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_ntt(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_add_subtract_intt(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
        context.evaluator().transform_from_ntt_inplace(encrypted1);
        context.evaluator().transform_from_ntt_inplace(encrypted2);
        Ciphertext added = context.evaluator().add_new(encrypted1, encrypted2);
        context.evaluator().transform_to_ntt_inplace(added);
        Plaintext decrypted = context.decryptor().decrypt_new(added);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.add(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        Ciphertext subtracted = context.evaluator().sub_new(encrypted1, encrypted2);
        context.evaluator().transform_to_ntt_inplace(subtracted);
        decrypted = context.decryptor().decrypt_new(subtracted);
        result = context.encoder().decode_simd(decrypted);
        truth = message1.sub(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostBGVAddINTT) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_intt(ghe);
    }
    TEST(EvaluatorTest, HostCKKSAddINTT) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract_intt(ghe);
    }
    TEST(EvaluatorTest, DeviceBGVAddINTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_intt(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSAddINTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract_intt(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_multiply(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();
    
        {
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

        {
            size_t n = context.params_host().poly_modulus_degree();
            {
                GeneralVector message1 = context.random_polynomial(n);
                GeneralVector message2 = context.random_polynomial(n);
                Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
                Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
                Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
                Ciphertext multiplied = context.evaluator().multiply_new(encrypted1, encrypted2);
                Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
                GeneralVector result = context.encoder().decode_polynomial(decrypted);
                GeneralVector truth = context.mul_poly(message1, message2);
                ASSERT_TRUE(truth.near_equal(result, tolerance));
            }
            {
                size_t c1 = n / 3;
                size_t c2 = n / 4;
                GeneralVector message1 = context.random_polynomial(c1);
                GeneralVector message2 = context.random_polynomial(c2);
                Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
                Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
                Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
                Ciphertext multiplied = context.evaluator().multiply_new(encrypted1, encrypted2);
                Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
                GeneralVector result = context.encoder().decode_polynomial(decrypted);
                GeneralVector truth = context.mul_poly(message1, message2);
                ASSERT_TRUE(truth.near_equal(result, tolerance));
            }
        }
    }
    

    TEST(EvaluatorTest, HostBFVMultiply) {
        {
            GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_multiply(ghe);
        }
        {
            // GeneralHeContext ghe(false, SchemeType::BFV, 32, 35, { 30, 30, 30, 30 }, false, 0x123, 0);
            // test_multiply(ghe);
        }
    }
    TEST(EvaluatorTest, HostBGVMultiply) {
        {
            GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_multiply(ghe);
        }
        {
            GeneralHeContext ghe(false, SchemeType::BGV, 32, 35, { 30, 30, 30, 30 }, false, 0x123, 0);
            test_multiply(ghe);
        }
    }
    TEST(EvaluatorTest, HostCKKSMultiply) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVMultiply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        {
            GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_multiply(ghe);
        }
        {
            GeneralHeContext ghe(true, SchemeType::BFV, 32, 35, { 30, 30, 30, 30 }, false, 0x123, 0);
            test_multiply(ghe);
        }
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVMultiply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        {
            GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
            test_multiply(ghe);
        }
        {
            GeneralHeContext ghe(true, SchemeType::BGV, 32, 35, { 30, 30, 30, 30 }, false, 0x123, 0);
            test_multiply(ghe);
        }
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSMultiply) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_square(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        Ciphertext squared = context.evaluator().square_new(encrypted);
        Plaintext decrypted = context.decryptor().decrypt_new(squared);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message.square(t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostBFVSquare) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_square(ghe);
    }
    TEST(EvaluatorTest, HostBGVSquare) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_square(ghe);
    }
    TEST(EvaluatorTest, HostCKKSSquare) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_square(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVSquare) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_square(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVSquare) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_square(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSSquare) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_square(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_keyswitching(const GeneralHeContext& context) {
        // uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        // create another keygenerator
        KeyGenerator keygen_other = KeyGenerator(context.context());
        SecretKey secret_key_other = keygen_other.secret_key();
        Encryptor encryptor_other = Encryptor(context.context());
        encryptor_other.set_secret_key(secret_key_other);

        KSwitchKeys kswitch_key = context.key_generator().create_keyswitching_key(secret_key_other, false);

        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext encrypted = encryptor_other.encrypt_symmetric_new(encoded, false);
        Ciphertext switched = context.evaluator().apply_keyswitching_new(encrypted, kswitch_key);
        Plaintext decrypted = context.decryptor().decrypt_new(switched);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message;
        ASSERT_TRUE(truth.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostBFVKeySwitching) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
    }
    TEST(EvaluatorTest, HostBGVKeySwitching) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
    }
    TEST(EvaluatorTest, HostCKKSKeySwitching) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_keyswitching(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVKeySwitching) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVKeySwitching) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSKeySwitching) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_keyswitching(ghe);
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

    TEST(EvaluatorTest, HostBFVRelinearize) {
        {
            GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
        {
            GeneralHeContext ghe(false, SchemeType::BFV, 32, 35, { 60, 30, 30, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
    }
    TEST(EvaluatorTest, HostBGVRelinearize) {
        {
            GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
        {
            GeneralHeContext ghe(false, SchemeType::BGV, 32, 35, { 60, 30, 30, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
    }
    TEST(EvaluatorTest, HostCKKSRelinearize) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_relinearize(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVRelinearize) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        {
            GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
        {
            GeneralHeContext ghe(true, SchemeType::BFV, 32, 35, { 60, 30, 30, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVRelinearize) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        {
            GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
        {
            GeneralHeContext ghe(false, SchemeType::BFV, 32, 35, { 60, 30, 30, 60 }, false, 0x123, 0);
            test_relinearize(ghe);
        }
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSRelinearize) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_relinearize(ghe);
        utils::MemoryPool::Destroy();
    }


    void test_mod_switch_to_next(const GeneralHeContext& context) {
        // uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        Ciphertext switched = context.evaluator().mod_switch_to_next_new(encrypted);
        Plaintext decrypted = context.decryptor().decrypt_new(switched);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        // result should be same with message
        ASSERT_TRUE(message.near_equal(result, tolerance));

        switched = context.evaluator().mod_switch_to_new(encrypted, context.context()->last_parms_id());
        decrypted = context.decryptor().decrypt_new(switched);
        result = context.encoder().decode_simd(decrypted);
        // result should be same with message
        ASSERT_TRUE(message.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostBFVModSwitchToNext) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
    }
    TEST(EvaluatorTest, HostBGVModSwitchToNext) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
    }
    TEST(EvaluatorTest, HostCKKSModSwitchToNext) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_mod_switch_to_next(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVModSwitchToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVModSwitchToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSModSwitchToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_mod_switch_to_next(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_mod_switch_plain_to_next(const GeneralHeContext& context) {
        // context must be ckks
        ASSERT_TRUE(context.params_host().scheme() == SchemeType::CKKS);
        // uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Plaintext switched = context.evaluator().mod_switch_plain_to_next_new(encoded);
        GeneralVector result = context.encoder().decode_simd(switched);
        // result should be same with message
        ASSERT_TRUE(message.near_equal(result, tolerance));

        switched = context.evaluator().mod_switch_plain_to_new(encoded, context.context()->first_parms_id());
        result = context.encoder().decode_simd(switched);
        // result should be same with message
        ASSERT_TRUE(message.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostCKKSModSwitchPlainToNext) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_mod_switch_plain_to_next(ghe);
    }
    TEST(EvaluatorTest, DeviceCKKSModSwitchPlainToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_mod_switch_plain_to_next(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_rescale_to_next(const GeneralHeContext& context) {
        // context must be ckks
        ASSERT_TRUE(context.params_host().scheme() == SchemeType::CKKS);
        // uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_simd_full();
        const EncryptionParameters& parms = context.params_host();
        auto coeff_modulus = parms.coeff_modulus();
        double expanded_scale = scale * coeff_modulus[coeff_modulus.size() - 2].value();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, expanded_scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        Ciphertext rescaled = context.evaluator().rescale_to_next_new(encrypted);
        Plaintext decrypted = context.decryptor().decrypt_new(rescaled);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        // result should be same with message
        ASSERT_TRUE(message.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostCKKSRescaleToNext) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_rescale_to_next(ghe);
    }
    TEST(EvaluatorTest, DeviceCKKSRescaleToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_rescale_to_next(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_add_subtract_plain(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        {
            GeneralVector message1 = context.random_simd_full();
            GeneralVector message2 = context.random_simd_full();
            Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
            Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            Ciphertext added = context.evaluator().add_plain_new(encrypted1, encoded2);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message1.add(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));

            Ciphertext subtracted = context.evaluator().sub_plain_new(encrypted1, encoded2);
            decrypted = context.decryptor().decrypt_new(subtracted);
            result = context.encoder().decode_simd(decrypted);
            truth = message1.sub(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }

        {
            size_t n = context.encoder().slot_count();
            size_t cc = n / 3;
            GeneralVector message1 = context.random_polynomial(cc);
            GeneralVector message2 = context.random_polynomial(cc);
            Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
            Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            Ciphertext added = context.evaluator().add_plain_new(encrypted1, encoded2);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_polynomial(decrypted);
            GeneralVector truth = message1.add(message2, t); truth.resize(n);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
    }

    TEST(EvaluatorTest, HostBFVAddPlain) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_plain(ghe);
    }
    TEST(EvaluatorTest, HostBGVAddPlain) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_plain(ghe);
    }
    TEST(EvaluatorTest, HostCKKSAddPlain) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract_plain(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVAddPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_plain(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVAddPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract_plain(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSAddPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract_plain(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_add_plain_scaled(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        {
            GeneralVector message1 = context.random_simd_full();
            GeneralVector message2 = context.random_simd_full();
            Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
            Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            context.encoder().batch().scale_up_inplace(encoded2, encrypted1.parms_id());
            Ciphertext added = context.evaluator().add_plain_new(encrypted1, encoded2);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message1.add(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }

        {
            size_t n = context.encoder().slot_count();
            size_t cc = n / 3;
            GeneralVector message1 = context.random_polynomial(cc);
            GeneralVector message2 = context.random_polynomial(cc);
            Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
            ASSERT_EQ(encoded1.coeff_count(), cc);
            Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
            ASSERT_EQ(encoded2.coeff_count(), cc);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            context.encoder().batch().scale_up_inplace(encoded2, encrypted1.parms_id());
            ASSERT_EQ(encoded2.coeff_count(), cc);
            ASSERT_EQ(encoded2.poly_modulus_degree(), n);
            ASSERT_EQ(encoded2.data().size(), encoded2.coeff_modulus_size() * cc);
            Ciphertext added = context.evaluator().add_plain_new(encrypted1, encoded2);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_polynomial(decrypted);
            GeneralVector truth = message1.add(message2, t); truth.resize(n);
            std::cout << result << std::endl;
            std::cout << truth << std::endl;
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
    }

    TEST(EvaluatorTest, HostBFVAddPlainScaled) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_plain_scaled(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVAddPlainScaled) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_plain_scaled(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_add_plain_scaled_ntt(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        {
            GeneralVector message1 = context.random_simd_full();
            GeneralVector message2 = context.random_simd_full();
            Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
            Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            context.encoder().batch().scale_up_inplace(encoded2, encrypted1.parms_id());
            context.evaluator().transform_to_ntt_inplace(encrypted1);
            context.evaluator().transform_plain_to_ntt_inplace(encoded2, encrypted1.parms_id());
            Ciphertext added = context.evaluator().add_plain_new(encrypted1, encoded2);
            context.evaluator().transform_from_ntt_inplace(added);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message1.add(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }

        {
            size_t n = context.encoder().slot_count();
            size_t cc = n / 3;
            GeneralVector message1 = context.random_polynomial(cc);
            GeneralVector message2 = context.random_polynomial(cc);
            Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
            ASSERT_EQ(encoded1.coeff_count(), cc);
            Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
            ASSERT_EQ(encoded2.coeff_count(), cc);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            context.encoder().batch().scale_up_inplace(encoded2, encrypted1.parms_id());
            ASSERT_EQ(encoded2.coeff_count(), cc);
            ASSERT_EQ(encoded2.poly_modulus_degree(), n);
            ASSERT_EQ(encoded2.data().size(), encoded2.coeff_modulus_size() * cc);
            context.evaluator().transform_to_ntt_inplace(encrypted1);
            context.evaluator().transform_plain_to_ntt_inplace(encoded2, encrypted1.parms_id());
            ASSERT_EQ(encoded2.coeff_count(), n);
            ASSERT_EQ(encoded2.poly_modulus_degree(), n);
            ASSERT_EQ(encoded2.data().size(), encoded2.coeff_modulus_size() * n);
            Ciphertext added = context.evaluator().add_plain_new(encrypted1, encoded2);
            context.evaluator().transform_from_ntt_inplace(added);
            Plaintext decrypted = context.decryptor().decrypt_new(added);
            GeneralVector result = context.encoder().decode_polynomial(decrypted);
            GeneralVector truth = message1.add(message2, t); truth.resize(n);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
    }

    TEST(EvaluatorTest, HostBFVAddPlainScaledNTT) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_plain_scaled_ntt(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVAddPlainScaledNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_plain_scaled_ntt(ghe);
        utils::MemoryPool::Destroy();
    }


    void test_multiply_plain(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        // native multiply
        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        Ciphertext multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
        Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.mul(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        // if encrypted is not ntt
        if (!encrypted1.is_ntt_form()) {
            context.evaluator().transform_to_ntt_inplace(encrypted1);
            context.evaluator().transform_plain_to_ntt_inplace(encoded2, encrypted1.parms_id());
            multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
            context.evaluator().transform_from_ntt_inplace(multiplied);
            decrypted = context.decryptor().decrypt_new(multiplied);
            result = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        } else {
            context.evaluator().transform_from_ntt_inplace(encrypted1);
            context.evaluator().transform_to_ntt_inplace(encrypted1);
            multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
            decrypted = context.decryptor().decrypt_new(multiplied);
            result = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }

        {
            size_t n = context.params_host().poly_modulus_degree();
            size_t cc = n / 3;
            GeneralVector message1 = context.random_polynomial(cc);
            GeneralVector message2 = context.random_polynomial(cc);
            Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
            Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            Ciphertext multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
            Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
            GeneralVector result = context.encoder().decode_polynomial(decrypted);
            GeneralVector truth = context.mul_poly(message1, message2); truth.resize(n);
            ASSERT_TRUE(truth.near_equal(result, tolerance));

            if (!encrypted1.is_ntt_form()) {
                context.evaluator().transform_to_ntt_inplace(encrypted1);
                context.evaluator().transform_plain_to_ntt_inplace(encoded2, encrypted1.parms_id());
                multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
                context.evaluator().transform_from_ntt_inplace(multiplied);
                decrypted = context.decryptor().decrypt_new(multiplied);
                result = context.encoder().decode_polynomial(decrypted);
                ASSERT_TRUE(truth.near_equal(result, tolerance));
            } else {
                context.evaluator().transform_from_ntt_inplace(encrypted1);
                context.evaluator().transform_to_ntt_inplace(encrypted1);
                multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
                decrypted = context.decryptor().decrypt_new(multiplied);
                result = context.encoder().decode_polynomial(decrypted);
                ASSERT_TRUE(truth.near_equal(result, tolerance));
            }
        }
    }

    TEST(EvaluatorTest, HostBFVMultiplyPlain) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
    }
    TEST(EvaluatorTest, HostBGVMultiplyPlain) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
    }
    TEST(EvaluatorTest, HostCKKSMultiplyPlain) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply_plain(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVMultiplyPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVMultiplyPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSMultiplyPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply_plain(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_multiply_plain_ntt(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        // native multiply
        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        context.evaluator().transform_plain_to_ntt_inplace(encoded2, encrypted1.parms_id());
        Ciphertext multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
        Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.mul(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        {
            size_t n = context.params_host().poly_modulus_degree();
            size_t cc = n / 3;
            GeneralVector message1 = context.random_polynomial(cc);
            GeneralVector message2 = context.random_polynomial(cc);
            Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
            ASSERT_EQ(encoded1.coeff_count(), cc);
            Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
            ASSERT_EQ(encoded2.coeff_count(), cc);
            Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            context.evaluator().transform_plain_to_ntt_inplace(encoded2, encrypted1.parms_id());
            ASSERT_EQ(encoded2.coeff_count(), n);
            ASSERT_EQ(encoded2.poly_modulus_degree(), n);
            ASSERT_EQ(encoded2.data().size(), encoded2.coeff_modulus_size() * n);
            Ciphertext multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
            Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
            GeneralVector result = context.encoder().decode_polynomial(decrypted);
            GeneralVector truth = context.mul_poly(message1, message2); truth.resize(n);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
    }

    TEST(EvaluatorTest, HostBFVMultiplyPlainNTT) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
    }
    TEST(EvaluatorTest, HostBGVMultiplyPlainNTT) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVMultiplyPlainNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVMultiplyPlainNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_multiply_plain_centralized(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message1 = context.random_simd_full();
        GeneralVector message2 = context.random_simd_full();
        Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
        Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
        Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
        context.encoder().batch().centralize_inplace(encoded2, std::nullopt);
        Ciphertext multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
        Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message1.mul(message2, t);
        ASSERT_TRUE(truth.near_equal(result, tolerance));

        {
            size_t n = context.encoder().slot_count();
            size_t cc = n / 3;
            {
                GeneralVector message1 = context.random_polynomial(n);
                GeneralVector message2 = context.random_polynomial(n);
                Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
                ASSERT_EQ(encoded1.coeff_count(), n);
                Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
                ASSERT_EQ(encoded2.coeff_count(), n);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
                context.encoder().batch().centralize_inplace(encoded2, std::nullopt);
                ASSERT_EQ(encoded2.coeff_count(), n);
                ASSERT_EQ(encoded2.poly_modulus_degree(), n);
                ASSERT_EQ(encoded2.data().size(), encoded2.coeff_modulus_size() * n);
                Ciphertext multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
                Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
                GeneralVector result = context.encoder().decode_polynomial(decrypted);
                GeneralVector truth = context.mul_poly(message1, message2); truth.resize(n);
                ASSERT_TRUE(truth.near_equal(result, tolerance));
            }
            {
                GeneralVector message1 = context.random_polynomial(cc);
                GeneralVector message2 = context.random_polynomial(cc);
                Plaintext encoded1 = context.encoder().encode_polynomial(message1, std::nullopt, scale);
                ASSERT_EQ(encoded1.coeff_count(), cc);
                Plaintext encoded2 = context.encoder().encode_polynomial(message2, std::nullopt, scale);
                ASSERT_EQ(encoded2.coeff_count(), cc);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
                context.encoder().batch().centralize_inplace(encoded2, std::nullopt);
                ASSERT_EQ(encoded2.coeff_count(), cc);
                ASSERT_EQ(encoded2.poly_modulus_degree(), n);
                ASSERT_EQ(encoded2.data().size(), encoded2.coeff_modulus_size() * cc);
                Ciphertext multiplied = context.evaluator().multiply_plain_new(encrypted1, encoded2);
                Plaintext decrypted = context.decryptor().decrypt_new(multiplied);
                GeneralVector result = context.encoder().decode_polynomial(decrypted);
                GeneralVector truth = context.mul_poly(message1, message2); truth.resize(n);
                ASSERT_TRUE(truth.near_equal(result, tolerance));
            
            }
        }
    }

    TEST(EvaluatorTest, HostBFVMultiplyPlainCentralized) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_centralized(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVMultiplyPlainCentralized) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_centralized(ghe);
        utils::MemoryPool::Destroy();
    }


    void test_rotate(const GeneralHeContext& context) {
        // uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        Ciphertext rotated;
        {
            if (context.params_host().scheme() == SchemeType::CKKS) {
                rotated = context.evaluator().rotate_vector_new(encrypted, 1, glk);
            } else {
                rotated = context.evaluator().rotate_rows_new(encrypted, 1, glk);
            }
            Plaintext decrypted = context.decryptor().decrypt_new(rotated);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message.rotate(1);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
        
        // rotate more steps
        {
            int step = 7;
            if (context.params_host().scheme() == SchemeType::CKKS) {
                rotated = context.evaluator().rotate_vector_new(encrypted, step, glk);
            } else {
                rotated = context.evaluator().rotate_rows_new(encrypted, step, glk);
            }
            Plaintext decrypted = context.decryptor().decrypt_new(rotated);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message.rotate(step);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
    }

    TEST(EvaluatorTest, HostBFVRotateRows) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
    }
    TEST(EvaluatorTest, HostBGVRotateRows) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
    }
    TEST(EvaluatorTest, HostCKKSRotateVector) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_rotate(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVRotateRows) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVRotateRows) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSRotateVector) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_rotate(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_conjugate(const GeneralHeContext& context) {
        // uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        Ciphertext rotated;
        if (context.params_host().scheme() == SchemeType::CKKS) {
            rotated = context.evaluator().complex_conjugate_new(encrypted, glk);
        } else {
            rotated = context.evaluator().rotate_columns_new(encrypted, glk);
        }
        Plaintext decrypted = context.decryptor().decrypt_new(rotated);
        GeneralVector result = context.encoder().decode_simd(decrypted);
        GeneralVector truth = message.conjugate();
        ASSERT_TRUE(truth.near_equal(result, tolerance));
    }

    TEST(EvaluatorTest, HostBFVRotateColumns) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
    }
    TEST(EvaluatorTest, HostBGVRotateColumns) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
    }
    TEST(EvaluatorTest, HostCKKSComplexConjugate) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_conjugate(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVRotateColumns) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVRotateColumns) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSComplexConjugate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_conjugate(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_transform_plain_ntt(const GeneralHeContext& context) {
        double scale = context.scale();
        size_t n = context.params_host().poly_modulus_degree();
        size_t cc = n / 3;
        SchemeType scheme = context.params_host().scheme();

        {
            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
            if (!encoded.is_ntt_form()) {
                context.evaluator().transform_plain_to_ntt_inplace(encoded, context.context()->first_parms_id());
            }

            Plaintext t1 = context.evaluator().transform_plain_from_ntt_new(encoded);
            Plaintext result = context.evaluator().transform_plain_to_ntt_new(t1, encoded.parms_id());
            ASSERT_TRUE(encoded.data().to_vector() == result.data().to_vector());
        }

        {
            GeneralVector message = context.random_polynomial(cc);
            Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);
            if (!encoded.is_ntt_form()) {
                context.evaluator().transform_plain_to_ntt_inplace(encoded, context.context()->first_parms_id());
            }

            Plaintext t1 = context.evaluator().transform_plain_from_ntt_new(encoded);
            Plaintext result = context.evaluator().transform_plain_to_ntt_new(t1, encoded.parms_id());
            ASSERT_TRUE(encoded.data().to_vector() == result.data().to_vector());
        }

        {
            // try encrypt and decrypt
            GeneralVector message = context.random_polynomial(cc);
            Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);
            if (encoded.is_ntt_form()) {
                context.evaluator().transform_plain_from_ntt_inplace(encoded);
            } else {
                if (scheme == SchemeType:: BFV) {
                    context.encoder().batch().scale_up_inplace(encoded, std::nullopt);
                }
                context.evaluator().transform_plain_to_ntt_inplace(encoded, context.context()->first_parms_id());
            }
            Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
            if (scheme == SchemeType::CKKS) {
                context.evaluator().transform_to_ntt_inplace(encrypted);
            } 
            if (scheme == SchemeType::BFV) {
                context.evaluator().transform_from_ntt_inplace(encrypted);
            }
            Plaintext decrypted = context.decryptor().decrypt_new(encrypted);
            GeneralVector result = context.encoder().decode_polynomial(decrypted);
            ASSERT_TRUE(message.near_equal(result, context.tolerance()));
        }


    }

    TEST(EvaluatorTest, HostBFVTransformPlainNTT) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_transform_plain_ntt(ghe);
    }
    TEST(EvaluatorTest, HostBGVTransformPlainNTT) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_transform_plain_ntt(ghe);
    }
    TEST(EvaluatorTest, HostCKKSTransformPlainNTT) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_transform_plain_ntt(ghe);
    }
    TEST(EvaluatorTest, DeviceBFVTransformPlainNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_transform_plain_ntt(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceBGVTransformPlainNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_transform_plain_ntt(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorTest, DeviceCKKSTransformPlainNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_transform_plain_ntt(ghe);
        utils::MemoryPool::Destroy();
    }
}