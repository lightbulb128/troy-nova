#include <gtest/gtest.h>
#include "test_adv.h"
#include "test.h"
#include <sstream>

namespace serialize_zstd {

    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;
    using std::stringstream;

    template <typename T>
    void reserialize(T& t) {
        stringstream ss;
        t.save(ss, CompressionMode::Zstd);
        ASSERT_TRUE(t.serialized_size_upperbound(CompressionMode::Zstd) >= ss.str().size());
        t = T::load_new(ss);
    }

    template <typename T>
    void reserialize(T& t, HeContextPointer context) {
        stringstream ss;
        t.save(ss, context, CompressionMode::Zstd);
        ASSERT_TRUE(t.serialized_size_upperbound(context, CompressionMode::Zstd) >= ss.str().size());
        t = T::load_new(ss, context);
    }

    void test_plaintext(const GeneralHeContext& context) {
        double scale = context.scale();
        double tolerance = context.tolerance();
        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
        reserialize(encoded);
        GeneralVector decoded = context.encoder().decode_simd(encoded);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));
    }

    TEST(SerializeZstdTest, HostBFVPlaintext) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_plaintext(ghe);
    }
    TEST(SerializeZstdTest, HostBGVPlaintext) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_plaintext(ghe);
    }
    TEST(SerializeZstdTest, HostCKKSPlaintext) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_plaintext(ghe);
    }
    TEST(SerializeZstdTest, DeviceBFVPlaintext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_plaintext(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceBGVPlaintext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_plaintext(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceCKKSPlaintext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_plaintext(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_ciphertext(const GeneralHeContext& context) {
        double scale = context.scale();
        double tolerance = context.tolerance();
        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);

        // test expand seed
        Ciphertext encrypted = context.encryptor().encrypt_symmetric_new(encoded, true);
        Ciphertext cloned = encrypted.clone();
        cloned.expand_seed(context.context());
        Plaintext decrypted = context.decryptor().decrypt_new(cloned);
        GeneralVector decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));

        // test save and load asymmetric encrypted
        encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        reserialize(encrypted, context.context());
        decrypted = context.decryptor().decrypt_new(encrypted);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));

        // test save and load symmetric encrypted
        encrypted = context.encryptor().encrypt_symmetric_new(encoded, false);
        reserialize(encrypted, context.context());
        decrypted = context.decryptor().decrypt_new(encrypted);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));
        encrypted = context.encryptor().encrypt_symmetric_new(encoded, true);
        reserialize(encrypted, context.context());
        decrypted = context.decryptor().decrypt_new(encrypted);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));
    }

    TEST(SerializeZstdTest, HostBFVCiphertext) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext(ghe);
    }
    TEST(SerializeZstdTest, HostBGVCiphertext) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext(ghe);
    }
    TEST(SerializeZstdTest, HostCKKSCiphertext) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_ciphertext(ghe);
    }
    TEST(SerializeZstdTest, DeviceBFVCiphertext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceBGVCiphertext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceCKKSCiphertext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_ciphertext(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_secret_public_key(const GeneralHeContext& context) {
        double scale = context.scale();
        double tolerance = context.tolerance();
        GeneralVector message = context.random_simd_full();
        Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);

        // reserialize secret key
        HeContextPointer he = context.context();
        SecretKey secret_key = context.key_generator().secret_key();
        reserialize(secret_key);
        Encryptor encryptor = Encryptor(he);
        encryptor.set_secret_key(secret_key);
        // cipher encrypted by encryptor could be decrypted with context.decryptor()
        Ciphertext encrypted = encryptor.encrypt_symmetric_new(encoded, false);
        Plaintext decrypted = context.decryptor().decrypt_new(encrypted);
        GeneralVector decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));

        // reserialize public key, don't save seed
        PublicKey public_key = context.key_generator().create_public_key(false);
        reserialize(public_key, he);
        encryptor.set_public_key(public_key);
        encrypted = encryptor.encrypt_asymmetric_new(encoded);
        decrypted = context.decryptor().decrypt_new(encrypted);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));

        // reserialize public key, save seed
        public_key = context.key_generator().create_public_key(true);
        reserialize(public_key, he);
        encrypted = encryptor.encrypt_asymmetric_new(encoded);
        decrypted = context.decryptor().decrypt_new(encrypted);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(message.near_equal(decoded, tolerance));
    }

    TEST(SerializeZstdTest, HostBFVSecretPublicKey) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_secret_public_key(ghe);
    }
    TEST(SerializeZstdTest, HostBGVSecretPublicKey) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_secret_public_key(ghe);
    }
    TEST(SerializeZstdTest, HostCKKSCSecretPublicKey) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_secret_public_key(ghe);
    }
    TEST(SerializeZstdTest, DeviceBFVSecretPublicKey) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_secret_public_key(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceBGVSecretPublicKey) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_secret_public_key(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceCKKSSecretPublicKey) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_secret_public_key(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_kswitch_keys(const GeneralHeContext& context) {
        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        { // key switching
            // create another keygenerator
            KeyGenerator keygen_other = KeyGenerator(context.context());
            SecretKey secret_key_other = keygen_other.secret_key();
            Encryptor encryptor_other = Encryptor(context.context());
            encryptor_other.set_secret_key(secret_key_other);

            KSwitchKeys kswitch_key = context.key_generator().create_keyswitching_key(secret_key_other, false);
            reserialize(kswitch_key, context.context());

            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
            Ciphertext encrypted = encryptor_other.encrypt_symmetric_new(encoded, false);
            Ciphertext switched = context.evaluator().apply_keyswitching_new(encrypted, kswitch_key);
            Plaintext decrypted = context.decryptor().decrypt_new(switched);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message;
            ASSERT_TRUE(truth.near_equal(result, tolerance));

            kswitch_key = context.key_generator().create_keyswitching_key(secret_key_other, true);
            reserialize(kswitch_key, context.context());

            message = context.random_simd_full();
            encoded = context.encoder().encode_simd(message, std::nullopt, scale);
            encrypted = encryptor_other.encrypt_symmetric_new(encoded, false);
            switched = context.evaluator().apply_keyswitching_new(encrypted, kswitch_key);
            decrypted = context.decryptor().decrypt_new(switched);
            result = context.encoder().decode_simd(decrypted);
            truth = message;
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
        { // relin keys
            GeneralVector message1 = context.random_simd_full();
            GeneralVector message2 = context.random_simd_full();
            RelinKeys relin_keys = context.key_generator().create_relin_keys(false);
            reserialize(relin_keys, context.context());
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
            
            message1 = context.random_simd_full();
            message2 = context.random_simd_full();
            relin_keys = context.key_generator().create_relin_keys(true);
            reserialize(relin_keys, context.context());
            encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
            encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
            encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
            encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
            multiplied = context.evaluator().multiply_new(encrypted1, encrypted2);
            relined = context.evaluator().relinearize_new(multiplied, relin_keys);
            decrypted = context.decryptor().decrypt_new(relined);
            result = context.encoder().decode_simd(decrypted);
            truth = message1.mul(message2, t);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
        { // rotate
            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale);
            Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
            GaloisKeys glk = context.key_generator().create_galois_keys(false);
            reserialize(glk, context.context());
            Ciphertext rotated;
            if (context.params_host().scheme() == SchemeType::CKKS) {
                rotated = context.evaluator().rotate_vector_new(encrypted, 1, glk);
            } else {
                rotated = context.evaluator().rotate_rows_new(encrypted, 1, glk);
            }
            Plaintext decrypted = context.decryptor().decrypt_new(rotated);
            GeneralVector result = context.encoder().decode_simd(decrypted);
            GeneralVector truth = message.rotate(1);
            ASSERT_TRUE(truth.near_equal(result, tolerance));

            message = context.random_simd_full();
            encoded = context.encoder().encode_simd(message, std::nullopt, scale);
            encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
            glk = context.key_generator().create_galois_keys(true);
            reserialize(glk, context.context());
            if (context.params_host().scheme() == SchemeType::CKKS) {
                rotated = context.evaluator().rotate_vector_new(encrypted, 1, glk);
            } else {
                rotated = context.evaluator().rotate_rows_new(encrypted, 1, glk);
            }
            decrypted = context.decryptor().decrypt_new(rotated);
            result = context.encoder().decode_simd(decrypted);
            truth = message.rotate(1);
            ASSERT_TRUE(truth.near_equal(result, tolerance));
        }
    }

    TEST(SerializeZstdTest, HostBFVKSwitchKeys) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_kswitch_keys(ghe);
    }
    TEST(SerializeZstdTest, HostBGVKSwitchKeys) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_kswitch_keys(ghe);
    }
    TEST(SerializeZstdTest, HostCKKSKSwitchKeys) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_kswitch_keys(ghe);
    }
    TEST(SerializeZstdTest, DeviceBFVKSwitchKeys) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_kswitch_keys(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceBGVKSwitchKeys) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_kswitch_keys(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceCKKSKSwitchKeys) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_kswitch_keys(ghe);
        utils::MemoryPool::Destroy();
    }
    
    void reserialize_terms(Ciphertext& t, HeContextPointer context, const std::vector<size_t>& terms) {
        stringstream ss;
        t.save_terms(ss, context, terms, MemoryPool::GlobalPool(), CompressionMode::Zstd);
        ASSERT_TRUE(t.serialized_terms_size_upperbound(context, terms, CompressionMode::Zstd) >= ss.str().size());
        t = Ciphertext::load_terms_new(ss, context, terms);
    }

    void test_ciphertext_terms(const GeneralHeContext& context) {
        double scale = context.scale();
        double tolerance = context.tolerance();

        std::vector<size_t> terms = { 1, 3, 5, 7 };

        // without seed
        GeneralVector message = context.random_polynomial_full();
        Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);
        Ciphertext encrypted = context.encryptor().encrypt_symmetric_new(encoded, false);
        reserialize_terms(encrypted, context.context(), terms);
        Plaintext decrypted = context.decryptor().decrypt_new(encrypted);
        GeneralVector decoded = context.encoder().decode_polynomial(decrypted);
        for (size_t term: terms) {
            ASSERT_TRUE(message.element(term).near_equal(decoded.element(term), tolerance));
        }

        // with seed
        message = context.random_polynomial_full();
        encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);
        encrypted = context.encryptor().encrypt_symmetric_new(encoded, true);
        reserialize_terms(encrypted, context.context(), terms);
        decrypted = context.decryptor().decrypt_new(encrypted);
        decoded = context.encoder().decode_polynomial(decrypted);
        for (size_t term: terms) {
            ASSERT_TRUE(message.element(term).near_equal(decoded.element(term), tolerance));
        }

        // one multiplication
        message = context.random_polynomial_full();
        encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);
        encrypted = context.encryptor().encrypt_asymmetric_new(encoded);
        Ciphertext multiplied = context.evaluator().multiply_new(encrypted, encrypted);
        reserialize_terms(multiplied, context.context(), terms);
        decrypted = context.decryptor().decrypt_new(multiplied);
        decoded = context.encoder().decode_polynomial(decrypted);
        
        Ciphertext multiplied_truth = context.evaluator().multiply_new(encrypted, encrypted);
        decrypted = context.decryptor().decrypt_new(multiplied_truth);
        GeneralVector truth = context.encoder().decode_polynomial(decrypted);
        for (size_t term: terms) {
            ASSERT_TRUE(truth.element(term).near_equal(decoded.element(term), tolerance));
        }

    }

    TEST(SerializeZstdTest, HostBFVCiphertextTerms) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext_terms(ghe);
    }
    TEST(SerializeZstdTest, HostBGVCiphertextTerms) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext_terms(ghe);
    }
    TEST(SerializeZstdTest, HostCKKSCiphertextTerms) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_ciphertext_terms(ghe);
    }
    TEST(SerializeZstdTest, DeviceBFVCiphertextTerms) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext_terms(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceBGVCiphertextTerms) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_ciphertext_terms(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(SerializeZstdTest, DeviceCKKSCiphertextTerms) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1<<16, 1e-2);
        test_ciphertext_terms(ghe);
        utils::MemoryPool::Destroy();
    }
}