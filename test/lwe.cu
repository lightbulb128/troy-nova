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
        double scale = context.scale();
        double tolerance = context.tolerance();

        ASSERT_TRUE(context.params_host().poly_modulus_degree() == 32);
        size_t poly_modulus_degree = context.params_host().poly_modulus_degree();

        GaloisKeys automorphism_key = context.key_generator().create_automorphism_keys(false);

        auto test_setting = [&](size_t n) {
            auto message = context.batch_random_polynomial_full(n);
            auto encoded = context.encoder().batch_encode_polynomial(message, std::nullopt, scale);
            auto encrypted = context.batch_encrypt_asymmetric(encoded);
            GeneralVector truth = GeneralVector::zeros_like(message[0], message[0].size());
            size_t r = 1; while (r < n) r *= 2;
            size_t interval = poly_modulus_degree / r;
            for (size_t i = 0; i < n; i++) {
                truth.set(i * interval, message[i].element(0));
            }
            std::vector<LWECiphertext> extracted(n);
            for (size_t i = 0; i < n; i++) {
                extracted[i] = context.evaluator().extract_lwe_new(encrypted[i], 0);
            }

            Ciphertext assembled = context.evaluator().pack_lwe_ciphertexts_new(extracted, automorphism_key);
            Plaintext decrypted = context.decryptor().decrypt_new(assembled);
            GeneralVector decoded = context.encoder().decode_polynomial(decrypted);
            ASSERT_TRUE(truth.near_equal(decoded, tolerance));
        };

        test_setting(32);
        test_setting(7);
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

    void test_pack_lwes_batched(const GeneralHeContext& context) {
        double scale = context.scale();
        double tolerance = context.tolerance();

        ASSERT_TRUE(context.params_host().poly_modulus_degree() == 32);
        size_t poly_modulus_degree = context.params_host().poly_modulus_degree();

        GaloisKeys automorphism_key = context.key_generator().create_automorphism_keys(false);

        auto test_setting = [&](const std::vector<size_t>& ns) {
            size_t max_n = *std::max_element(ns.begin(), ns.end());
            size_t r = 1; while (r < max_n) r *= 2;
            size_t interval = poly_modulus_degree / r;
            std::vector<std::vector<GeneralVector>> message; message.reserve(ns.size());
            std::vector<std::vector<Plaintext>> encoded; encoded.reserve(ns.size());
            std::vector<std::vector<Ciphertext>> encrypted; encrypted.reserve(ns.size());
            std::vector<std::vector<LWECiphertext>> extracted; extracted.reserve(ns.size());
            std::vector<std::vector<const LWECiphertext*>> extracted_ptrs; extracted_ptrs.reserve(ns.size());
            std::vector<GeneralVector> truth; truth.reserve(ns.size());
            for (size_t i = 0; i < ns.size(); i++) {
                size_t n = ns[i];
                message.push_back(context.batch_random_polynomial_full(n));
                encoded.push_back(context.encoder().batch_encode_polynomial(message[i], std::nullopt, scale));
                encrypted.push_back(context.batch_encrypt_asymmetric(encoded[i]));
                GeneralVector t = GeneralVector::zeros_like(message[i][0], message[i][0].size());
                for (size_t j = 0; j < n; j++) {
                    t.set(j * interval, message[i][j].element(0));
                }
                truth.push_back(std::move(t));
                std::vector<LWECiphertext> e(n);
                for (size_t j = 0; j < n; j++) {
                    e[j] = context.evaluator().extract_lwe_new(encrypted[i][j], 0);
                }
                extracted.push_back(std::move(e));
                extracted_ptrs.push_back(batch_utils::collect_const_pointer(extracted[i]));
            }

            auto assembled = context.evaluator().pack_lwe_ciphertexts_new_batched(extracted_ptrs, automorphism_key);
            auto decrypted = context.batch_decrypt(assembled);
            auto decoded = context.encoder().batch_decode_polynomial(decrypted);
            for (size_t i = 0; i < ns.size(); i++) {
                ASSERT_TRUE(truth[i].near_equal(decoded[i], tolerance));
            }
        };

        test_setting({32, 20, 10, 5});
        test_setting({7, 3, 1});
    }

    TEST(LweTest, HostBFVPackLWEsBatched) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes_batched(ghe);
    }
    TEST(LweTest, HostBGVPackLWEsBatched) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes_batched(ghe);
    }
    TEST(LweTest, HostCKKSPackLWEsBatched) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_lwes_batched(ghe);
    }
    TEST(LweTest, DeviceBFVPackLWEsBatched) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes_batched(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceBGVPackLWEsBatched) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_lwes_batched(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceCKKSPackLWEsBatched) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_lwes_batched(ghe);
        utils::MemoryPool::Destroy();
    }



    void test_pack_rlwes(const GeneralHeContext& context) {
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


    void test_pack_rlwes_batched(const GeneralHeContext& context) {
        double scale = context.scale();
        double tolerance = context.tolerance();

        const size_t poly_modulus_degree = context.params_host().poly_modulus_degree();
        ASSERT_TRUE(context.params_host().poly_modulus_degree() == 32);

        GaloisKeys automorphism_key = context.key_generator().create_automorphism_keys(false);

        // pack 7 lwes with shift -3, input interval 8, output interval 1
        auto test_setting = [&](const std::vector<size_t> ns, size_t input_interval, size_t output_interval, int shift_){
            for (size_t n: ns) {
                ASSERT_TRUE(n <= input_interval / output_interval);
            }
            ASSERT_TRUE(shift_ <= 0 && static_cast<size_t>(-shift_) < input_interval);
            size_t shift = 2 * poly_modulus_degree + shift_;
            std::vector<std::vector<GeneralVector>> message; message.reserve(ns.size());
            std::vector<std::vector<Plaintext>> encoded; encoded.reserve(ns.size());
            std::vector<std::vector<Ciphertext>> encrypted; encrypted.reserve(ns.size());
            std::vector<std::vector<const Ciphertext*>> encrypted_ptrs; encrypted_ptrs.reserve(ns.size());
            std::vector<GeneralVector> truth; truth.reserve(ns.size());
            for (size_t i = 0; i < ns.size(); i++) {
                size_t n = ns[i];
                message.push_back(context.batch_random_polynomial_full(n));
                encoded.push_back(context.encoder().batch_encode_polynomial(message[i], std::nullopt, scale));
                encrypted.push_back(context.batch_encrypt_asymmetric(encoded[i]));
                GeneralVector t = GeneralVector::zeros_like(message[i][0], message[i][0].size());
                for (size_t j = 0; j < n; j++) {
                    for (size_t k = 0; k < poly_modulus_degree; k += input_interval) {
                        t.set(j * output_interval + k, message[i][j].element(k - shift_));
                    }
                }
                truth.push_back(std::move(t));
                encrypted_ptrs.push_back(batch_utils::collect_const_pointer(encrypted[i]));
            }
            auto assembled = context.evaluator().pack_rlwe_ciphertexts_new_batched(encrypted_ptrs, automorphism_key, shift, input_interval, output_interval); 
            auto decrypted = context.batch_decrypt(assembled);
            auto decoded = context.encoder().batch_decode_polynomial(decrypted);
            for (size_t i = 0; i < ns.size(); i++) {
                ASSERT_TRUE(truth[i].near_equal(decoded[i], tolerance));
            }
        };

        test_setting({32, 30, 17, 11}, 32, 1, 0);
        test_setting({5, 10, 16}, 16, 1, 0);
        test_setting({1, 2, 3, 4}, 8, 2, -3);
    }

    TEST(LweTest, HostBFVPackRLWEsBatched) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes_batched(ghe);
    }
    TEST(LweTest, HostBGVPackRLWEsBatched) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes_batched(ghe);
    }
    TEST(LweTest, HostCKKSPackRLWEsBatched) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_rlwes_batched(ghe);
    }
    TEST(LweTest, DeviceBFVPackRLWEsBatched) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes_batched(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceBGVPackRLWEsBatched) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_pack_rlwes_batched(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(LweTest, DeviceCKKSPackRLWEsBatched) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_pack_rlwes_batched(ghe);
        utils::MemoryPool::Destroy();
    }

}