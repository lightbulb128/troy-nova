#include <gtest/gtest.h>
#include "test.h"
#include "test_adv.h"
#include "../src/key_generator.h"
#include "../src/encryptor.h"
#include "../src/decryptor.h"
#include "../src/evaluator.h"
#include "../src/batch_utils.h"

namespace evaluator_batched {

    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;

    constexpr int batch_size = 128;

    void test_negate(const GeneralHeContext& context) {
        double scale = context.scale();

        auto message = context.batch_random_simd_full(batch_size);
        auto encoded = context.encoder().batch_encode_simd(message, std::nullopt, scale);
        auto encrypted = context.batch_encrypt_asymmetric(encoded);

        auto enc_ptrs = batch_utils::collect_const_pointer(encrypted);

        { // new
            auto negated = context.evaluator().negate_new_batched(enc_ptrs);
            auto decrypted = context.batch_decrypt(negated);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_negate(message);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }

        { // assign
            auto destination = std::vector<Ciphertext>(batch_size);
            auto destination_ptrs = batch_utils::collect_pointer(destination);
            context.evaluator().negate_batched(enc_ptrs, destination_ptrs);
            auto decrypted = context.batch_decrypt(destination);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_negate(message);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }

        { // inplace
            auto destination = batch_utils::clone(encrypted);
            auto destination_ptrs = batch_utils::collect_pointer(destination);
            context.evaluator().negate_inplace_batched(destination_ptrs);
            auto decrypted = context.batch_decrypt(destination);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_negate(message);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }
    }

    TEST(EvaluatorBatchTest, HostBFVNegate) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVNegate) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSNegate) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_negate(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVNegate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVNegate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_negate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSNegate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_negate(ghe);
        utils::MemoryPool::Destroy();
    }


    void test_add_subtract(const GeneralHeContext& context) {
        double scale = context.scale();

        auto message1 = context.batch_random_simd_full(batch_size);
        auto message2 = context.batch_random_simd_full(batch_size);
        auto encoded1 = context.encoder().batch_encode_simd(message1, std::nullopt, scale);
        auto encoded2 = context.encoder().batch_encode_simd(message2, std::nullopt, scale);
        auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);
        auto encrypted2 = context.batch_encrypt_asymmetric(encoded2);

        auto enc1_ptrs = batch_utils::collect_const_pointer(encrypted1);
        auto enc2_ptrs = batch_utils::collect_const_pointer(encrypted2);

        { // new
            auto added = context.evaluator().add_new_batched(enc1_ptrs, enc2_ptrs);
            auto decrypted = context.batch_decrypt(added);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_add(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));

            auto subtracted = context.evaluator().sub_new_batched(enc1_ptrs, enc2_ptrs);
            decrypted = context.batch_decrypt(subtracted);
            result = context.encoder().batch_decode_simd(decrypted);
            truth = context.batch_sub(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }

        { // assign
            auto destination = std::vector<Ciphertext>(batch_size);
            auto destination_ptrs = batch_utils::collect_pointer(destination);
            context.evaluator().add_batched(enc1_ptrs, enc2_ptrs, destination_ptrs);
            auto decrypted = context.batch_decrypt(destination);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_add(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));

            context.evaluator().sub_batched(enc1_ptrs, enc2_ptrs, destination_ptrs);
            decrypted = context.batch_decrypt(destination);
            result = context.encoder().batch_decode_simd(decrypted);
            truth = context.batch_sub(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }

        { // inplace
            auto destination = batch_utils::clone(encrypted1);
            auto destination_ptrs = batch_utils::collect_pointer(destination);
            context.evaluator().add_inplace_batched(destination_ptrs, enc2_ptrs);
            auto decrypted = context.batch_decrypt(destination);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_add(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));

            destination = batch_utils::clone(encrypted1);
            destination_ptrs = batch_utils::collect_pointer(destination);
            context.evaluator().sub_inplace_batched(destination_ptrs, enc2_ptrs);
            decrypted = context.batch_decrypt(destination);
            truth = context.batch_sub(message1, message2);
            result = context.encoder().batch_decode_simd(decrypted);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }
    }

    TEST(EvaluatorBatchTest, HostBFVAdd) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVAdd) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSAdd) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVAdd) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVAdd) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_add_subtract(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSAdd) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 40, 40, 40 }, false, 0x123, 10, 1<<20, 1e-2);
        test_add_subtract(ghe);
        utils::MemoryPool::Destroy();
    }



    void test_multiply_plain(const GeneralHeContext& context) {
        double scale = context.scale();

        // native multiply
        {
            auto message1 = context.batch_random_simd_full(batch_size);
            auto message2 = context.batch_random_simd_full(batch_size);

            auto encoded1 = context.encoder().batch_encode_simd(message1, std::nullopt, scale);
            auto encoded2 = context.encoder().batch_encode_simd(message2, std::nullopt, scale);
            auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);

            auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
            auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
            auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
            auto p2_ptrs = batch_utils::collect_pointer(encoded2);

            auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
            auto decrypted = context.batch_decrypt(multiplied);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_mul(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));

            // if encrypted is not ntt
            if (!encrypted1[0].is_ntt_form()) {
                context.evaluator().transform_to_ntt_inplace_batched(e1_ptrs);
                context.evaluator().transform_plain_to_ntt_inplace_batched(p2_ptrs, encrypted1[0].parms_id());
                auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
                auto multiplied_ptrs = batch_utils::collect_pointer(multiplied);
                context.evaluator().transform_from_ntt_inplace_batched(multiplied_ptrs);
                auto decrypted = context.batch_decrypt(multiplied);
                auto result = context.encoder().batch_decode_simd(decrypted);
                ASSERT_TRUE(context.batch_near_equal(truth, result));
            } else {
                context.evaluator().transform_from_ntt_inplace_batched(e1_ptrs);
                context.evaluator().transform_to_ntt_inplace_batched(e1_ptrs);
                auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
                auto decrypted = context.batch_decrypt(multiplied);
                auto result = context.encoder().batch_decode_simd(decrypted);
                ASSERT_TRUE(context.batch_near_equal(truth, result));
            }
        }

        {
            size_t n = context.params_host().poly_modulus_degree();
            size_t cc = n / 3;
            auto message1 = context.batch_random_polynomial(batch_size, cc);
            auto message2 = context.batch_random_polynomial(batch_size, cc);
            auto encoded1 = context.encoder().batch_encode_polynomial(message1, std::nullopt, scale);
            auto encoded2 = context.encoder().batch_encode_polynomial(message2, std::nullopt, scale);
            auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);
            
            auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
            auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
            auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
            auto p2_ptrs = batch_utils::collect_pointer(encoded2);

            auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
            auto decrypted = context.batch_decrypt(multiplied);
            auto result = context.encoder().batch_decode_polynomial(decrypted);
            auto truth = context.batch_mul_poly(message1, message2); 
            for (auto& item: truth) item.resize(n);
            ASSERT_TRUE(context.batch_near_equal(truth, result));

            // if encrypted is not ntt
            if (!encrypted1[0].is_ntt_form()) {
                context.evaluator().transform_to_ntt_inplace_batched(e1_ptrs);
                context.evaluator().transform_plain_to_ntt_inplace_batched(p2_ptrs, encrypted1[0].parms_id());
                auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
                auto multiplied_ptrs = batch_utils::collect_pointer(multiplied);
                context.evaluator().transform_from_ntt_inplace_batched(multiplied_ptrs);
                auto decrypted = context.batch_decrypt(multiplied);
                auto result = context.encoder().batch_decode_polynomial(decrypted);
                ASSERT_TRUE(context.batch_near_equal(truth, result));
            } else {
                context.evaluator().transform_from_ntt_inplace_batched(e1_ptrs);
                context.evaluator().transform_to_ntt_inplace_batched(e1_ptrs);
                auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
                auto decrypted = context.batch_decrypt(multiplied);
                auto result = context.encoder().batch_decode_polynomial(decrypted);
                ASSERT_TRUE(context.batch_near_equal(truth, result));
            }
        }
    }

    TEST(EvaluatorBatchTest, HostBFVMultiplyPlain) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVMultiplyPlain) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSMultiplyPlain) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply_plain(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVMultiplyPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVMultiplyPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSMultiplyPlain) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply_plain(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_multiply_plain_ntt(const GeneralHeContext& context) {
        double scale = context.scale();

        // native multiply
        {
            auto message1 = context.batch_random_simd_full(batch_size);
            auto message2 = context.batch_random_simd_full(batch_size);

            auto encoded1 = context.encoder().batch_encode_simd(message1, std::nullopt, scale);
            auto encoded2 = context.encoder().batch_encode_simd(message2, std::nullopt, scale);
            auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);

            auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
            auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
            auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
            auto p2_ptrs = batch_utils::collect_pointer(encoded2);

            context.evaluator().transform_plain_to_ntt_inplace_batched(p2_ptrs, encrypted1[0].parms_id());
            auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
            auto decrypted = context.batch_decrypt(multiplied);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_mul(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }

        {
            size_t n = context.params_host().poly_modulus_degree();
            size_t cc = n / 3;


            auto message1 = context.batch_random_polynomial(batch_size, cc);
            auto message2 = context.batch_random_polynomial(batch_size, cc);
            auto encoded1 = context.encoder().batch_encode_polynomial(message1, std::nullopt, scale);
            for (auto& item: encoded1) {
                ASSERT_EQ(item.coeff_count(), cc);
            }
            auto encoded2 = context.encoder().batch_encode_polynomial(message2, std::nullopt, scale);
            for (auto& item: encoded2) {
                ASSERT_EQ(item.coeff_count(), cc);
            }
            auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);
            
            auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
            auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
            auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
            auto p2_ptrs = batch_utils::collect_pointer(encoded2);

            context.evaluator().transform_plain_to_ntt_inplace_batched(p2_ptrs, encrypted1[0].parms_id());
            for (auto& item: p2_cptrs) {
                ASSERT_EQ(item->coeff_count(), n);
                ASSERT_EQ(item->poly_modulus_degree(), n);
                ASSERT_EQ(item->data().size(), item->coeff_modulus_size() * n);
            }
            auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
            auto decrypted = context.batch_decrypt(multiplied);
            auto result = context.encoder().batch_decode_polynomial(decrypted);
            auto truth = context.batch_mul_poly(message1, message2); 
            for (auto& item: truth) item.resize(n);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }
    }

    TEST(EvaluatorBatchTest, HostBFVMultiplyPlainNTT) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVMultiplyPlainNTT) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVMultiplyPlainNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVMultiplyPlainNTT) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_ntt(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_multiply_plain_centralized(const GeneralHeContext& context) {
        double scale = context.scale();

        {
            auto message1 = context.batch_random_simd_full(batch_size);
            auto message2 = context.batch_random_simd_full(batch_size);

            auto encoded1 = context.encoder().batch_encode_simd(message1, std::nullopt, scale);
            auto encoded2 = context.encoder().batch_encode_simd(message2, std::nullopt, scale);
            auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);

            auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
            auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
            auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
            auto p2_ptrs = batch_utils::collect_pointer(encoded2);

            for (auto item: p2_ptrs) {
                context.encoder().batch().centralize_inplace(*item, std::nullopt);
            }
            auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
            auto decrypted = context.batch_decrypt(multiplied);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_mul(message1, message2);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }

        {
            size_t n = context.encoder().slot_count();
            size_t cc = n / 3;
            {
                auto message1 = context.batch_random_polynomial(batch_size, n);
                auto message2 = context.batch_random_polynomial(batch_size, n);
                auto encoded1 = context.encoder().batch_encode_polynomial(message1, std::nullopt, scale);
                for (auto& item: encoded1) {
                    ASSERT_EQ(item.coeff_count(), n);
                }
                auto encoded2 = context.encoder().batch_encode_polynomial(message2, std::nullopt, scale);
                for (auto& item: encoded2) {
                    ASSERT_EQ(item.coeff_count(), n);
                }
                auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);
                
                auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
                auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
                auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
                auto p2_ptrs = batch_utils::collect_pointer(encoded2);

                for (auto item: p2_ptrs) {
                    context.encoder().batch().centralize_inplace(*item, std::nullopt);
                }
                for (auto& item: p2_cptrs) {
                    ASSERT_EQ(item->coeff_count(), n);
                    ASSERT_EQ(item->poly_modulus_degree(), n);
                    ASSERT_EQ(item->data().size(), item->coeff_modulus_size() * n);
                }
                auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
                auto decrypted = context.batch_decrypt(multiplied);
                auto result = context.encoder().batch_decode_polynomial(decrypted);
                auto truth = context.batch_mul_poly(message1, message2); 
                for (auto& item: truth) item.resize(n);
                ASSERT_TRUE(context.batch_near_equal(truth, result));
            }
            {
                auto message1 = context.batch_random_polynomial(batch_size, cc);
                auto message2 = context.batch_random_polynomial(batch_size, cc);
                auto encoded1 = context.encoder().batch_encode_polynomial(message1, std::nullopt, scale);
                for (auto& item: encoded1) {
                    ASSERT_EQ(item.coeff_count(), cc);
                }
                auto encoded2 = context.encoder().batch_encode_polynomial(message2, std::nullopt, scale);
                for (auto& item: encoded2) {
                    ASSERT_EQ(item.coeff_count(), cc);
                }
                auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);
                
                auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
                auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
                auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
                auto p2_ptrs = batch_utils::collect_pointer(encoded2);

                for (auto item: p2_ptrs) {
                    context.encoder().batch().centralize_inplace(*item, std::nullopt);
                }
                for (auto& item: p2_cptrs) {
                    ASSERT_EQ(item->coeff_count(), cc);
                    ASSERT_EQ(item->poly_modulus_degree(), n);
                    ASSERT_EQ(item->data().size(), item->coeff_modulus_size() * cc);
                }
                auto multiplied = context.evaluator().multiply_plain_new_batched(e1_cptrs, p2_cptrs);
                auto decrypted = context.batch_decrypt(multiplied);
                auto result = context.encoder().batch_decode_polynomial(decrypted);
                auto truth = context.batch_mul_poly(message1, message2); 
                for (auto& item: truth) item.resize(n);
                ASSERT_TRUE(context.batch_near_equal(truth, result));
            }
        }
    }

    TEST(EvaluatorBatchTest, HostBFVMultiplyPlainCentralized) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_centralized(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVMultiplyPlainCentralized) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_centralized(ghe);
        utils::MemoryPool::Destroy();
    }



    void test_multiply_plain_accumulate(const GeneralHeContext& context) {
        double scale = context.scale();
        constexpr size_t block = 8;
        static_assert(batch_size % block == 0);

        auto message1 = context.batch_random_simd_full(batch_size);
        auto message2 = context.batch_random_simd_full(batch_size);

        auto encoded1 = context.encoder().batch_encode_simd(message1, std::nullopt, scale);
        auto encoded2 = context.encoder().batch_encode_simd(message2, std::nullopt, scale);
        auto encrypted1 = context.batch_encrypt_asymmetric(encoded1);

        auto e1_cptrs = batch_utils::collect_const_pointer(encrypted1);
        auto e1_ptrs = batch_utils::collect_pointer(encrypted1);
        auto p2_cptrs = batch_utils::collect_const_pointer(encoded2);
        auto p2_ptrs = batch_utils::collect_pointer(encoded2);

        std::vector<Ciphertext> destination(batch_size / block);
        std::vector<Ciphertext*> destination_ptrs(batch_size);
        for (size_t i = 0; i < destination_ptrs.size(); i++) {
            destination_ptrs[i] = &destination[i % (batch_size / block)];
        }

        context.evaluator().multiply_plain_accumulate(e1_cptrs, p2_cptrs, destination_ptrs);

        auto decrypted = context.batch_decrypt(destination);
        auto result = context.encoder().batch_decode_simd(decrypted);

        std::vector<GeneralVector> truth(batch_size / block);
        for (size_t i = 0; i < batch_size / block; i++) {
            for (size_t j = 0; j < block; j++) {
                auto mul = context.mul(message1[j * (batch_size / block) + i], message2[j * (batch_size / block) + i]);
                if (j == 0) {
                    truth[i] = mul;
                } else {
                    truth[i] = context.add(truth[i], mul);
                }
            }
        }
        ASSERT_TRUE(context.batch_near_equal(truth, result));

    }

    
    TEST(EvaluatorBatchTest, HostBFVMultiplyPlainAccumulate) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_accumulate(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVMultiplyPlainAccumulate) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_accumulate(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSMultiplyPlainAccumulate) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply_plain_accumulate(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVMultiplyPlainAccumulate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_accumulate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVMultiplyPlainAccumulate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 40, 40, 40 }, false, 0x123, 0);
        test_multiply_plain_accumulate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSMultiplyPlainAccumulate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 60, 60 }, false, 0x123, 10, 1ull<<20, 1e-2);
        test_multiply_plain_accumulate(ghe);
        utils::MemoryPool::Destroy();
    }

    
    void test_keyswitching(const GeneralHeContext& context) {
        // uint64_t t = context.t();
        double scale = context.scale();

        // create another keygenerator
        KeyGenerator keygen_other = KeyGenerator(context.context());
        SecretKey secret_key_other = keygen_other.secret_key();
        Encryptor encryptor_other = Encryptor(context.context());
        encryptor_other.set_secret_key(secret_key_other);

        KSwitchKeys kswitch_key = context.key_generator().create_keyswitching_key(secret_key_other, false);

        auto message = context.batch_random_simd_full(batch_size);
        auto encoded = context.encoder().batch_encode_simd(message, std::nullopt, scale);
        std::vector<Ciphertext> encrypted(batch_size);
        for (size_t i = 0; i < batch_size; i++) encrypted[i] = encryptor_other.encrypt_symmetric_new(encoded[i], false);
        auto switched = context.evaluator().apply_keyswitching_new_batched(batch_utils::collect_const_pointer(encrypted), kswitch_key);
        auto decrypted = context.batch_decrypt(switched);
        auto result = context.encoder().batch_decode_simd(decrypted);
        ASSERT_TRUE(context.batch_near_equal(message, result));
    }

    TEST(EvaluatorBatchTest, HostBFVKeySwitching) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVKeySwitching) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSKeySwitching) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_keyswitching(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVKeySwitching) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVKeySwitching) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_keyswitching(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSKeySwitching) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_keyswitching(ghe);
        utils::MemoryPool::Destroy();
    }



    void test_rotate(const GeneralHeContext& context) {
        double scale = context.scale();

        auto message = context.batch_random_simd_full(batch_size);
        auto encoded = context.encoder().batch_encode_simd(message, std::nullopt, scale);
        auto encrypted = context.batch_encrypt_asymmetric(encoded);
        auto encrypted_ptrs = batch_utils::collect_const_pointer(encrypted);
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        std::vector<Ciphertext> rotated;
        for (int step: {1, 7}) {
            if (context.params_host().scheme() == SchemeType::CKKS) {
                rotated = context.evaluator().rotate_vector_new_batched(encrypted_ptrs, step, glk);
            } else {
                rotated = context.evaluator().rotate_rows_new_batched(encrypted_ptrs, step, glk);
            }
            auto decrypted = context.batch_decrypt(rotated);
            auto result = context.encoder().batch_decode_simd(decrypted);
            auto truth = context.batch_rotate(message, step);
            ASSERT_TRUE(context.batch_near_equal(truth, result));
        }
        
    }

    TEST(EvaluatorBatchTest, HostBFVRotateRows) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVRotateRows) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSRotateVector) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_rotate(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVRotateRows) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVRotateRows) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_rotate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSRotateVector) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_rotate(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_conjugate(const GeneralHeContext& context) {
        double scale = context.scale();

        auto message = context.batch_random_simd_full(batch_size);
        auto encoded = context.encoder().batch_encode_simd(message, std::nullopt, scale);
        auto encrypted = context.batch_encrypt_asymmetric(encoded);
        auto encrypted_ptrs = batch_utils::collect_const_pointer(encrypted);
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        std::vector<Ciphertext> rotated;
        if (context.params_host().scheme() == SchemeType::CKKS) {
            rotated = context.evaluator().complex_conjugate_new_batched(encrypted_ptrs, glk);
        } else {
            rotated = context.evaluator().rotate_columns_new_batched(encrypted_ptrs, glk);
        }
        auto decrypted = context.batch_decrypt(rotated);
        auto result = context.encoder().batch_decode_simd(decrypted);
        auto truth = context.batch_conjugate(message);
        ASSERT_TRUE(context.batch_near_equal(truth, result));
    }

    TEST(EvaluatorBatchTest, HostBFVRotateColumns) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVRotateColumns) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSComplexConjugate) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_conjugate(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVRotateColumns) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVRotateColumns) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_conjugate(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSComplexConjugate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_conjugate(ghe);
        utils::MemoryPool::Destroy();
    }

    void test_negacyclic_shift(const GeneralHeContext& context) {
        double scale = context.scale();

        auto message = context.batch_random_simd_full(batch_size);
        auto encoded = context.encoder().batch_encode_simd(message, std::nullopt, scale);
        auto encrypted = context.batch_encrypt_asymmetric(encoded);
        auto encrypted_ptrs = batch_utils::collect_const_pointer(encrypted);
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        std::vector<Ciphertext> rotated;
        if (context.params_host().scheme() == SchemeType::CKKS) {
            rotated = context.evaluator().complex_conjugate_new_batched(encrypted_ptrs, glk);
        } else {
            rotated = context.evaluator().rotate_columns_new_batched(encrypted_ptrs, glk);
        }
        auto decrypted = context.batch_decrypt(rotated);
        auto result = context.encoder().batch_decode_simd(decrypted);
        auto truth = context.batch_conjugate(message);
        ASSERT_TRUE(context.batch_near_equal(truth, result));
    }

    void test_mod_switch_to_next(const GeneralHeContext& context) {
        double scale = context.scale();

        auto message = context.batch_random_simd_full(batch_size);
        auto encoded = context.encoder().batch_encode_simd(message, std::nullopt, scale);
        auto encrypted = context.batch_encrypt_asymmetric(encoded);
        auto encrypted_ptrs = batch_utils::collect_const_pointer(encrypted);
        auto switched = context.evaluator().mod_switch_to_next_new_batched(encrypted_ptrs);
        auto decrypted = context.batch_decrypt(switched);
        auto result = context.encoder().batch_decode_simd(decrypted);
        // result should be same with message
        ASSERT_TRUE(context.batch_near_equal(message, result));

        switched = context.evaluator().mod_switch_to_new_batched(encrypted_ptrs, context.context()->last_parms_id());
        decrypted = context.batch_decrypt(switched);
        result = context.encoder().batch_decode_simd(decrypted);
        // result should be same with message
        ASSERT_TRUE(context.batch_near_equal(message, result));
    }

    TEST(EvaluatorBatchTest, HostBFVModSwitchToNext) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
    }
    TEST(EvaluatorBatchTest, HostBGVModSwitchToNext) {
        GeneralHeContext ghe(false, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
    }
    TEST(EvaluatorBatchTest, HostCKKSModSwitchToNext) {
        GeneralHeContext ghe(false, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_mod_switch_to_next(ghe);
    }
    TEST(EvaluatorBatchTest, DeviceBFVModSwitchToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceBGVModSwitchToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BGV, 32, 20, { 60, 40, 40, 60 }, true, 0x123, 0);
        test_mod_switch_to_next(ghe);
        utils::MemoryPool::Destroy();
    }
    TEST(EvaluatorBatchTest, DeviceCKKSModSwitchToNext) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::CKKS, 32, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_mod_switch_to_next(ghe);
        utils::MemoryPool::Destroy();
    }

}