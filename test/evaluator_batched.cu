#include "cuda_runtime.h"
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

    constexpr int batch_size = 16;

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

}