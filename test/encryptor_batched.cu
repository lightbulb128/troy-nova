#include "cuda_runtime.h"
#include <gtest/gtest.h>
#include <optional>
#include "test.h"
#include "test_adv.h"
#include "../src/batch_encoder.h"
#include "../src/ckks_encoder.h"
#include "../src/encryptor.h"
#include "../src/decryptor.h"
#include "../src/evaluator.h"

namespace encryptor_batched {

    using namespace troy;
    using std::vector;
    using troy::utils::ConstSlice;
    using troy::utils::Slice;
    using std::optional;
    using std::complex;
    using tool::GeneralEncoder;
    using tool::GeneralHeContext;

    template<typename T>
    ConstSlice<T> sfv(const vector<T> &vec) {
        return ConstSlice<T>(vec.data(), vec.size(), false, nullptr);
    }

    bool all_parms_id(const std::vector<Ciphertext>& c, ParmsID p) {
        for (const auto& cipher : c) {
            if (cipher.parms_id() != p) {
                return false;
            }
        }
        return true;
    } 

    void test_suite(const GeneralHeContext& context) {
        
        const Encryptor& encryptor = context.encryptor();
        const Decryptor& decryptor = context.decryptor();
        const GeneralEncoder& encoder = context.encoder();
        auto scheme = context.scheme();
        auto first_parms_id = context.context()->first_parms_id();

        constexpr size_t batch_size = 16;

        // encrypt zero, symmetric
        auto cipher = encryptor.encrypt_zero_symmetric_new_batched(batch_size, false);
        if (scheme == SchemeType::CKKS) for (size_t i = 0; i < batch_size; i++) cipher[i].scale() = 1e6;
        ASSERT_TRUE(all_parms_id(cipher, first_parms_id));
        auto decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
        auto decoded = context.encoder().batch_decode_simd(decrypted);
        auto zeros = context.batch_zeros_simd(batch_size);
        ASSERT_TRUE(context.batch_near_equal(decoded, zeros));

        // encrypt zero, asymmetric
        cipher = encryptor.encrypt_zero_asymmetric_new_batched(batch_size);
        if (scheme == SchemeType::CKKS) for (size_t i = 0; i < batch_size; i++) cipher[i].scale() = 1e6;
        ASSERT_TRUE(all_parms_id(cipher, first_parms_id));
        decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
        decoded = context.encoder().batch_decode_simd(decrypted);
        ASSERT_TRUE(context.batch_near_equal(decoded, zeros));

        auto he = context.context();
        if (he->first_context_data().value()->next_context_data()) {
            ParmsID second_parms_id = he->first_context_data().value()->next_context_data().value()->parms_id();
            // asymmetric
            cipher = encryptor.encrypt_zero_asymmetric_new_batched(batch_size, second_parms_id);
            if (scheme == SchemeType::CKKS) for (size_t i = 0; i < batch_size; i++) cipher[i].scale() = 1e6;
            ASSERT_TRUE(all_parms_id(cipher, second_parms_id));
            decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
            decoded = context.encoder().batch_decode_simd(decrypted);
            ASSERT_TRUE(context.batch_near_equal(decoded, zeros));
            // symmetric
            cipher = encryptor.encrypt_zero_asymmetric_new_batched(batch_size, second_parms_id);
            if (scheme == SchemeType::CKKS) for (size_t i = 0; i < batch_size; i++) cipher[i].scale() = 1e6;
            ASSERT_TRUE(all_parms_id(cipher, second_parms_id));
            decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
            decoded = context.encoder().batch_decode_simd(decrypted);
            ASSERT_TRUE(context.batch_near_equal(decoded, zeros));
        }

        // all slots, symmetric
        auto message = context.batch_random_simd_full(batch_size);
        auto plain = context.encoder().batch_encode_simd(message);
        auto plain_ptrs = batch_utils::collect_const_pointer(plain);
        cipher = encryptor.encrypt_symmetric_new_batched(plain_ptrs, false);
        decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
        decoded = context.encoder().batch_decode_simd(decrypted);
        ASSERT_TRUE(context.batch_near_equal(decoded, message));

        if (scheme == SchemeType::BFV) {

            // scale up
            auto plain = encoder.batch_encode_simd(message);
            for (size_t i = 0; i < batch_size; i++) plain[i] = encoder.batch().scale_up_new(plain[i], std::nullopt);
            auto plain_ptrs = batch_utils::collect_const_pointer(plain);
            cipher = encryptor.encrypt_symmetric_new_batched(plain_ptrs, false);
            decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
            decoded = context.encoder().batch_decode_simd(decrypted);
            ASSERT_TRUE(context.batch_near_equal(decoded, message));

            // scale down
            plain = encoder.batch_encode_simd(message);
            plain_ptrs = batch_utils::collect_const_pointer(plain);
            cipher = encryptor.encrypt_symmetric_new_batched(plain_ptrs, false);
            for (size_t i = 0; i < batch_size; i++) decrypted[i] = decryptor.bfv_decrypt_without_scaling_down_new(cipher[i]);
            for (size_t i = 0; i < batch_size; i++) encoder.batch().scale_down_inplace(decrypted[i]);
            decoded = context.encoder().batch_decode_simd(decrypted);
            ASSERT_TRUE(context.batch_near_equal(decoded, message));
        }
        
        // all slots, asymmetric
        plain = context.encoder().batch_encode_simd(message);
        plain_ptrs = batch_utils::collect_const_pointer(plain);
        cipher = encryptor.encrypt_asymmetric_new_batched(plain_ptrs);
        decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
        decoded = context.encoder().batch_decode_simd(decrypted);
        ASSERT_TRUE(context.batch_near_equal(decoded, message));

        // partial slots, asymmetric
        size_t used_slots = (int)(encoder.slot_count() * 0.3);
        message = context.batch_random_simd(batch_size, used_slots);
        plain = context.encoder().batch_encode_simd(message);
        plain_ptrs = batch_utils::collect_const_pointer(plain);
        cipher = encryptor.encrypt_asymmetric_new_batched(plain_ptrs);
        decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
        decoded = context.encoder().batch_decode_simd(decrypted);
        for (size_t i = 0; i < batch_size; i++) message[i].resize(decoded[i].size());
        ASSERT_TRUE(context.batch_near_equal(decoded, message));

        // partial slots, symmetric
        message = context.batch_random_simd(batch_size, used_slots);
        plain = context.encoder().batch_encode_simd(message);
        plain_ptrs = batch_utils::collect_const_pointer(plain);
        cipher = encryptor.encrypt_symmetric_new_batched(plain_ptrs, false);
        decrypted = decryptor.decrypt_batched_new(batch_utils::collect_const_pointer(cipher));
        decoded = context.encoder().batch_decode_simd(decrypted);
        for (size_t i = 0; i < batch_size; i++) message[i].resize(decoded[i].size());
        ASSERT_TRUE(context.batch_near_equal(decoded, message));

        // encrypting symmetric with the same u_prng should give the same c1
        int prng_seed = 0x1234;
        utils::RandomGenerator rng1(prng_seed);
        utils::RandomGenerator rng2(prng_seed);
        auto cipher1 = encryptor.encrypt_symmetric_new_batched(plain_ptrs, false, &rng1);
        auto cipher2 = encryptor.encrypt_symmetric_new_batched(plain_ptrs, false, &rng2);
        for (size_t i = 0; i < batch_size; i++) {
            ASSERT_TRUE(same_vector(cipher1[i].poly(1).as_const(), cipher2[i].poly(1).as_const()));
        }
    }

    void test_bfv(bool device) {
        test_suite(GeneralHeContext(device, SchemeType::BFV, 64, 30, {40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BFV, 64, 30, {40, 40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BFV, 64, 30, {40, 40, 40}, true, 123)); 
    }

    TEST(EncryptorBatchTest, HostBFV) {
        test_bfv(false);
    }

    TEST(EncryptorBatchTest, DeviceBFV) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_bfv(true);
        utils::MemoryPool::Destroy();
    }

    void test_bgv(bool device) {
        test_suite(GeneralHeContext(device, SchemeType::BGV, 64, 30, {40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BGV, 64, 30, {40, 40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BGV, 64, 30, {40, 40, 40}, true, 123));
    }

    TEST(EncryptorBatchTest, HostBGV) {
        test_bgv(false);
    }

    TEST(EncryptorBatchTest, DeviceBGV) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_bgv(true);
        utils::MemoryPool::Destroy();
    }

    void test_ckks(bool device) {
        test_suite(GeneralHeContext(device, SchemeType::CKKS, 64, 30, {40}, false, 123, 10, 1<<16, 1e-2));
        test_suite(GeneralHeContext(device, SchemeType::CKKS, 64, 30, {40, 40}, false, 123, 10, 1<<16, 1e-2));
        test_suite(GeneralHeContext(device, SchemeType::CKKS, 64, 30, {40, 40, 40}, true, 123, 10, 1<<16, 1e-2));
    }

    TEST(EncryptorBatchTest, HostCKKS) {
        test_ckks(false);
    }

    TEST(EncryptorBatchTest, DeviceCKKS) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_ckks(true);
        utils::MemoryPool::Destroy();
    }

}