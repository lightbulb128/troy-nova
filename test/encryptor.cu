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

namespace encryptor {

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

    void test_suite(const GeneralHeContext& context) {
        
        const Encryptor& encryptor = context.encryptor();
        const Decryptor& decryptor = context.decryptor();
        const GeneralEncoder& encoder = context.encoder();
        auto scheme = context.scheme();
        auto first_parms_id = context.context()->first_parms_id();

        // encrypt zero, symmetric
        auto cipher = encryptor.encrypt_zero_symmetric_new(false);
        if (scheme == SchemeType::CKKS) cipher.scale() = 1e6;
        ASSERT_EQ(cipher.parms_id(), first_parms_id);
        auto decrypted = decryptor.decrypt_new(cipher);
        auto decoded = context.encoder().decode_simd(decrypted);
        auto zeros = context.zeros_simd();
        ASSERT_TRUE(context.near_equal(decoded, zeros));

        // encrypt zero, asymmetric
        cipher = encryptor.encrypt_zero_asymmetric_new();
        if (scheme == SchemeType::CKKS) cipher.scale() = 1e6;
        ASSERT_EQ(cipher.parms_id(), first_parms_id);
        decrypted = decryptor.decrypt_new(cipher);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(context.near_equal(decoded, zeros));

        auto he = context.context();
        if (he->first_context_data().value()->next_context_data()) {
            ParmsID second_parms_id = he->first_context_data().value()->next_context_data().value()->parms_id();
            // asymmetric
            cipher = encryptor.encrypt_zero_asymmetric_new(second_parms_id);
            if (scheme == SchemeType::CKKS) cipher.scale() = 1e6;
            ASSERT_EQ(cipher.parms_id(), second_parms_id);
            decrypted = decryptor.decrypt_new(cipher);
            decoded = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(context.near_equal(decoded, zeros));
            // symmetric
            cipher = encryptor.encrypt_zero_asymmetric_new(second_parms_id);
            if (scheme == SchemeType::CKKS) cipher.scale() = 1e6;
            ASSERT_EQ(cipher.parms_id(), second_parms_id);
            decrypted = decryptor.decrypt_new(cipher);
            decoded = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(context.near_equal(decoded, zeros));
        }

        // all slots, symmetric
        auto message = context.random_simd_full();
        auto plain = context.encoder().encode_simd(message);
        cipher = encryptor.encrypt_symmetric_new(plain, false);
        decrypted = decryptor.decrypt_new(cipher);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(context.near_equal(decoded, message));

        if (scheme == SchemeType::BFV) {

            // scale up
            auto plain = encoder.batch().encode_new(message.integers());
            plain = encoder.batch().scale_up_new(plain, std::nullopt);
            cipher = encryptor.encrypt_symmetric_new(plain, false);
            decrypted = decryptor.decrypt_new(cipher);
            decoded = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(context.near_equal(decoded, message));

            // scale down
            plain = encoder.batch().encode_new(message.integers());
            cipher = encryptor.encrypt_symmetric_new(plain, false);
            decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cipher);
            encoder.batch().scale_down_inplace(decrypted);
            decoded = context.encoder().decode_simd(decrypted);
            ASSERT_TRUE(context.near_equal(decoded, message));
        }
        
        // all slots, asymmetric
        plain = context.encoder().encode_simd(message);
        cipher = encryptor.encrypt_asymmetric_new(plain);
        decrypted = decryptor.decrypt_new(cipher);
        decoded = context.encoder().decode_simd(decrypted);
        ASSERT_TRUE(context.near_equal(decoded, message));

        // partial slots, asymmetric
        size_t used_slots = (int)(encoder.slot_count() * 0.3);
        message = context.random_simd(used_slots);
        plain = context.encoder().encode_simd(message);
        cipher = encryptor.encrypt_asymmetric_new(plain);
        decrypted = decryptor.decrypt_new(cipher);
        decoded = context.encoder().decode_simd(decrypted);
        message.resize(decoded.size());
        ASSERT_TRUE(context.near_equal(decoded, message));

        // partial slots, symmetric
        message = context.random_simd(used_slots);
        plain = context.encoder().encode_simd(message);
        cipher = encryptor.encrypt_symmetric_new(plain, false);
        decrypted = decryptor.decrypt_new(cipher);
        decoded = context.encoder().decode_simd(decrypted);
        message.resize(decoded.size());
        ASSERT_TRUE(context.near_equal(decoded, message));

        // encrypting symmetric with the same u_prng should give the same c1
        int prng_seed = 0x1234;
        utils::RandomGenerator rng1(prng_seed);
        utils::RandomGenerator rng2(prng_seed);
        auto cipher1 = encryptor.encrypt_symmetric_new(plain, false, &rng1);
        auto cipher2 = encryptor.encrypt_symmetric_new(plain, false, &rng2);
        ASSERT_TRUE(same_vector(cipher1.poly(1).as_const(), cipher2.poly(1).as_const()));
    }

    void test_bfv(bool device) {
        test_suite(GeneralHeContext(device, SchemeType::BFV, 64, 30, {40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BFV, 64, 30, {40, 40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BFV, 64, 30, {40, 40, 40}, true, 123)); 
    }

    TEST(EncryptorTest, HostBFV) {
        test_bfv(false);
    }

    TEST(EncryptorTest, DeviceBFV) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_bfv(true);
        utils::MemoryPool::Destroy();
    }

    void test_bgv(bool device) {
        test_suite(GeneralHeContext(device, SchemeType::BGV, 64, 30, {40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BGV, 64, 30, {40, 40}, false, 123));
        test_suite(GeneralHeContext(device, SchemeType::BGV, 64, 30, {40, 40, 40}, true, 123));
    }

    TEST(EncryptorTest, HostBGV) {
        test_bgv(false);
    }

    TEST(EncryptorTest, DeviceBGV) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_bgv(true);
        utils::MemoryPool::Destroy();
    }

    void test_ckks(bool device) {
        test_suite(GeneralHeContext(device, SchemeType::CKKS, 64, 30, {40}, false, 123, 10, 1<<16, 1e-2));
        test_suite(GeneralHeContext(device, SchemeType::CKKS, 64, 30, {40, 40}, false, 123, 10, 1<<16, 1e-2));
        test_suite(GeneralHeContext(device, SchemeType::CKKS, 64, 30, {40, 40, 40}, true, 123, 10, 1<<16, 1e-2));
    }

    TEST(EncryptorTest, HostCKKS) {
        test_ckks(false);
    }

    TEST(EncryptorTest, DeviceCKKS) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_ckks(true);
        utils::MemoryPool::Destroy();
    }

    void test_invariant_noise_budget(bool device, bool is_bgv) {
        size_t n = 4096;
        size_t log_t = 20;
        vector<size_t> log_qi = {35, 30, 35};
        EncryptionParameters parms(is_bgv ? SchemeType::BGV : SchemeType::BFV);
        parms.set_poly_modulus_degree(n);
        parms.set_plain_modulus(PlainModulus::batching(n, log_t));
        parms.set_coeff_modulus(CoeffModulus::create(n, log_qi));

        auto context = HeContext::create(parms, true, SecurityLevel::Nil);
        auto encoder = BatchEncoder(context);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }
        auto key_generator = KeyGenerator(context);
        auto public_key = key_generator.create_public_key(false);
        auto encryptor = Encryptor(context);
        encryptor.set_public_key(public_key);
        encryptor.set_secret_key(key_generator.secret_key());
        auto decryptor = Decryptor(context, key_generator.secret_key());
        auto evaluator = Evaluator(context);

        auto ciphertext = encryptor.encrypt_zero_asymmetric_new();
        size_t budget = decryptor.invariant_noise_budget(ciphertext);
        ASSERT_TRUE(budget >= 30 && budget <= 40);
        ASSERT_EQ(encoder.decode_new(decryptor.decrypt_new(ciphertext)), vector<uint64_t>(4096, 0));
        
        evaluator.square_inplace(ciphertext);
        budget = decryptor.invariant_noise_budget(ciphertext);
        ASSERT_TRUE(budget <= 10);
        ASSERT_EQ(encoder.decode_new(decryptor.decrypt_new(ciphertext)), vector<uint64_t>(4096, 0));
        
        evaluator.square_inplace(ciphertext);
        budget = decryptor.invariant_noise_budget(ciphertext);
        ASSERT_TRUE(budget == 0);
        auto wrong = encoder.decode_new(decryptor.decrypt_new(ciphertext));
        size_t non_zero_count = 0;
        for (size_t i = 0; i < n; i++) {
            if (wrong[i] != 0) non_zero_count++;
        }
        ASSERT_TRUE(non_zero_count > 4000);
    }

    TEST(EncryptorTest, HostInvariantNoiseBudget) {
        test_invariant_noise_budget(false, false);
        test_invariant_noise_budget(false, true);
    }

    TEST(EncryptorTest, DeviceInvariantNoiseBudget) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_invariant_noise_budget(true, false);
        test_invariant_noise_budget(true, true);
        utils::MemoryPool::Destroy();
    }
}