#include <gtest/gtest.h>
#include "test.cuh"
#include "../src/batch_encoder.cuh"
#include "../src/ckks_encoder.cuh"
#include "../src/encryptor.cuh"
#include "../src/decryptor.cuh"

namespace encryptor {

    using namespace troy;
    using std::vector;
    using troy::utils::ConstSlice;
    using troy::utils::Slice;
    using std::optional;
    using std::complex;

    template<typename T>
    ConstSlice<T> sfv(const vector<T> &vec) {
        return ConstSlice<T>(vec.data(), vec.size(), false);
    }

    class GeneralEncoder {
    private:
        optional<BatchEncoder> batch_;
        optional<CKKSEncoder> ckks_;
    public: 
        GeneralEncoder(BatchEncoder&& batch): batch_(std::move(batch)) {}
        GeneralEncoder(CKKSEncoder&& ckks): ckks_(std::move(ckks)) {}
        const BatchEncoder& batch() const {
            return *batch_;
        }
        const CKKSEncoder& ckks() const {
            return *ckks_;
        }
        void to_device_inplace() {
            if (batch_) {
                batch_->to_device_inplace();
            }
            if (ckks_) {
                ckks_->to_device_inplace();
            }
        }
        size_t slot_count() const {
            return batch_ ? batch_->slot_count() : ckks_->slot_count();
        }
    };

    void test_suite(bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, bool expand_mod_chain, uint64_t seed, double scale) {
        // create enc params
        EncryptionParameters parms(scheme);
        parms.set_poly_modulus_degree(n);
        if (scheme != SchemeType::CKKS) {
            parms.set_plain_modulus(PlainModulus::batching(n, log_t));
        }
        parms.set_coeff_modulus(CoeffModulus::create(n, log_qi));
        // create gadgets
        bool ckks = scheme == SchemeType::CKKS;
        auto context = HeContext::create(parms, expand_mod_chain, SecurityLevel::None, seed);
        auto encoder = ckks ? GeneralEncoder(CKKSEncoder(context)) : GeneralEncoder(BatchEncoder(context));
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
        uint64_t t = ckks ? 0 : parms.plain_modulus()->value();

        vector<uint64_t> message_uint64;
        vector<complex<double>> message_complex64;
        vector<uint64_t> decoded_uint64;
        vector<complex<double>> decoded_complex64;
        size_t slot_count = encoder.slot_count();
        message_complex64.resize(slot_count);
        message_uint64.resize(slot_count);

        // encrypt zero, symmetric
        auto cipher = encryptor.encrypt_zero_symmetric_new(false);
        ASSERT_EQ(cipher.parms_id(), context->first_parms_id());
        auto decrypted = decryptor.decrypt_new(cipher);
        if (ckks) {
            decrypted.scale() = scale;
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
        } else {
            decoded_uint64 = encoder.batch().decode_new(decrypted);
            ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
        }

        // encrypt zero, asymmetric
        if (ckks) for (size_t i = 0; i < slot_count; i++) message_complex64[i] = 0;
        else for (size_t i = 0; i < slot_count; i++) message_uint64[i] = 0;
        cipher = encryptor.encrypt_zero_asymmetric_new();
        ASSERT_EQ(cipher.parms_id(), context->first_parms_id());
        decrypted = decryptor.decrypt_new(cipher);
        if (ckks) {
            decrypted.scale() = scale;
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
        } else {
            decoded_uint64 = encoder.batch().decode_new(decrypted);
            ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
        }

        if (context->first_context_data().value()->next_context_data()) {
            ParmsID second_parms_id = context->first_context_data().value()->next_context_data().value()->parms_id();
            // asymmetric
            cipher = encryptor.encrypt_zero_asymmetric_new(second_parms_id);
            ASSERT_EQ(cipher.parms_id(), second_parms_id);
            decrypted = decryptor.decrypt_new(cipher);
            if (ckks) {
                decrypted.scale() = scale;
                decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
                ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
            } else {
                decoded_uint64 = encoder.batch().decode_new(decrypted);
                ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
            }
            // symmetric
            cipher = encryptor.encrypt_zero_symmetric_new(false, second_parms_id);
            ASSERT_EQ(cipher.parms_id(), second_parms_id);
            decrypted = decryptor.decrypt_new(cipher);
            if (ckks) {
                decrypted.scale() = scale;
                decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
                ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
            } else {
                decoded_uint64 = encoder.batch().decode_new(decrypted);
                ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
            }
        }

        // all slots, symmetric
        if (ckks) message_complex64 = random_complex64_vector(encoder.slot_count());
        else message_uint64 = random_uint64_vector(encoder.slot_count(), t);
        auto plain = ckks ? encoder.ckks().encode_complex64_simd_new(message_complex64, std::nullopt, scale) : encoder.batch().encode_new(message_uint64);
        cipher = encryptor.encrypt_symmetric_new(plain, false);
        decrypted = decryptor.decrypt_new(cipher);
        if (ckks) {
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
        } else {
            decoded_uint64 = encoder.batch().decode_new(decrypted);
            ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
        }
        
        // all slots, asymmetric
        if (ckks) message_complex64 = random_complex64_vector(encoder.slot_count());
        else message_uint64 = random_uint64_vector(encoder.slot_count(), t);
        plain = ckks ? encoder.ckks().encode_complex64_simd_new(message_complex64, std::nullopt, scale) : encoder.batch().encode_new(message_uint64);
        cipher = encryptor.encrypt_asymmetric_new(plain);
        decrypted = decryptor.decrypt_new(cipher);
        if (ckks) {
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
        } else {
            decoded_uint64 = encoder.batch().decode_new(decrypted);
            ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
        }
        
        
        if (ckks && context->first_context_data().value()->next_context_data()) {
            ParmsID second_parms_id = context->first_context_data().value()->next_context_data().value()->parms_id();
            // asymmetric
            cipher = encryptor.encrypt_asymmetric_new(plain);
            decrypted = decryptor.decrypt_new(cipher);
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
            // symmetric
            cipher = encryptor.encrypt_symmetric_new(plain, false);
            decrypted = decryptor.decrypt_new(cipher);
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
        }

        // partial slots, asymmetric
        size_t used_slots = (int)(slot_count * 0.3);
        if (ckks) message_complex64 = random_complex64_vector(used_slots);
        else message_uint64 = random_uint64_vector(used_slots, t);
        plain = ckks ? encoder.ckks().encode_complex64_simd_new(message_complex64, std::nullopt, scale) : encoder.batch().encode_new(message_uint64);
        cipher = encryptor.encrypt_asymmetric_new(plain);
        decrypted = decryptor.decrypt_new(cipher);
        if (ckks) {
            message_complex64.resize(slot_count); for (size_t i = used_slots; i < slot_count; i++) message_complex64[i] = 0;
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
        } else {
            message_uint64.resize(slot_count); for (size_t i = used_slots; i < slot_count; i++) message_uint64[i] = 0;
            decoded_uint64 = encoder.batch().decode_new(decrypted);
            ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
        }

        // partial slots, symmetric
        if (ckks) message_complex64 = random_complex64_vector(used_slots);
        else message_uint64 = random_uint64_vector(used_slots, t);
        plain = ckks ? encoder.ckks().encode_complex64_simd_new(message_complex64, std::nullopt, scale) : encoder.batch().encode_new(message_uint64);
        cipher = encryptor.encrypt_symmetric_new(plain, false);
        decrypted = decryptor.decrypt_new(cipher);
        if (ckks) {
            message_complex64.resize(slot_count); for (size_t i = used_slots; i < slot_count; i++) message_complex64[i] = 0;
            decoded_complex64 = encoder.ckks().decode_complex64_simd_new(decrypted);
            ASSERT_TRUE(near_vector(message_complex64, decoded_complex64));
        } else {
            message_uint64.resize(slot_count); for (size_t i = used_slots; i < slot_count; i++) message_uint64[i] = 0;
            decoded_uint64 = encoder.batch().decode_new(decrypted);
            ASSERT_TRUE(same_vector(message_uint64, decoded_uint64));
        }

        // encrypting symmetric with the same u_prng should give the same c1
        auto prng_seed = seed + 1;
        utils::RandomGenerator rng1(prng_seed);
        utils::RandomGenerator rng2(prng_seed);
        if (device) {
            rng1.init_curand_states(n);
            rng2.init_curand_states(n);
        }
        if (ckks) message_complex64 = random_complex64_vector(encoder.slot_count());
        else message_uint64 = random_uint64_vector(encoder.slot_count(), t);
        plain = ckks ? encoder.ckks().encode_complex64_simd_new(message_complex64, std::nullopt, scale) : encoder.batch().encode_new(message_uint64);
        auto cipher1 = encryptor.encrypt_symmetric_new(plain, false, &rng1);
        auto cipher2 = encryptor.encrypt_symmetric_new(plain, false, &rng2);
        ASSERT_TRUE(same_vector(cipher1.poly(1).as_const(), cipher2.poly(1).as_const()));
    }

    void test_bfv(bool device) {
        test_suite(device, SchemeType::BFV, 64, 30, {40}, false, 123, 1<<20);
        test_suite(device, SchemeType::BFV, 64, 30, {40, 40}, false, 123, 1<<20);
        test_suite(device, SchemeType::BFV, 64, 30, {40, 40, 40}, true, 123, 1<<20);
    }

    TEST(EncryptorTest, HostBFV) {
        test_bfv(false);
    }

    TEST(EncryptorTest, DeviceBFV) {
        test_bfv(true);
    }

    void test_bgv(bool device) {
        test_suite(device, SchemeType::BGV, 64, 30, {40}, false, 123, 1<<20);
        test_suite(device, SchemeType::BGV, 64, 30, {40, 40}, false, 123, 1<<20);
        test_suite(device, SchemeType::BGV, 64, 30, {40, 40, 40}, true, 123, 1<<20);
    }

    TEST(EncryptorTest, HostBGV) {
        test_bgv(false);
    }

    TEST(EncryptorTest, DeviceBGV) {
        test_bgv(true);
    }

    void test_ckks(bool device) {
        test_suite(device, SchemeType::CKKS, 64, 30, {40}, false, 123, 1<<16);
        test_suite(device, SchemeType::CKKS, 64, 30, {40, 40}, false, 123, 1<<16);
        test_suite(device, SchemeType::CKKS, 64, 30, {40, 40, 40}, true, 123, 1<<16);
    }

    TEST(EncryptorTest, HostCKKS) {
        test_ckks(false);
    }

    TEST(EncryptorTest, DeviceCKKS) {
        test_ckks(true);
    }
}