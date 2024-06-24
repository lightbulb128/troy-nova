#include <cstdlib>
#include <gtest/gtest.h>
#include "../../src/app/bfv_ring2k.cuh"
#include "../../src/troy.cuh"

namespace bfv_ring2k {

    using namespace troy;
    using namespace troy::linear;
    using std::stringstream;
    using std::vector;
    using uint128_t = __uint128_t;
    using troy::linear::PolynomialEncoderRing2k;

    template<typename T>
    void print_vector(const vector<T>& vec) {
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); i++) {
            if (i != 0) std::cout << ", ";
            std::cout << vec[i];
        }
        std::cout << "]" << std::endl;
    }

    template <typename T>
    vector<T> random_sampler(size_t size, size_t bit_length) {
        vector<T> result(size);
        T mask = bit_length == (sizeof(T) * 8) ? (static_cast<T>(-1)) : (static_cast<T>(1) << bit_length) - 1;
        for (size_t i = 0; i < size; i++) {
            result[i] = rand() & mask;
        }
        return result;
    }

    template <>
    vector<uint128_t> random_sampler<uint128_t>(size_t size, size_t bit_length) {
        vector<uint128_t> result(size);
        uint128_t mask = bit_length == (sizeof(uint128_t) * 8) ? (static_cast<uint128_t>(-1)) : (static_cast<uint128_t>(1) << bit_length) - 1;
        for (size_t i = 0; i < size; i++) {
            result[i] = ((static_cast<uint128_t>(rand()) << 64) | rand()) & mask;
        }
        return result;
    }

    template <typename T>
    bool same_vector(const vector<T>& a, const vector<T>& b) {
        if (b.size() < a.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        // rest of b is 0
        for (size_t i = a.size(); i < b.size(); i++) {
            if (b[i] != 0) return false;
        }
        return true;
    }

    template <typename T>
    void template_test_scale_up_down(bool device, vector<size_t> q_bits, vector<size_t> t_bits) {
        for (size_t t_bit_length: t_bits) {
            size_t poly_modulus_degree = 32;
            EncryptionParameters parms = EncryptionParameters(SchemeType::BFV);
            parms.set_plain_modulus(PlainModulus::batching(poly_modulus_degree, 30)); // this does not affect anything
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::create(poly_modulus_degree, q_bits));
            HeContextPointer he_context = HeContext::create(parms, true, SecurityLevel::Nil, 0x123);
            PolynomialEncoderRing2k<T> encoder(he_context, t_bit_length);

            if (device) {
                he_context->to_device_inplace();
                encoder.to_device_inplace();
            }

            constexpr size_t n = 10;
            auto message = random_sampler<T>(n, t_bit_length);
            auto p = CoeffModulus::create(poly_modulus_degree, q_bits).to_vector()[0];
            
            Plaintext plaintext = encoder.scale_up_new(message, std::nullopt);
            std::vector<T> result = encoder.scale_down_new(plaintext);
            ASSERT_TRUE(same_vector(message, result));

            ParmsID second_parms_id = he_context->first_context_data_pointer()->next_context_data_pointer()->parms_id();
            message = random_sampler<T>(n, t_bit_length);
            plaintext = encoder.scale_up_new(message, second_parms_id);
            result = encoder.scale_down_new(plaintext);
            ASSERT_TRUE(same_vector(message, result));
        }
    }

    TEST(BFVRing2kTest, HostScaleUpDownU32Test) {
        template_test_scale_up_down<uint32_t>(false, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceScaleUpDownU32Test) {
        template_test_scale_up_down<uint32_t>(true, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, HostScaleUpDownU64Test) {
        template_test_scale_up_down<uint64_t>(false, 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, DeviceScaleUpDownU64Test) {
        template_test_scale_up_down<uint64_t>(true, 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, HostScaleUpDownU128Test) {
        template_test_scale_up_down<uint128_t>(false, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }
    TEST(BFVRing2kTest, DeviceScaleUpDownU128Test) {
        template_test_scale_up_down<uint128_t>(true, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }

    template <typename T>
    void template_test_encrypt(bool device, vector<size_t> q_bits, vector<size_t> t_bits) {
        for (size_t t_bit_length: t_bits) {
            size_t poly_modulus_degree = 32;
            EncryptionParameters parms = EncryptionParameters(SchemeType::BFV);
            parms.set_plain_modulus(PlainModulus::batching(poly_modulus_degree, 30)); // this does not affect anything
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::create(poly_modulus_degree, q_bits));
            HeContextPointer he_context = HeContext::create(parms, true, SecurityLevel::Nil, 0x123);
            PolynomialEncoderRing2k<T> encoder(he_context, t_bit_length);

            if (device) {
                he_context->to_device_inplace();
                encoder.to_device_inplace();
            }

            KeyGenerator keygen(he_context);
            Encryptor encryptor(he_context); encryptor.set_public_key(keygen.create_public_key(false));
            Decryptor decryptor(he_context, keygen.secret_key());
            Evaluator evaluator(he_context);

            constexpr size_t n = 10;
            auto message = random_sampler<T>(n, t_bit_length);
            auto p = CoeffModulus::create(poly_modulus_degree, q_bits).to_vector()[0];
            
            Plaintext plaintext = encoder.scale_up_new(message, std::nullopt);
            Ciphertext encrypted = encryptor.encrypt_asymmetric_new(plaintext);
            Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(encrypted);
            std::vector<T> result = encoder.scale_down_new(plaintext);
            ASSERT_TRUE(same_vector(message, result));
        }
    }

    TEST(BFVRing2kTest, HostEncryptU32Test) {
        template_test_encrypt<uint32_t>(false, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceEncryptU32Test) {
        template_test_encrypt<uint32_t>(true, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, HostEncryptU64Test) {
        template_test_encrypt<uint64_t>(false, 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, DeviceEncryptU64Test) {
        template_test_encrypt<uint64_t>(true, 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, HostEncryptU128Test) {
        template_test_encrypt<uint128_t>(false, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }
    TEST(BFVRing2kTest, DeviceEncryptU128Test) {
        template_test_encrypt<uint128_t>(true, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }

    template <typename T>
    void template_test_he_operations(bool device, vector<size_t> q_bits, vector<size_t> t_bits) {
        for (size_t t_bit_length: t_bits) {
            T t_mask = t_bit_length == (sizeof(T) * 8) ? (static_cast<T>(-1)) : (static_cast<T>(1) << t_bit_length) - 1;
            size_t poly_modulus_degree = 32;
            EncryptionParameters parms = EncryptionParameters(SchemeType::BFV);
            parms.set_plain_modulus(PlainModulus::batching(poly_modulus_degree, 30)); // this does not affect anything
            parms.set_poly_modulus_degree(poly_modulus_degree);
            parms.set_coeff_modulus(CoeffModulus::create(poly_modulus_degree, q_bits));
            HeContextPointer he_context = HeContext::create(parms, true, SecurityLevel::Nil, 0x123);
            PolynomialEncoderRing2k<T> encoder(he_context, t_bit_length);

            if (device) {
                he_context->to_device_inplace();
                encoder.to_device_inplace();
            }

            KeyGenerator keygen(he_context);
            Encryptor encryptor(he_context); encryptor.set_public_key(keygen.create_public_key(false));
            Decryptor decryptor(he_context, keygen.secret_key());
            Evaluator evaluator(he_context);

            { // cipher add
                vector<uint64_t> m0 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<uint64_t> m1 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p1 = encoder.scale_up_new(m1, std::nullopt);
                Ciphertext c1 = encryptor.encrypt_asymmetric_new(p1);

                Ciphertext cadd = evaluator.add_new(c0, c1);
                vector<uint64_t> madd(poly_modulus_degree);
                for (size_t i = 0; i < poly_modulus_degree; i++) {
                    madd[i] = (m0[i] + m1[i]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<uint64_t> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }

            { // plain add
                vector<uint64_t> m0 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<uint64_t> m1 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p1 = encoder.scale_up_new(m1, std::nullopt);

                Ciphertext cadd = evaluator.add_plain_new(c0, p1);
                vector<uint64_t> madd(poly_modulus_degree);
                for (size_t i = 0; i < poly_modulus_degree; i++) {
                    madd[i] = (m0[i] + m1[i]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<uint64_t> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }
            
            { // cipher poly mult by plain scalar
                vector<uint64_t> m0 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<uint64_t> m1 = random_sampler<T>(1, t_bit_length); // one element
                Plaintext p1 = encoder.scale_up_new(m1, std::nullopt);

                Ciphertext cadd = evaluator.add_plain_new(c0, p1);
                vector<uint64_t> madd(poly_modulus_degree);
                for (size_t i = 0; i < poly_modulus_degree; i++) {
                    madd[i] = (m0[i] + m1[0]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<uint64_t> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }

            { // cipher scalar mult by plain poly
                vector<uint64_t> m0 = random_sampler<T>(1, t_bit_length); // one element
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<uint64_t> m1 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p1 = encoder.scale_up_new(m1, std::nullopt);

                Ciphertext cadd = evaluator.add_plain_new(c0, p1);
                vector<uint64_t> madd(poly_modulus_degree);
                for (size_t i = 0; i < poly_modulus_degree; i++) {
                    madd[i] = (m0[0] + m1[i]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<uint64_t> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }
            
            { // cipher poly mult by plain poly
                vector<uint64_t> m0 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<uint64_t> m1 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p1 = encoder.centralize_new(m1, std::nullopt);

                Ciphertext cadd = evaluator.add_plain_new(c0, p1);
                vector<uint64_t> madd(poly_modulus_degree, 0);
                for (size_t i = 0; i < poly_modulus_degree; i++) {
                    for (size_t j = 0; j < poly_modulus_degree; j++) {
                        T piece = (m0[i] * m1[j]) & t_mask;
                        size_t id = (i + j) % poly_modulus_degree;
                        if (i + j < poly_modulus_degree) {
                            madd[id] = (madd[id] + piece) & t_mask;
                        } else {
                            madd[id] = (madd[id] - piece) & t_mask;
                        }
                    }
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<uint64_t> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }
        }
    }

    TEST(BFVRing2kTest, HostHeOperationsU32Test) {
        template_test_encrypt<uint32_t>(false, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceHeOperationsU32Test) {
        template_test_encrypt<uint32_t>(true, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, HostHeOperationsU64Test) {
        template_test_encrypt<uint64_t>(false, 
            {40, 40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, DeviceHeOperationsU64Test) {
        template_test_encrypt<uint64_t>(true, 
            {40, 40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, HostHeOperationsU128Test) {
        template_test_encrypt<uint128_t>(false, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }
    TEST(BFVRing2kTest, DeviceHeOperationsU128Test) {
        template_test_encrypt<uint128_t>(true, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }

}