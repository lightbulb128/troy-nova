#include <cstdlib>
#include <gtest/gtest.h>
#include "../../src/app/bfv_ring2k.h"
#include "../../src/troy.h"
#include "../test.h"

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
            
            Plaintext plaintext = encoder.scale_up_new(message, std::nullopt);
            ASSERT_TRUE(plaintext.coeff_count() == n);
            ASSERT_TRUE(plaintext.poly_modulus_degree() == poly_modulus_degree);
            ASSERT_TRUE(plaintext.data().size() == plaintext.coeff_modulus_size() * n);
            std::vector<T> result = encoder.scale_down_new(plaintext);
            ASSERT_TRUE(same_vector(message, result));

            ParmsID second_parms_id = he_context->first_context_data_pointer()->next_context_data_pointer()->parms_id();
            message = random_sampler<T>(n, t_bit_length);
            plaintext = encoder.scale_up_new(message, second_parms_id);
            result = encoder.scale_down_new(plaintext);
            ASSERT_TRUE(same_vector(message, result));

            // we should be able to scale down more than poly degree's slots
            {
                constexpr size_t n = 54;
                auto message = random_sampler<T>(n, t_bit_length);
                Plaintext total;
                if (device) total.to_device_inplace();
                total.resize_rns_partial(*he_context, he_context->first_parms_id(), n, false, false);
                for (size_t i = 0; i < n; i += poly_modulus_degree) {
                    size_t len = std::min(poly_modulus_degree, n - i);
                    vector<T> partial_message(message.begin() + i, message.begin() + i + len);
                    Plaintext plaintext = encoder.scale_up_new(partial_message, std::nullopt);
                    ASSERT_TRUE(plaintext.coeff_count() == len);
                    ASSERT_TRUE(plaintext.poly_modulus_degree() == poly_modulus_degree);
                    ASSERT_TRUE(plaintext.data().size() == plaintext.coeff_modulus_size() * len);
                    for (size_t j = 0; j < plaintext.coeff_modulus_size(); j++) {
                        total.component(j).slice(i, i + len).copy_from_slice(plaintext.const_component(j));
                    }
                }
                std::vector<T> result = encoder.scale_down_new(total);
                ASSERT_TRUE(same_vector(message, result));
            }
        }
    }

    TEST(BFVRing2kTest, HostScaleUpDownU32Test) {
        template_test_scale_up_down<uint32_t>(false, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceScaleUpDownU32Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
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
        SKIP_WHEN_NO_CUDA_DEVICE;
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
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_scale_up_down<uint128_t>(true, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }


    template <typename T>
    void template_test_scale_up_down_batched(vector<size_t> q_bits, vector<size_t> t_bits) {
        constexpr bool device = true;
        for (size_t t_bit_length: t_bits) {

            constexpr size_t poly_modulus_degree = 32;
            constexpr size_t batch_size = 16;
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

            std::vector<size_t> ns; for (size_t i = 0; i < batch_size; i++) ns.push_back(i);
            std::vector<utils::Array<T>> message(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                message[i] = utils::Array<T>::from_vector(random_sampler<T>(ns[i], t_bit_length));
                message[i].to_device_inplace();
            }

            {
                std::vector<Plaintext> plaintext(batch_size);
                encoder.scale_up_slice_batched(batch_utils::rcollect_const_reference<utils::Array<T>, T>(message), std::nullopt, batch_utils::collect_pointer(plaintext));
                for (size_t i = 0; i < batch_size; i++) {
                    ASSERT_TRUE(plaintext[i].coeff_count() == ns[i]);
                    ASSERT_TRUE(plaintext[i].poly_modulus_degree() == poly_modulus_degree);
                    ASSERT_TRUE(plaintext[i].data().size() == plaintext[i].coeff_modulus_size() * ns[i]);
                    std::vector<T> result = encoder.scale_down_new(plaintext[i]);
                    ASSERT_TRUE(same_vector(message[i].to_vector(), result));
                }
            }
            
            {
                ParmsID second_parms_id = he_context->first_context_data_pointer()->next_context_data_pointer()->parms_id();
                std::vector<Plaintext> plaintext(batch_size);
                encoder.scale_up_slice_batched(batch_utils::rcollect_const_reference<utils::Array<T>, T>(message), second_parms_id, batch_utils::collect_pointer(plaintext));
                for (size_t i = 0; i < batch_size; i++) {
                    ASSERT_TRUE(plaintext[i].coeff_count() == ns[i]);
                    ASSERT_TRUE(plaintext[i].poly_modulus_degree() == poly_modulus_degree);
                    ASSERT_TRUE(plaintext[i].data().size() == plaintext[i].coeff_modulus_size() * ns[i]);
                    std::vector<T> result = encoder.scale_down_new(plaintext[i]);
                    ASSERT_TRUE(same_vector(message[i].to_vector(), result));
                }
            }
        }
    }


    TEST(BFVRing2kTest, DeviceScaleUpDownBatchedU32Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_scale_up_down_batched<uint32_t>( 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceScaleUpDownBatchedU64Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_scale_up_down_batched<uint64_t>( 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, DeviceScaleUpDownBatchedU128Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_scale_up_down_batched<uint128_t>( 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }


    template <typename T>
    void template_test_centralize_decentralize(bool device, vector<size_t> q_bits, vector<size_t> t_bits) {
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
            
            Plaintext plaintext = encoder.centralize_new(message, std::nullopt);
            ASSERT_TRUE(plaintext.coeff_count() == n);
            ASSERT_TRUE(plaintext.poly_modulus_degree() == poly_modulus_degree);
            ASSERT_TRUE(plaintext.data().size() == plaintext.coeff_modulus_size() * n);
            std::vector<T> result = encoder.decentralize_new(plaintext);
            ASSERT_TRUE(same_vector(message, result));

            ParmsID second_parms_id = he_context->first_context_data_pointer()->next_context_data_pointer()->parms_id();
            message = random_sampler<T>(n, t_bit_length);
            plaintext = encoder.centralize_new(message, second_parms_id);
            result = encoder.decentralize_new(plaintext);
            ASSERT_TRUE(same_vector(message, result));
        }
    }

    TEST(BFVRing2kTest, HostCentralizeDecentralizeU32Test) {
        template_test_centralize_decentralize<uint32_t>(false, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceCentralizeDecentralizeU32Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_centralize_decentralize<uint32_t>(true, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, HostCentralizeDecentralizeU64Test) {
        template_test_centralize_decentralize<uint64_t>(false, 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, DeviceCentralizeDecentralizeU64Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_centralize_decentralize<uint64_t>(true, 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, HostCentralizeDecentralizeU128Test) {
        template_test_centralize_decentralize<uint128_t>(false, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }
    TEST(BFVRing2kTest, DeviceCentralizeDecentralizeU128Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_centralize_decentralize<uint128_t>(true, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }

    
    template <typename T>
    void template_test_centralize_decentralize_batched(vector<size_t> q_bits, vector<size_t> t_bits) {
        constexpr bool device = true;
        for (size_t t_bit_length: t_bits) {

            constexpr size_t poly_modulus_degree = 32;
            constexpr size_t batch_size = 16;
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

            std::vector<size_t> ns; for (size_t i = 0; i < batch_size; i++) ns.push_back(i);
            std::vector<utils::Array<T>> message(batch_size);
            for (size_t i = 0; i < batch_size; i++) {
                message[i] = utils::Array<T>::from_vector(random_sampler<T>(ns[i], t_bit_length));
                message[i].to_device_inplace();
            }

            {
                std::vector<Plaintext> plaintext(batch_size);
                encoder.centralize_slice_batched(batch_utils::rcollect_const_reference<utils::Array<T>, T>(message), std::nullopt, batch_utils::collect_pointer(plaintext));
                for (size_t i = 0; i < batch_size; i++) {
                    ASSERT_TRUE(plaintext[i].coeff_count() == ns[i]);
                    ASSERT_TRUE(plaintext[i].poly_modulus_degree() == poly_modulus_degree);
                    ASSERT_TRUE(plaintext[i].data().size() == plaintext[i].coeff_modulus_size() * ns[i]);
                    std::vector<T> result = encoder.decentralize_new(plaintext[i]);
                    ASSERT_TRUE(same_vector(message[i].to_vector(), result));
                }
            }
            
            {
                ParmsID second_parms_id = he_context->first_context_data_pointer()->next_context_data_pointer()->parms_id();
                std::vector<Plaintext> plaintext(batch_size);
                encoder.centralize_slice_batched(batch_utils::rcollect_const_reference<utils::Array<T>, T>(message), second_parms_id, batch_utils::collect_pointer(plaintext));
                for (size_t i = 0; i < batch_size; i++) {
                    ASSERT_TRUE(plaintext[i].coeff_count() == ns[i]);
                    ASSERT_TRUE(plaintext[i].poly_modulus_degree() == poly_modulus_degree);
                    ASSERT_TRUE(plaintext[i].data().size() == plaintext[i].coeff_modulus_size() * ns[i]);
                    std::vector<T> result = encoder.decentralize_new(plaintext[i]);
                    ASSERT_TRUE(same_vector(message[i].to_vector(), result));
                }
            }
        }
    }

    TEST(BFVRing2kTest, DeviceCentralizeDecentralizeBatchedU32Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_centralize_decentralize_batched<uint32_t>( 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceCentralizeDecentralizeBatchedU64Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_centralize_decentralize_batched<uint64_t>( 
            {40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, DeviceCentralizeDecentralizeBatchedU128Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_centralize_decentralize_batched<uint128_t>( 
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
            Plaintext plaintext = encoder.scale_up_new(message, std::nullopt);
            Ciphertext encrypted = encryptor.encrypt_asymmetric_new(plaintext);
            Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(encrypted);
            std::vector<T> result = encoder.scale_down_new(decrypted);
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
        SKIP_WHEN_NO_CUDA_DEVICE;
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
        SKIP_WHEN_NO_CUDA_DEVICE;
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
        SKIP_WHEN_NO_CUDA_DEVICE;
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

            size_t n = 10;
            { // cipher add
                vector<T> m0 = random_sampler<T>(n, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                ASSERT_EQ(p0.coeff_count(), n);
                ASSERT_EQ(p0.poly_modulus_degree(), poly_modulus_degree);
                ASSERT_EQ(p0.data().size(), p0.coeff_modulus_size() * n);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<T> m1 = random_sampler<T>(n, t_bit_length);
                Plaintext p1 = encoder.scale_up_new(m1, std::nullopt);
                Ciphertext c1 = encryptor.encrypt_asymmetric_new(p1);

                Ciphertext cadd = evaluator.add_new(c0, c1);
                vector<T> madd(n);
                for (size_t i = 0; i < n; i++) {
                    madd[i] = (m0[i] + m1[i]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<T> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }

            { // plain add
                vector<T> m0 = random_sampler<T>(n, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<T> m1 = random_sampler<T>(n, t_bit_length);
                Plaintext p1 = encoder.scale_up_new(m1, std::nullopt);

                Ciphertext cadd = evaluator.add_plain_new(c0, p1);
                vector<T> madd(n);
                for (size_t i = 0; i < n; i++) {
                    madd[i] = (m0[i] + m1[i]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<T> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }
            
            { // cipher poly mult by plain scalar
                vector<T> m0 = random_sampler<T>(n, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<T> m1 = random_sampler<T>(1, t_bit_length); // one element
                Plaintext p1 = encoder.centralize_new(m1, std::nullopt); 

                Ciphertext cadd = evaluator.multiply_plain_new(c0, p1);
                vector<T> madd(n);
                for (size_t i = 0; i < n; i++) {
                    madd[i] = (m0[i] * m1[0]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<T> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }

            { // cipher scalar mult by plain poly
                vector<T> m0 = random_sampler<T>(1, t_bit_length); // one element
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<T> m1 = random_sampler<T>(n, t_bit_length);
                Plaintext p1 = encoder.centralize_new(m1, std::nullopt);
                ASSERT_EQ(p1.coeff_count(), n);
                ASSERT_EQ(p1.poly_modulus_degree(), poly_modulus_degree);
                ASSERT_EQ(p1.data().size(), p1.coeff_modulus_size() * n);

                Ciphertext cadd = evaluator.multiply_plain_new(c0, p1);
                vector<T> madd(n);
                for (size_t i = 0; i < n; i++) {
                    madd[i] = (m0[0] * m1[i]) & t_mask;
                }

                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cadd);
                vector<T> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }
            
            { // cipher poly mult by plain poly
                vector<T> m0 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<T> m1 = random_sampler<T>(poly_modulus_degree, t_bit_length);
                Plaintext p1 = encoder.centralize_new(m1, std::nullopt);

                Ciphertext cadd = evaluator.multiply_plain_new(c0, p1);
                vector<T> madd(poly_modulus_degree, 0);
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
                vector<T> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }

            { // cipher poly mult by plain poly partial
                size_t n0 = 10; size_t n1 = 15;
                vector<T> m0 = random_sampler<T>(n0, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                vector<T> m1 = random_sampler<T>(n1, t_bit_length);
                Plaintext p1 = encoder.centralize_new(m1, std::nullopt);

                Ciphertext cadd = evaluator.multiply_plain_new(c0, p1);
                vector<T> madd(poly_modulus_degree, 0);
                for (size_t i = 0; i < n0; i++) {
                    for (size_t j = 0; j < n1; j++) {
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
                vector<T> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(madd, result));
            }

            {
                // mod switch down
                vector<T> m0 = random_sampler<T>(n, t_bit_length);
                Plaintext p0 = encoder.scale_up_new(m0, std::nullopt);
                Ciphertext c0 = encryptor.encrypt_asymmetric_new(p0);
                Ciphertext c1 = evaluator.mod_switch_to_next_new(c0);
                Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(c1);
                vector<T> result = encoder.scale_down_new(decrypted);
                ASSERT_TRUE(same_vector(m0, result));
            }
        }
    }

    TEST(BFVRing2kTest, HostHeOperationsU32Test) {
        template_test_he_operations<uint32_t>(false, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, DeviceHeOperationsU32Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_he_operations<uint32_t>(true, 
            {40, 40, 40}, 
            {32, 20, 17}
        );
    }
    TEST(BFVRing2kTest, HostHeOperationsU64Test) {
        template_test_he_operations<uint64_t>(false, 
            {40, 40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, DeviceHeOperationsU64Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_he_operations<uint64_t>(true, 
            {40, 40, 40, 40, 40}, 
            {64, 50, 33}
        );
    }
    TEST(BFVRing2kTest, HostHeOperationsU128Test) {
        template_test_he_operations<uint128_t>(false, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }
    TEST(BFVRing2kTest, DeviceHeOperationsU128Test) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        template_test_he_operations<uint128_t>(true, 
            {60, 60, 60, 60, 60, 60},
            {128, 100, 65}
        );
    }

}