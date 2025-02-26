#include <gtest/gtest.h>
#include <sstream>
#include "../test_adv.h"
#include "../../src/app/matmul.h"
#include "../test.h"

namespace matmul_ring2k {

    using namespace troy;
    using namespace troy::linear;
    using std::stringstream;
    using std::vector;
    using uint128_t = __uint128_t;

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
        if (b.size() != a.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    template <typename T>
    void test_matmul(
        bool device,
        size_t t_bits, size_t poly_degree,
        std::vector<size_t> q_bits,
        size_t m, size_t r, size_t n, 
        bool pack_lwe, bool mod_switch_to_next
    ) {
        EncryptionParameters params(SchemeType::BFV);
        params.set_poly_modulus_degree(poly_degree);
        params.set_plain_modulus(1 << 20); // this does not matter
        params.set_coeff_modulus(CoeffModulus::create(poly_degree, q_bits));
        uint128_t t_mask = (t_bits == 128) ? static_cast<uint128_t>(-1) : ((static_cast<uint128_t>(1) << t_bits) - 1);

        HeContextPointer context = HeContext::create(params, true, SecurityLevel::Nil, 0x123);
        PolynomialEncoderRing2k<T> encoder(context, t_bits);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }
        
        vector<T> x = random_sampler<T>(m * r, t_bits);
        vector<T> w = random_sampler<T>(r * n, t_bits);
        vector<T> s = random_sampler<T>(m * n, t_bits);
        MatmulHelper helper(m, r, n, poly_degree, MatmulObjective::EncryptLeft, pack_lwe);

        KeyGenerator keygen(context);
        PublicKey public_key = keygen.create_public_key(false);
        Encryptor encryptor(context); encryptor.set_public_key(public_key);
        encryptor.set_secret_key(keygen.secret_key());
        Evaluator evaluator(context);
        Decryptor decryptor(context, keygen.secret_key());

        GaloisKeys automorphism_key;
        if (pack_lwe) {
            automorphism_key = keygen.create_automorphism_keys(false);
        }
        
        Cipher2d x_encrypted = helper.encrypt_inputs_ring2k(encryptor, encoder, x.data(), std::nullopt);
        Plain2d w_encoded = helper.encode_weights_ring2k(encoder, w.data(), std::nullopt);
        Plain2d s_encoded = helper.encode_outputs_ring2k(encoder, s.data(), std::nullopt);

        stringstream x_serialized;
        x_encrypted.save(x_serialized, context);
        x_encrypted = Cipher2d::load_new(x_serialized, context);

        Cipher2d y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }
        if (pack_lwe) {
            y_encrypted = helper.pack_outputs(evaluator, automorphism_key, y_encrypted);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<T> y_decrypted = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);   

        vector<T> y_truth(m * n, 0);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < r; k++) {
                    y_truth[i * n + j] += x[i * r + k] * w[k * n + j];
                }
                y_truth[i * n + j] += s[i * n + j];
                y_truth[i * n + j] &= t_mask;
            }
        }
        
        ASSERT_TRUE(same_vector(y_truth, y_decrypted));
    }
    
    
    template <typename T>
    void test_matmul_reverse(
        bool device,
        size_t t_bits, size_t poly_degree,
        std::vector<size_t> q_bits,
        size_t m, size_t r, size_t n, 
        bool pack_lwe, bool mod_switch_to_next
    ) {
        EncryptionParameters params(SchemeType::BFV);
        params.set_poly_modulus_degree(poly_degree);
        params.set_plain_modulus(1 << 20); // this does not matter
        params.set_coeff_modulus(CoeffModulus::create(poly_degree, q_bits));
        uint128_t t_mask = (t_bits == 128) ? static_cast<uint128_t>(-1) : ((static_cast<uint128_t>(1) << t_bits) - 1);

        HeContextPointer context = HeContext::create(params, true, SecurityLevel::Nil, 0x123);
        PolynomialEncoderRing2k<T> encoder(context, t_bits);
        if (device) {
            context->to_device_inplace();
            encoder.to_device_inplace();
        }
        
        vector<T> x = random_sampler<T>(m * r, t_bits);
        vector<T> w = random_sampler<T>(r * n, t_bits);
        vector<T> s = random_sampler<T>(m * n, t_bits);
        MatmulHelper helper(m, r, n, poly_degree, MatmulObjective::EncryptRight, pack_lwe);

        KeyGenerator keygen(context);
        PublicKey public_key = keygen.create_public_key(false);
        Encryptor encryptor(context); encryptor.set_public_key(public_key);
        encryptor.set_secret_key(keygen.secret_key());
        Evaluator evaluator(context);
        Decryptor decryptor(context, keygen.secret_key());

        GaloisKeys automorphism_key;
        if (pack_lwe) {
            automorphism_key = keygen.create_automorphism_keys(false);
        }
        
        Plain2d x_encoded = helper.encode_inputs_ring2k(encoder, x.data(), std::nullopt);
        Cipher2d w_encrypted = helper.encrypt_weights_ring2k(encryptor, encoder, w.data(), std::nullopt);
        Plain2d s_encoded = helper.encode_outputs_ring2k(encoder, s.data(), std::nullopt);

        stringstream w_serialized;
        w_encrypted.save(w_serialized, context);
        w_encrypted = Cipher2d::load_new(w_serialized, context);

        Cipher2d y_encrypted = helper.matmul_reverse(evaluator, x_encoded, w_encrypted);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }
        if (pack_lwe) {
            y_encrypted = helper.pack_outputs(evaluator, automorphism_key, y_encrypted);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<T> y_decrypted = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);   

        vector<T> y_truth(m * n, 0);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < r; k++) {
                    y_truth[i * n + j] += x[i * r + k] * w[k * n + j];
                }
                y_truth[i * n + j] += s[i * n + j];
                y_truth[i * n + j] &= t_mask;
            }
        }
        
        ASSERT_TRUE(same_vector(y_truth, y_decrypted));
    }
    

    void ring32_test_suite(bool device) {
        test_matmul<uint32_t>(device, 32, 4096, {60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint32_t>(device, 20, 4096, {60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint32_t>(device, 17, 4096, {60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint32_t>(device, 32, 4096, {60, 60, 60}, 40, 50, 60, false, false);
        test_matmul<uint32_t>(device, 32, 4096, {60, 60, 60}, 40, 50, 60, true, false);
    }

    void ring64_test_suite(bool device) {
        test_matmul<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 4, 5, 6, false, false);  
        test_matmul<uint64_t>(device, 50, 4096, {60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint64_t>(device, 33, 4096, {60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 40, 50, 60, false, false);
        test_matmul<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 40, 50, 60, true, false);
    }

    void ring128_test_suite(bool device) {
        test_matmul<uint128_t>(device, 128, 16, {60, 60, 60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint128_t>(device, 100, 4096, {60, 60, 60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint128_t>(device, 65, 4096, {60, 60, 60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 40, 50, 60, false, false);
        test_matmul<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 40, 50, 60, true, false);
    }

    
    void ring32_test_reverse_suite(bool device) {
        test_matmul_reverse<uint32_t>(device, 32, 4096, {60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint32_t>(device, 20, 4096, {60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint32_t>(device, 17, 4096, {60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint32_t>(device, 32, 4096, {60, 60, 60}, 40, 50, 60, false, false);
        test_matmul_reverse<uint32_t>(device, 32, 4096, {60, 60, 60}, 40, 50, 60, true, false);
    }

    void ring64_test_reverse_suite(bool device) {
        test_matmul_reverse<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 4, 5, 6, false, false);  
        test_matmul_reverse<uint64_t>(device, 50, 4096, {60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint64_t>(device, 33, 4096, {60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 40, 50, 60, false, false);
        test_matmul_reverse<uint64_t>(device, 64, 4096, {60, 60, 60, 60}, 40, 50, 60, true, false);
    }

    void ring128_test_reverse_suite(bool device) {
        test_matmul_reverse<uint128_t>(device, 128, 16, {60, 60, 60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint128_t>(device, 100, 4096, {60, 60, 60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint128_t>(device, 65, 4096, {60, 60, 60, 60, 60, 60}, 4, 5, 6, false, false);
        test_matmul_reverse<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 40, 50, 60, false, false);
        test_matmul_reverse<uint128_t>(device, 128, 4096, {60, 60, 60, 60, 60, 60}, 40, 50, 60, true, false);
    }


    TEST(MatmulTest, HostRing32Matmul) { 
        ring32_test_suite(false); 
    }
    TEST(MatmulTest, DeviceRing32Matmul) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring32_test_suite(true); 
    }
    TEST(MatmulTest, HostRing64Matmul) { 
        ring64_test_suite(false); 
    }
    TEST(MatmulTest, DeviceRing64Matmul) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring64_test_suite(true); 
    }
    TEST(MatmulTest, HostRing128Matmul) { 
        ring128_test_suite(false); 
    }
    TEST(MatmulTest, DeviceRing128Matmul) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring128_test_suite(true); 
    }

    TEST(MatmulTest, HostRing32MatmulReverse) { 
        ring32_test_reverse_suite(false); 
    }
    TEST(MatmulTest, DeviceRing32MatmulReverse) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring32_test_reverse_suite(true); 
    }
    TEST(MatmulTest, HostRing64MatmulReverse) { 
        ring64_test_reverse_suite(false); 
    }
    TEST(MatmulTest, DeviceRing64MatmulReverse) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring64_test_reverse_suite(true); 
    }
    TEST(MatmulTest, HostRing128MatmulReverse) { 
        ring128_test_reverse_suite(false); 
    }
    TEST(MatmulTest, DeviceRing128MatmulReverse) { 
        SKIP_WHEN_NO_CUDA_DEVICE;
        ring128_test_reverse_suite(true); 
    }


}