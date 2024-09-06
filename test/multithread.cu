#include "test_adv.h"
#include "test.h"
#include <gtest/gtest.h>
#include <thread>
#include <future>

namespace multithread {

#define IF_FALSE_RETURN(condition) if (!(condition)) { return false; }
#define IF_FALSE_PRINT_RETURN(condition, message) \
    if (!(condition)) {                           \
        std::cerr << "[" << thread_index << "] File " << __FILE__ << ", Line " << __LINE__ << ": " << message << std::endl; \
        return false;                             \
    }                                   
#define CHECKPOINT(message) std::cerr << "[" << thread_index << "] " << message << std::endl; 

    using namespace std;
    using namespace troy;
    using troy::utils::Array;
    using tool::GeneralHeContext;
    using tool::GeneralVector;

    void test_allocate(bool device, size_t threads, size_t repeat, size_t size) {

        auto allocate = [size, device](int t) {
            return std::move(Array<int>(size, device));
        };

        for (size_t r = 0; r < repeat; r++) {
            vector<future<Array<int>>> futures;
            vector<Array<int>> arrays;
            for (size_t t = 0; t < threads; t++) {
                futures.push_back(std::move(std::async(allocate, t)));
            }
            for (size_t t = 0; t < threads; t++) {
                arrays.push_back(std::move(futures[t].get()));
            }
            // check no array have the same address
            for (size_t t = 0; t < threads; t++) {
                for (size_t t2 = t + 1; t2 < threads; t2++) {
                    int* ptr1 = arrays[t].raw_pointer();
                    int* ptr2 = arrays[t2].raw_pointer();
                    ASSERT_NE(ptr1, ptr2);
                }
            }
        }

    }

    TEST(MultithreadTest, HostAllocate) {
        test_allocate(false, 64, 4, 64);
    }
    TEST(MultithreadTest, DeviceAllocate) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_allocate(true, 64, 4, 64);
    }

    void test_single_pool_multi_thread(const GeneralHeContext& context, size_t threads, size_t repeat) {

        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        auto test_thread = [t, scale, repeat, tolerance, &context](int thread) {

            for (size_t rep = 0; rep < repeat; rep++) {
            
                GeneralVector message1 = context.random_simd_full();
                GeneralVector message2 = context.random_simd_full();
                RelinKeys relin_keys = context.key_generator().create_relin_keys(false, 3);
                Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
                Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
                Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);
                Ciphertext multiplied = context.evaluator().multiply_new(encrypted1, encrypted2);
                Ciphertext relined = context.evaluator().relinearize_new(multiplied, relin_keys);
                Plaintext decrypted = context.decryptor().decrypt_new(relined);
                GeneralVector result = context.encoder().decode_simd(decrypted);
                GeneralVector truth = message1.mul(message2, t);
                IF_FALSE_RETURN(truth.near_equal(result, tolerance));

                GeneralVector message3 = context.random_simd_full();
                Plaintext encoded3 = context.encoder().encode_simd(message3, std::nullopt, relined.scale());
                Ciphertext encrypted3 = context.encryptor().encrypt_asymmetric_new(encoded3);
                multiplied = context.evaluator().multiply_new(multiplied, encrypted3);
                relined = context.evaluator().relinearize_new(multiplied, relin_keys);
                decrypted = context.decryptor().decrypt_new(relined);
                result = context.encoder().decode_simd(decrypted);
                truth = truth.mul(message3, t);
                IF_FALSE_RETURN(truth.near_equal(result, tolerance));

            }

            return true;

        };

        utils::stream_sync();
        vector<std::future<bool>> thread_instances;
        for (size_t i = 0; i < threads; i++) {
            thread_instances.push_back(std::async(test_thread, i));
        }

        for (size_t i = 0; i < threads; i++) {
            ASSERT_TRUE(thread_instances[i].get());
        }

    }    
    
    TEST(MultithreadTest, HostSharedPoolSimple) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
        test_single_pool_multi_thread(ghe, 64, 4);
    }
    TEST(MultithreadTest, DeviceSharedPoolSimple) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
        test_single_pool_multi_thread(ghe, 64, 4);
        utils::MemoryPool::Destroy();
    }

    bool test_troublesome_pools(const GeneralHeContext& context, size_t device_index, size_t thread_index = 0, bool check_pool = true, bool shared_pool = false) {

        auto context_pool = context.context()->pool();

        SchemeType scheme = context.scheme();
        double scale = context.scale();
        double tolerance = context.tolerance();
        bool device = context.context()->on_device();
        if (!device) {
            IF_FALSE_PRINT_RETURN(context_pool == nullptr, "context_pool");
        }
        
        auto good_pool = [device, context_pool, check_pool](MemoryPoolHandle pool, MemoryPoolHandle expect) {
            if (!check_pool) return true;
            if (device) {
                bool all_on_device = pool != nullptr && expect != nullptr && context_pool != nullptr;
                bool same = pool != context_pool && pool == expect;
                return all_on_device && same;
            } else {
                bool all_nullptr = pool == nullptr && expect == nullptr && context_pool == nullptr;
                return all_nullptr;
            }
        };

        auto create_new_memory_pool = [device, device_index, shared_pool, &context]() {
            if (shared_pool) return context.pool();
            return device ? MemoryPool::create(device_index) : nullptr;
        };


        // test batch encoder
        if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) { 
            const BatchEncoder& encoder = context.encoder().batch();

            if (encoder.simd_encoding_supported()) {

                { // encode
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_simd_full();
                    Plaintext encoded;
                    encoder.encode(message.integers(), encoded, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode/pool");
                    GeneralVector decoded = context.encoder().decode_simd(encoded, pool);
                    IF_FALSE_PRINT_RETURN(message.near_equal(decoded, tolerance), "encode/correct");
                }

                { // encode_new
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_simd_full();
                    Plaintext encoded = encoder.encode_new(message.integers(), pool);
                    IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_new/pool");
                    GeneralVector decoded = context.encoder().decode_simd(encoded, pool);
                    IF_FALSE_PRINT_RETURN(message.near_equal(decoded, tolerance), "encode_new/correct");
                }

            } // if simd_encoding_supported


            { // encode_polynomial
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_polynomial_full();
                Plaintext encoded;
                encoder.encode_polynomial(message.integers(), encoded, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_polynomial/pool");
                GeneralVector decoded = context.encoder().decode_polynomial(encoded, pool);
                IF_FALSE_PRINT_RETURN(message.near_equal(decoded, tolerance), "encode_polynomial/correct");                        
            }

            { // encode_polynomial_new
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_polynomial_full();
                Plaintext encoded = encoder.encode_polynomial_new(message.integers(), pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_polynomial_new/pool");
                GeneralVector decoded = context.encoder().decode_polynomial(encoded, pool);
                IF_FALSE_PRINT_RETURN(message.near_equal(decoded, tolerance), "encode_polynomial_new/correct");
            }


            if (scheme == SchemeType::BFV) {

                { // scale_up and scale_down
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_polynomial_full();
                    Plaintext encoded = encoder.encode_polynomial_new(message.integers(), pool);
                    Plaintext scaled;
                    encoder.scale_up(encoded, scaled, std::nullopt, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(scaled.pool(), pool), "scale_up/pool");
                    Plaintext scaled_down;
                    encoder.scale_down(scaled, scaled_down, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(scaled_down.pool(), pool), "scale_down/pool");
                    GeneralVector decoded = context.encoder().decode_polynomial(scaled_down, pool);
                    IF_FALSE_PRINT_RETURN(message.near_equal(decoded, tolerance), "scale_up/correct");
                }

                { // scale_up_new and scale_down_new
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_polynomial_full();
                    Plaintext encoded = encoder.encode_polynomial_new(message.integers(), pool);
                    Plaintext scaled = encoder.scale_up_new(encoded, std::nullopt, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(scaled.pool(), pool), "scale_up_new/pool");
                    Plaintext scaled_down = encoder.scale_down_new(scaled, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(scaled_down.pool(), pool), "scale_down_new/pool");
                    GeneralVector decoded = context.encoder().decode_polynomial(scaled_down, pool);
                    IF_FALSE_PRINT_RETURN(message.near_equal(decoded, tolerance), "scale_up_new/correct");
                }

                { // scale_up_inplace and scale_down_inplace
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_polynomial_full();
                    Plaintext encoded = encoder.encode_polynomial_new(message.integers(), pool);
                    Plaintext scaled = encoded.clone(pool);
                    IF_FALSE_PRINT_RETURN(good_pool(scaled.pool(), pool), "scale_up_inplace/clone");
                    encoder.scale_up_inplace(scaled, std::nullopt, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(scaled.pool(), pool), "scale_up_inplace/pool");
                    encoder.scale_down_inplace(scaled, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(scaled.pool(), pool), "scale_down_inplace/pool");
                    GeneralVector decoded = context.encoder().decode_polynomial(scaled, pool);
                    IF_FALSE_PRINT_RETURN(message.near_equal(decoded, tolerance), "scale_up_inplace/correct");
                }

                { // centralize
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_polynomial_full();
                    Plaintext encoded = encoder.encode_polynomial_new(message.integers(), pool);
                    Plaintext centralized;
                    encoder.centralize(encoded, centralized, std::nullopt, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(centralized.pool(), pool), "centralize/pool");
                }

                { // centralize_new
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_polynomial_full();
                    Plaintext encoded = encoder.encode_polynomial_new(message.integers(), pool);
                    Plaintext centralized = encoder.centralize_new(encoded, std::nullopt, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(centralized.pool(), pool), "centralize_new/pool");
                    
                }
                
                { // centralize_inplace
                    MemoryPoolHandle pool = create_new_memory_pool();
                    GeneralVector message = context.random_polynomial_full();
                    Plaintext encoded = encoder.encode_polynomial_new(message.integers(), pool);
                    Plaintext centralized = encoded.clone(pool);
                    IF_FALSE_PRINT_RETURN(good_pool(centralized.pool(), pool), "centralize_inplace/clone");
                    encoder.centralize_inplace(centralized, std::nullopt, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(centralized.pool(), pool), "centralize_inplace/pool");
                }

            } // if scheme == bfv

        }

        // test ckks encoder
        if (scheme == SchemeType::CKKS) {
            const CKKSEncoder& encoder = context.encoder().ckks();
            
            { // encode_complex64_simd
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_simd_full();
                Plaintext encoded;
                encoder.encode_complex64_simd(message.complexes(), std::nullopt, scale, encoded, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_complex64_simd/pool");
                GeneralVector decoded = context.encoder().decode_simd(encoded, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_complex64_simd/correct");
            }

            { // encode_complex64_simd_new
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_simd_full();
                Plaintext encoded = encoder.encode_complex64_simd_new(message.complexes(), std::nullopt, scale, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_complex64_simd_new/pool");
                GeneralVector decoded = context.encoder().decode_simd(encoded, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_complex64_simd_new/correct");
            }

            { // encode_float64_single
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_coefficient_repeated_full();
                Plaintext encoded;
                encoder.encode_float64_single(message.doubles()[0], std::nullopt, scale, encoded, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_float64_single/pool");
                GeneralVector decoded = context.encoder().decode_polynomial(encoded, pool);
                for (size_t i = 1; i < context.coeff_count(); i++) message.doubles()[i] = 0;
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_float64_single/correct");
            }

            { // encode_float64_single_new
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_coefficient_repeated_full();
                Plaintext encoded = encoder.encode_float64_single_new(message.doubles()[0], std::nullopt, scale, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_float64_single_new/pool");
                GeneralVector decoded = context.encoder().decode_polynomial(encoded, pool);
                for (size_t i = 1; i < context.coeff_count(); i++) message.doubles()[i] = 0;
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_float64_single_new/correct");
            }

            { // encode_float64_polynomial
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_polynomial_full();
                Plaintext encoded;
                encoder.encode_float64_polynomial(message.doubles(), std::nullopt, scale, encoded, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_float64_polynomial/pool");
                GeneralVector decoded = context.encoder().decode_polynomial(encoded, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_float64_polynomial/correct");
            }

            { // encode_float64_polynomial_new
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_polynomial_full();
                Plaintext encoded = encoder.encode_float64_polynomial_new(message.doubles(), std::nullopt, scale, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_float64_polynomial_new/pool");
                GeneralVector decoded = context.encoder().decode_polynomial(encoded, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_float64_polynomial_new/correct");
            }

            { // encode_complex64_single
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_slot_repeated_full();
                Plaintext encoded;
                encoder.encode_complex64_single(message.complexes()[0], std::nullopt, scale, encoded, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_complex64_single/pool");
                GeneralVector decoded = context.encoder().decode_simd(encoded, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_complex64_single/correct");
            }

            { // encode_complex64_single_new
                MemoryPoolHandle pool = create_new_memory_pool();
                GeneralVector message = context.random_slot_repeated_full();
                Plaintext encoded = encoder.encode_complex64_single_new(message.complexes()[0], std::nullopt, scale, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded.pool(), pool), "encode_complex64_single_new/pool");
                GeneralVector decoded = context.encoder().decode_simd(encoded, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "encode_complex64_single_new/correct");
            }

        } // if scheme == ckks

        { // test encryption and decryption
            GeneralVector message = context.random_polynomial_full();
            GeneralVector zeros = context.zeros_polynomial();

            { // encrypt_asymmetric
                MemoryPoolHandle pool = create_new_memory_pool();
                Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale, pool);
                Ciphertext encrypted;
                context.encryptor().encrypt_asymmetric(encoded, encrypted, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_asymmetric/pool");
                Plaintext decrypted;
                context.decryptor().decrypt(encrypted, decrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(decrypted.pool(), pool), "decrypt/pool");
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, result), "encrypt_asymmetric/correct");
            }

            { // encrypt_asymmetric_new
                MemoryPoolHandle pool = create_new_memory_pool();
                Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale, pool);
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_asymmetric_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(decrypted.pool(), pool), "decrypt_new/pool");
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, result), "encrypt_asymmetric_new/correct");
            }

            { // encrypt_zero_asymmetric
                MemoryPoolHandle pool = create_new_memory_pool();
                Ciphertext encrypted;
                context.encryptor().encrypt_zero_asymmetric(encrypted, std::nullopt, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_zero_asymmetric/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                // we don't test pool of decrypted again as it is already tested above
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(result.near_equal(zeros, tolerance * scale), "encrypt_zero_asymmetric/correct");
            }

            { // encrypt_zero_asymmetric_new
                MemoryPoolHandle pool = create_new_memory_pool();
                Ciphertext encrypted = context.encryptor().encrypt_zero_asymmetric_new(std::nullopt, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_zero_asymmetric_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(result.near_equal(zeros, tolerance * scale), "encrypt_zero_asymmetric_new/correct");
            }

            { // encrypt_symmetric
                MemoryPoolHandle pool = create_new_memory_pool();
                Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale, pool);
                Ciphertext encrypted;
                context.encryptor().encrypt_symmetric(encoded, false, encrypted, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_symmetric/pool");
                Plaintext decrypted;
                context.decryptor().decrypt(encrypted, decrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(decrypted.pool(), pool), "decrypt/pool");
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, result), "encrypt_symmetric/correct");
            }

            { // encrypt_symmetric_new
                MemoryPoolHandle pool = create_new_memory_pool();
                Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale, pool);
                Ciphertext encrypted = context.encryptor().encrypt_symmetric_new(encoded, false, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_symmetric_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(decrypted.pool(), pool), "decrypt_new/pool");
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, result), "encrypt_symmetric_new/correct");
            }

            { // encrypt_zero_symmetric
                MemoryPoolHandle pool = create_new_memory_pool();
                Ciphertext encrypted;
                context.encryptor().encrypt_zero_symmetric(false, encrypted, std::nullopt, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_zero_symmetric/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(result.near_equal(zeros, tolerance * scale), "encrypt_zero_symmetric/correct");
            }

            { // encrypt_zero_symmetric_new
                MemoryPoolHandle pool = create_new_memory_pool();
                Ciphertext encrypted = context.encryptor().encrypt_zero_symmetric_new(false, std::nullopt, nullptr, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "encrypt_zero_symmetric_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector result = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(result.near_equal(zeros, tolerance * scale), "encrypt_zero_symmetric_new/correct");
            }

            if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) { // invariant_noise_budget
                MemoryPoolHandle pool = create_new_memory_pool();
                Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale, pool);
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                int64_t budget = context.decryptor().invariant_noise_budget(encrypted, pool);
                IF_FALSE_PRINT_RETURN(budget >= 0, "invariant_noise_budget");
            }

        }
            

        { // negate
            MemoryPoolHandle pool = create_new_memory_pool();
            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale, pool);
            GeneralVector truth = context.negate(message);
            { // negate
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                Ciphertext result;
                context.evaluator().negate(encrypted, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "negate/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "negate/correct");
            }
            { // negate_inplace
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                context.evaluator().negate_inplace(encrypted);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "negate_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "negate_inplace/correct");
            }
            { // negate_new
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                Ciphertext result = context.evaluator().negate_new(encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "negate_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "negate_new/correct");
            }
        }

        { // add
            GeneralVector message0 = context.random_simd_full();
            GeneralVector message1 = context.random_simd_full();
            GeneralVector truth = context.add(message0, message1);
            MemoryPoolHandle pool = create_new_memory_pool();
            Plaintext encoded0 = context.encoder().encode_simd(message0, std::nullopt, scale, pool);
            Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale, pool);
            { // add
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                Ciphertext result;
                context.evaluator().add(encrypted0, encrypted1, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add/correct");
            }
            { // add_inplace
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                context.evaluator().add_inplace(encrypted0, encrypted1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "add_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted0, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_inplace/correct");
            }
            { // add_new
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                Ciphertext result = context.evaluator().add_new(encrypted0, encrypted1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_new/correct");
            }
            if (scheme == SchemeType::BFV) { // add ntt
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                context.evaluator().transform_to_ntt_inplace(encrypted0);
                context.evaluator().transform_to_ntt_inplace(encrypted1);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "add_ntt/pool");
                Ciphertext result = context.evaluator().add_new(encrypted0, encrypted1, pool);
                context.evaluator().transform_from_ntt_inplace(result);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add_ntt/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_ntt/correct");
            }
            if (scheme == SchemeType::BGV || scheme == SchemeType::CKKS) { // add intt
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                context.evaluator().transform_from_ntt_inplace(encrypted0);
                context.evaluator().transform_from_ntt_inplace(encrypted1);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "add_intt/pool");
                Ciphertext result = context.evaluator().add_new(encrypted0, encrypted1, pool);
                context.evaluator().transform_to_ntt_inplace(result);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add_intt/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_intt/correct");
            }
            { // add_plain
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext result;
                context.evaluator().add_plain(encrypted0, encoded1, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add_plain/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_plain/correct");
            }
            { // add_plain_inplace
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                context.evaluator().add_plain_inplace(encrypted0, encoded1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "add_plain_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted0, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_plain_inplace/correct");
            }
            { // add_plain_new
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext result = context.evaluator().add_plain_new(encrypted0, encoded1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add_plain_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_plain_new/correct");
            }
            if (scheme == SchemeType::BFV) { // add_plain scaled
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Plaintext encoded1_scaled_up = context.encoder().batch().scale_up_new(encoded1, std::nullopt, pool);
                Ciphertext result;
                context.evaluator().add_plain(encrypted0, encoded1_scaled_up, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add_plain_scaled/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_plain_scaled/correct");
            }
            if (scheme == SchemeType::BFV) { // add_plain scaled ntt
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Plaintext encoded1_scaled_up = context.encoder().batch().scale_up_new(encoded1, std::nullopt, pool);
                context.evaluator().transform_to_ntt_inplace(encrypted0);
                context.evaluator().transform_plain_to_ntt_inplace(encoded1_scaled_up, encrypted0.parms_id(), pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded1_scaled_up.pool(), pool), "add_plain_scaled_ntt/pool");
                Ciphertext result;
                context.evaluator().add_plain(encrypted0, encoded1_scaled_up, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "add_plain_scaled/pool");
                context.evaluator().transform_from_ntt_inplace(result);
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "add_plain_scaled/correct");
            }
        }

        { // sub
            GeneralVector message0 = context.random_simd_full();
            GeneralVector message1 = context.random_simd_full();
            GeneralVector truth = context.sub(message0, message1);
            MemoryPoolHandle pool = create_new_memory_pool();
            Plaintext encoded0 = context.encoder().encode_simd(message0, std::nullopt, scale, pool);
            Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale, pool);
            { // sub
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                Ciphertext result;
                context.evaluator().sub(encrypted0, encrypted1, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub/correct");
            }
            { // sub_inplace
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                context.evaluator().sub_inplace(encrypted0, encrypted1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "sub_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted0, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_inplace/correct");
            }
            { // sub_new
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                Ciphertext result = context.evaluator().sub_new(encrypted0, encrypted1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_new/correct");
            }
            if (scheme == SchemeType::BFV) { // sub ntt
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                context.evaluator().transform_to_ntt_inplace(encrypted0);
                context.evaluator().transform_to_ntt_inplace(encrypted1);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "sub_ntt/pool");
                Ciphertext result = context.evaluator().sub_new(encrypted0, encrypted1, pool);
                context.evaluator().transform_from_ntt_inplace(result);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub_ntt/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_ntt/correct");
            }
            if (scheme == SchemeType::BGV || scheme == SchemeType::CKKS) { // sub intt
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                context.evaluator().transform_from_ntt_inplace(encrypted0);
                context.evaluator().transform_from_ntt_inplace(encrypted1);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "sub_intt/pool");
                Ciphertext result = context.evaluator().sub_new(encrypted0, encrypted1, pool);
                context.evaluator().transform_to_ntt_inplace(result);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub_intt/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_intt/correct");
            }
            { // sub_plain
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext result;
                context.evaluator().sub_plain(encrypted0, encoded1, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub_plain/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_plain/correct");
            }
            { // sub_plain_inplace
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                context.evaluator().sub_plain_inplace(encrypted0, encoded1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "sub_plain_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted0, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_plain_inplace/correct");
            }
            { // sub_plain_new
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext result = context.evaluator().sub_plain_new(encrypted0, encoded1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub_plain_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_plain_new/correct");
            }
            if (scheme == SchemeType::BFV) { // sub_plain scaled
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Plaintext encoded1_scaled_up = context.encoder().batch().scale_up_new(encoded1, std::nullopt, pool);
                Ciphertext result;
                context.evaluator().sub_plain(encrypted0, encoded1_scaled_up, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub_plain_scaled/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_plain_scaled/correct");
            }
            if (scheme == SchemeType::BFV) { // sub_plain scaled ntt
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Plaintext encoded1_scaled_up = context.encoder().batch().scale_up_new(encoded1, std::nullopt, pool);
                context.evaluator().transform_to_ntt_inplace(encrypted0);
                context.evaluator().transform_plain_to_ntt_inplace(encoded1_scaled_up, encrypted0.parms_id(), pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded1_scaled_up.pool(), pool), "sub_plain_scaled_ntt/pool");
                Ciphertext result;
                context.evaluator().sub_plain(encrypted0, encoded1_scaled_up, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "sub_plain_scaled/pool");
                context.evaluator().transform_from_ntt_inplace(result);
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "sub_plain_scaled/correct");
            }
        }
        
        { // multiply
            GeneralVector message0 = context.random_simd_full();
            GeneralVector message1 = context.random_simd_full();
            GeneralVector truth = context.mul(message0, message1);
            MemoryPoolHandle pool = create_new_memory_pool();
            Plaintext encoded0 = context.encoder().encode_simd(message0, std::nullopt, scale, pool);
            Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale, pool);
            { // multiply
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                Ciphertext result;
                context.evaluator().multiply(encrypted0, encrypted1, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(decrypted.pool(), pool), "multiply/decrypt-pool");
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply/correct");
            }
            { // multiply_inplace
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                context.evaluator().multiply_inplace(encrypted0, encrypted1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "multiply_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted0, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_inplace/correct");
            }
            { // multiply_new
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1, nullptr, pool);
                Ciphertext result = context.evaluator().multiply_new(encrypted0, encrypted1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_new/correct");
            }
            { // multiply_plain
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext result;
                context.evaluator().multiply_plain(encrypted0, encoded1, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_plain/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_plain/correct");     
            }
            { // multiply_plain_inplace
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                context.evaluator().multiply_plain_inplace(encrypted0, encoded1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "multiply_plain_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted0, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_plain_inplace/correct");
            }
            { // multiply_plain_new
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext result = context.evaluator().multiply_plain_new(encrypted0, encoded1, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_plain_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_plain_new/correct");           
                if (!encrypted0.is_ntt_form()) {
                    context.evaluator().transform_to_ntt_inplace(encrypted0);
                    context.evaluator().transform_plain_to_ntt_inplace(encoded1, encrypted0.parms_id(), pool);
                    result = context.evaluator().multiply_plain_new(encrypted0, encoded1, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_plain_ntt1/pool");
                    context.evaluator().transform_from_ntt_inplace(result);
                    decrypted = context.decryptor().decrypt_new(result, pool);
                    decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_plain_ntt1/correct");
                } else {
                    context.evaluator().transform_from_ntt_inplace(encrypted0);
                    IF_FALSE_PRINT_RETURN(good_pool(encrypted0.pool(), pool), "multiply_plain_intt/pool");
                    context.evaluator().transform_to_ntt_inplace(encrypted0);
                    result = context.evaluator().multiply_plain_new(encrypted0, encoded1, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_plain_intt/pool");
                    decrypted = context.decryptor().decrypt_new(result, pool);
                    decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_plain_intt/correct");
                }
            }
            if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) { // multiply_plain_ntt
                encoded0 = context.encoder().encode_simd(message0, std::nullopt, scale, pool);
                encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale, pool);
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Ciphertext result;
                context.evaluator().transform_plain_to_ntt_inplace(encoded1, encrypted0.parms_id(), pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded1.pool(), pool), "multiply_plain_ntt2/pool");
                context.evaluator().multiply_plain(encrypted0, encoded1, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_plain_ntt2/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_plain_ntt2/correct");
            }
            if (scheme == SchemeType::BFV) { // multiply_plain centralized
                encoded0 = context.encoder().encode_simd(message0, std::nullopt, scale, pool);
                encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale, pool);
                Ciphertext encrypted0 = context.encryptor().encrypt_asymmetric_new(encoded0, nullptr, pool);
                Plaintext encoded1_centralized = context.encoder().batch().centralize_new(encoded1, std::nullopt, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encoded1_centralized.pool(), pool), "multiply_plain_centralized/pool");
                Ciphertext result;
                context.evaluator().multiply_plain(encrypted0, encoded1_centralized, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_plain_centralized/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_plain_centralized/correct");
            }
        }

        { // square
            GeneralVector message = context.random_simd_full();
            GeneralVector truth = context.mul(message, message);
            MemoryPoolHandle pool = create_new_memory_pool();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale, pool);
            { // square
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                Ciphertext result;
                context.evaluator().square(encrypted, result, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "square/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "square/correct");
            }
            { // square_inplace
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                context.evaluator().square_inplace(encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "square_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "square_inplace/correct");
            }
            { // square_new
                Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
                Ciphertext result = context.evaluator().square_new(encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(result.pool(), pool), "multiply_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(result, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "multiply_new/correct");
            }
        }

        { // keyswitching
            MemoryPoolHandle context_pool_other = create_new_memory_pool();
            KeyGenerator keygen_other = KeyGenerator(context.context(), context_pool_other);
            SecretKey secret_key_other = keygen_other.secret_key().clone(context_pool_other);
            Encryptor encryptor_other = Encryptor(context.context());
            encryptor_other.set_secret_key(secret_key_other, context_pool_other);
            KSwitchKeys kswitch_key = context.key_generator().create_keyswitching_key(secret_key_other, false, context_pool_other);
            if (device && check_pool && !shared_pool) context_pool_other->deny();

            MemoryPoolHandle pool = create_new_memory_pool();
            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale, pool);
            Ciphertext encrypted_original = encryptor_other.encrypt_symmetric_new(encoded, false, nullptr, pool);

            { // keyswitch
                Ciphertext encrypted;
                context.evaluator().apply_keyswitching(encrypted_original, kswitch_key, encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "keyswitch/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "keyswitch/correct");
            }
            { // keyswitch_inplace
                Ciphertext encrypted = encrypted_original.clone(pool);
                context.evaluator().apply_keyswitching_inplace(encrypted, kswitch_key, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "keyswitch_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "keyswitch_inplace/correct");
            }
            { // keyswitching_new
                Ciphertext encrypted = context.evaluator().apply_keyswitching_new(encrypted_original, kswitch_key, pool);
                IF_FALSE_PRINT_RETURN(good_pool(encrypted.pool(), pool), "keyswitch_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(encrypted, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "keyswitch_new/correct");
            }
        }

        { // relinearize
            MemoryPoolHandle pool = create_new_memory_pool();
            auto relin_keys = context.key_generator().create_relin_keys(false, 2, pool);
            IF_FALSE_PRINT_RETURN(good_pool(relin_keys.pool(), pool), "relinearize/keys-pool");
            GeneralVector message = context.random_simd_full();
            GeneralVector truth = context.mul(message, message);
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale, pool);
            Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
            context.evaluator().square_inplace(encrypted, pool);
            { // relinearize
                Ciphertext relinearized;
                context.evaluator().relinearize(encrypted, relin_keys, relinearized, pool);
                IF_FALSE_PRINT_RETURN(good_pool(relinearized.pool(), pool), "relinearize/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(relinearized, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "relinearize/correct");
            }
            { // relinearize_inplace
                Ciphertext relinearized = encrypted.clone(pool);
                context.evaluator().relinearize_inplace(relinearized, relin_keys, pool);
                IF_FALSE_PRINT_RETURN(good_pool(relinearized.pool(), pool), "relinearize_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(relinearized, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "relinearize_inplace/correct");
            }
            { // relinearize_new
                Ciphertext relinearized = context.evaluator().relinearize_new(encrypted, relin_keys, pool);
                IF_FALSE_PRINT_RETURN(good_pool(relinearized.pool(), pool), "relinearize_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(relinearized, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "relinearize_new/correct");
            }
        }

        { // mod_switch_to_next
            MemoryPoolHandle pool = create_new_memory_pool();
            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale, pool);
            Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
            { // mod_switch_to_next
                Ciphertext mod_switched;
                context.evaluator().mod_switch_to_next(encrypted, mod_switched, pool);
                IF_FALSE_PRINT_RETURN(good_pool(mod_switched.pool(), pool), "mod_switch_to_next/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(mod_switched, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "mod_switch_to_next/correct");
            }
            { // mod_switch_to_next_inplace
                Ciphertext mod_switched = encrypted.clone(pool);
                context.evaluator().mod_switch_to_next_inplace(mod_switched, pool);
                IF_FALSE_PRINT_RETURN(good_pool(mod_switched.pool(), pool), "mod_switch_to_next_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(mod_switched, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "mod_switch_to_next_inplace/correct");
            }
            { // mod_switch_to_next_new
                Ciphertext mod_switched = context.evaluator().mod_switch_to_next_new(encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(mod_switched.pool(), pool), "mod_switch_to_next_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(mod_switched, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "mod_switch_to_next_new/correct");
            }
        }

        if (scheme == SchemeType::CKKS) { // mod_switch_plain_to_next
            MemoryPoolHandle pool = create_new_memory_pool();
            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale, pool);
            { // mod_switch_plain_to_next
                Plaintext mod_switched;
                context.evaluator().mod_switch_plain_to_next(encoded, mod_switched, pool);
                IF_FALSE_PRINT_RETURN(good_pool(mod_switched.pool(), pool), "mod_switch_plain_to_next/pool");
                GeneralVector decoded = context.encoder().decode_simd(mod_switched, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "mod_switch_plain_to_next/correct");
            }
            { // mod_switch_plain_to_next_inplace
                Plaintext mod_switched = encoded.clone(pool);
                context.evaluator().mod_switch_plain_to_next_inplace(mod_switched, pool);
                IF_FALSE_PRINT_RETURN(good_pool(mod_switched.pool(), pool), "mod_switch_plain_to_next_inplace/pool");
                GeneralVector decoded = context.encoder().decode_simd(mod_switched, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "mod_switch_plain_to_next_inplace/correct");
            }
            { // mod_switch_plain_to_next_new
                Plaintext mod_switched = context.evaluator().mod_switch_plain_to_next_new(encoded, pool);
                IF_FALSE_PRINT_RETURN(good_pool(mod_switched.pool(), pool), "mod_switch_plain_to_next_new/pool");
                GeneralVector decoded = context.encoder().decode_simd(mod_switched, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "mod_switch_plain_to_next_new/correct");
            }
        }

        if (scheme == SchemeType::CKKS) { // rescale_to_next
            MemoryPoolHandle pool = create_new_memory_pool();
            GeneralVector message = context.random_simd_full();
            const EncryptionParameters& parms = context.params_host();
            auto coeff_modulus = parms.coeff_modulus();
            double expanded_scale = scale * coeff_modulus[coeff_modulus.size() - 2].value();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, expanded_scale, pool);
            Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
            { // rescale_to_next
                Ciphertext rescaled;
                context.evaluator().rescale_to_next(encrypted, rescaled, pool);
                IF_FALSE_PRINT_RETURN(good_pool(rescaled.pool(), pool), "rescale_to_next/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(rescaled, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "rescale_to_next/correct");
            }
            { // rescale_to_next_inplace
                Ciphertext rescaled = encrypted.clone(pool);
                context.evaluator().rescale_to_next_inplace(rescaled, pool);
                IF_FALSE_PRINT_RETURN(good_pool(rescaled.pool(), pool), "rescale_to_next_inplace/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(rescaled, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "rescale_to_next_inplace/correct");
            }
            { // rescale_to_next_new
                Ciphertext rescaled = context.evaluator().rescale_to_next_new(encrypted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(rescaled.pool(), pool), "rescale_to_next_new/pool");
                Plaintext decrypted = context.decryptor().decrypt_new(rescaled, pool);
                GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "rescale_to_next_new/correct");
            }
        }

        { // rotate
            MemoryPoolHandle pool = create_new_memory_pool();
            GeneralVector message = context.random_simd_full();
            Plaintext encoded = context.encoder().encode_simd(message, std::nullopt, scale, pool);
            Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);
            GaloisKeys gal_keys = context.key_generator().create_galois_keys(false, pool);
            IF_FALSE_PRINT_RETURN(good_pool(gal_keys.pool(), pool), "rotate/keys-pool");

            if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) {
                size_t steps = 7;
                GeneralVector truth = message.rotate(steps);
                { // rotate_rows
                    Ciphertext rotated;
                    context.evaluator().rotate_rows(encrypted, steps, gal_keys, rotated, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_rows/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_rows/correct");
                }
                { // rotate_rows_inplace
                    Ciphertext rotated = encrypted.clone(pool);
                    context.evaluator().rotate_rows_inplace(rotated, steps, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_rows_inplace/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_rows_inplace/correct");
                }
                { // rotate_rows_new
                    Ciphertext rotated = context.evaluator().rotate_rows_new(encrypted, steps, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_rows_new/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_rows_new/correct");
                }
                truth = message.conjugate();
                { // rotate_columns
                    Ciphertext rotated;
                    context.evaluator().rotate_columns(encrypted, gal_keys, rotated, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_columns/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_columns/correct");
                }
                { // rotate_columns_inplace
                    Ciphertext rotated = encrypted.clone(pool);
                    context.evaluator().rotate_columns_inplace(rotated, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_columns_inplace/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_columns_inplace/correct");
                }
                { // rotate_columns_new
                    Ciphertext rotated = context.evaluator().rotate_columns_new(encrypted, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_columns_new/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_columns_new/correct");
                }
            }

            if (scheme == SchemeType::CKKS) {
                size_t steps = 7;
                GeneralVector truth = message.rotate(steps);
                { // rotate_vector
                    Ciphertext rotated;
                    context.evaluator().rotate_vector(encrypted, steps, gal_keys, rotated, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_vector/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_vector/correct");
                }
                { // rotate_vector_inplace
                    Ciphertext rotated = encrypted.clone(pool);
                    context.evaluator().rotate_vector_inplace(rotated, steps, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_vector_inplace/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_vector_inplace/correct");
                }
                { // rotate_vector_new
                    Ciphertext rotated = context.evaluator().rotate_vector_new(encrypted, steps, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(rotated.pool(), pool), "rotate_vector_new/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(rotated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "rotate_vector_new/correct");
                }
                truth = message.conjugate();
                { // conjugate
                    Ciphertext conjugated;
                    context.evaluator().complex_conjugate(encrypted, gal_keys, conjugated, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(conjugated.pool(), pool), "conjugate/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(conjugated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "conjugate/correct");
                }
                { // conjugate_inplace
                    Ciphertext conjugated = encrypted.clone(pool);
                    context.evaluator().complex_conjugate_inplace(conjugated, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(conjugated.pool(), pool), "conjugate_inplace/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(conjugated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "conjugate_inplace/correct");
                }
                { // conjugate_new
                    Ciphertext conjugated = context.evaluator().complex_conjugate_new(encrypted, gal_keys, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(conjugated.pool(), pool), "conjugate_new/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(conjugated, pool);
                    GeneralVector decoded = context.encoder().decode_simd(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(truth, decoded), "conjugate_new/correct");
                }
            }
        }

        { // lwe related
            MemoryPoolHandle pool = create_new_memory_pool();
            GeneralVector message = context.random_polynomial_full();
            Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale, pool);
            Ciphertext encrypted = context.encryptor().encrypt_asymmetric_new(encoded, nullptr, pool);

            { // extract_lwe_new
                size_t term = 3;
                LWECiphertext extracted = context.evaluator().extract_lwe_new(encrypted, term, pool);
                IF_FALSE_PRINT_RETURN(good_pool(extracted.pool(), pool), "extract_lwe_new/pool");
                Ciphertext assembled = context.evaluator().assemble_lwe_new(extracted, pool);
                IF_FALSE_PRINT_RETURN(good_pool(assembled.pool(), pool), "assemble_lwe_new/pool");
                if (encrypted.is_ntt_form()) {
                    context.evaluator().transform_to_ntt_inplace(assembled);
                }
                Plaintext decrypted = context.decryptor().decrypt_new(assembled, pool);
                GeneralVector decoded = context.encoder().decode_polynomial(decrypted, pool);
                IF_FALSE_PRINT_RETURN(context.near_equal(message.element(term), decoded.element(0)), "extract_lwe_new/correct");
            }
            { // pack lwes
                size_t terms = 7;
                size_t terms_upper = 8;
                size_t n = context.coeff_count();
                if (n >= terms_upper) {
                    GaloisKeys automorphism_key = context.key_generator().create_automorphism_keys(false, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(automorphism_key.pool(), pool), "pack_lwes/keys-pool");
                    vector<LWECiphertext> lwes;
                    for (size_t i = 0; i < terms; i++) {
                        lwes.push_back(context.evaluator().extract_lwe_new(encrypted, i * (n / terms_upper), pool));
                    }
                    for (size_t i = 0; i < n; i++) {
                        if ((i % (n / terms_upper) == 0) and (i / (n / terms_upper) < terms)) continue;
                        if (message.is_integers()) message.integers()[i] = 0;
                        else message.doubles()[i] = 0;
                    }
                    Ciphertext assembled = context.evaluator().pack_lwe_ciphertexts_new(lwes, automorphism_key, pool);
                    IF_FALSE_PRINT_RETURN(good_pool(assembled.pool(), pool), "pack_lwes/pool");
                    Plaintext decrypted = context.decryptor().decrypt_new(assembled, pool);
                    GeneralVector decoded = context.encoder().decode_polynomial(decrypted, pool);
                    IF_FALSE_PRINT_RETURN(context.near_equal(message, decoded), "pack_lwes/correct");
                }
            }

        }

        return true;

    }

    void test_shared_pool(size_t threads, bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, 
            bool expand_mod_chain, uint64_t seed, double input_max = 0, double scale = 0, double tolerance = 1e-4,
            bool to_device_after_keygeneration = false, bool use_special_prime_for_encryption = false
    ) {

        GeneralHeContext context(
            device, scheme, n, log_t, log_qi, 
            expand_mod_chain, seed, input_max, scale, tolerance, 
            to_device_after_keygeneration, use_special_prime_for_encryption
        );

        auto test_thread = [&context](int thread) {
            return test_troublesome_pools(context, 0, thread, false, true);
        };

        utils::stream_sync();
        vector<std::future<bool>> thread_instances;
        for (size_t i = 0; i < threads; i++) {
            thread_instances.push_back(std::async(test_thread, i));
        }

        for (size_t i = 0; i < threads; i++) {
            ASSERT_TRUE(thread_instances[i].get());
        }

    }
    
    static constexpr size_t SHARED_POOL_THREADS = 16;

    TEST(MultithreadTest, HostBFVSharedPool) {
        test_shared_pool(4, false, 
            SchemeType::BFV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, DeviceBFVSharedPool) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_shared_pool(SHARED_POOL_THREADS, true, 
            SchemeType::BFV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
        MemoryPool::Destroy();
    }
    TEST(MultithreadTest, HostBGVSharedPool) {
        test_shared_pool(4, false, 
            SchemeType::BGV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, DeviceBGVSharedPool) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_shared_pool(SHARED_POOL_THREADS, true, 
            SchemeType::BGV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
        MemoryPool::Destroy();
    }
    TEST(MultithreadTest, HostCKKSSharedPool) {
        test_shared_pool(4, false,
            SchemeType::CKKS, 32, 0, 
            { 60, 40, 40, 60 }, true, 0x123, 
            10, 1ull<<20, 1e-2
        );
    }
    TEST(MultithreadTest, DeviceCKKSSharedPool) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_shared_pool(SHARED_POOL_THREADS, true,
            SchemeType::CKKS, 32, 0, 
            { 60, 40, 40, 60 }, true, 0x123, 
            10, 1ull<<20, 1e-2
        );
        MemoryPool::Destroy();
    }


    void test_shared_context_multiple_pools(size_t threads, bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, 
            bool expand_mod_chain, uint64_t seed, double input_max = 0, double scale = 0, double tolerance = 1e-4,
            bool to_device_after_keygeneration = false, bool use_special_prime_for_encryption = false,
            MemoryPoolHandle context_pool = MemoryPool::GlobalPool()
    ) {

        GeneralHeContext context(
            device, scheme, n, log_t, log_qi, 
            expand_mod_chain, seed, input_max, scale, tolerance, 
            to_device_after_keygeneration, use_special_prime_for_encryption, 
            context_pool
        );
        
        // create relin keys so that skarray is expanded
        context.key_generator().create_relin_keys(false, 2, context_pool);
        // decrypt a 3-sized ciphertext so that skarray is expanded
        auto c = context.encryptor().encrypt_zero_asymmetric_new(std::nullopt, nullptr, context_pool);
        context.evaluator().square_inplace(c, context_pool);
        context.decryptor().decrypt_new(c, context_pool);

        if (device) {
            context_pool->deny();
            utils::MemoryPool::GlobalPool()->deny(false);
        }

        auto test_thread = [
            context_pool, scheme, &context
        ](int thread) {
            return test_troublesome_pools(context, 0, thread);
        };

        utils::stream_sync();
        vector<std::future<bool>> thread_instances;
        for (size_t i = 0; i < threads; i++) {
            thread_instances.push_back(std::async(test_thread, i));
        }

        for (size_t i = 0; i < threads; i++) {
            ASSERT_TRUE(thread_instances[i].get());
        }

    }

    static constexpr size_t DEVICE_THREADS = 4;

    TEST(MultithreadTest, HostBFVSharedContextMultiPools) {
        test_shared_context_multiple_pools(4, false, 
            SchemeType::BFV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, DeviceBFVSharedContextMultiPools) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_shared_context_multiple_pools(DEVICE_THREADS, true, 
            SchemeType::BFV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
        MemoryPool::Destroy();
    }
    TEST(MultithreadTest, HostBGVSharedContextMultiPools) {
        test_shared_context_multiple_pools(4, false, 
            SchemeType::BGV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, DeviceBGVSharedContextMultiPools) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_shared_context_multiple_pools(DEVICE_THREADS, true, 
            SchemeType::BGV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
        MemoryPool::Destroy();
    }
    TEST(MultithreadTest, HostCKKSSharedContextMultiPools) {
        test_shared_context_multiple_pools(4, false,
            SchemeType::CKKS, 32, 0, 
            { 60, 40, 40, 60 }, true, 0x123, 
            10, 1ull<<20, 1e-2
        );
    }
    TEST(MultithreadTest, DeviceCKKSSharedContextMultiPools) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        test_shared_context_multiple_pools(DEVICE_THREADS, true,
            SchemeType::CKKS, 32, 0, 
            { 60, 40, 40, 60 }, true, 0x123, 
            10, 1ull<<20, 1e-2
        );
        MemoryPool::Destroy();
    }

    
    void test_multi_devices(size_t threads, size_t device_count, bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, 
            bool expand_mod_chain, uint64_t seed, double input_max = 0, double scale = 0, double tolerance = 1e-4,
            bool to_device_after_keygeneration = false, bool use_special_prime_for_encryption = false
    ) {

        std::vector<std::shared_ptr<GeneralHeContext>> contexts;

        for (size_t i = 0; i < device_count; i++) {
            MemoryPoolHandle context_pool = device ? MemoryPool::create(i) : nullptr;
            auto context = std::make_shared<GeneralHeContext>(
                device, scheme, n, log_t, log_qi, 
                expand_mod_chain, seed, input_max, scale, tolerance, 
                to_device_after_keygeneration, use_special_prime_for_encryption, 
                context_pool
            );
            contexts.push_back(context);
            
            // create relin keys so that skarray is expanded
            context->key_generator().create_relin_keys(false, 2, context_pool);
            // decrypt a 3-sized ciphertext so that skarray is expanded
            auto c = context->encryptor().encrypt_zero_asymmetric_new(std::nullopt, nullptr, context_pool);
            context->evaluator().square_inplace(c, context_pool);
            context->decryptor().decrypt_new(c, context_pool);

            if (device) {
                context_pool->deny();
                utils::MemoryPool::GlobalPool()->deny();
            }
        }

        utils::stream_sync();
        auto test_thread = [=](int thread, std::shared_ptr<GeneralHeContext> context) {
            size_t device_index = thread % device_count;
            return test_troublesome_pools(*context, device_index, thread);
        };

        vector<std::future<bool>> thread_instances;
        for (size_t i = 0; i < threads; i++) {
            thread_instances.push_back(std::async(test_thread, i, contexts[i % device_count]));
        }

        for (size_t i = 0; i < threads; i++) {
            ASSERT_TRUE(thread_instances[i].get());
        }

    }

    static constexpr size_t DEVICE_THREADS_MULTIPLE_OF_DEVICES = 4;

    TEST(MultithreadTest, HostBFVMultiDevices) {
        test_multi_devices(8, 4, false, 
            SchemeType::BFV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, DeviceBFVMultiDevices) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        int device_count;
        cudaError_t success = cudaGetDeviceCount(&device_count);
        if (success != cudaSuccess || device_count <= 1) {
            GTEST_SKIP_("No multiple devices available");
        }
        test_multi_devices(device_count * DEVICE_THREADS_MULTIPLE_OF_DEVICES + 1, device_count, true, 
            SchemeType::BFV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, HostBGVMultiDevices) {
        test_multi_devices(8, 4, false, 
            SchemeType::BGV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, DeviceBGVMultiDevices) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        int device_count;
        cudaError_t success = cudaGetDeviceCount(&device_count);
        if (success != cudaSuccess || device_count <= 1) {
            GTEST_SKIP_("No multiple devices available");
        }
        test_multi_devices(device_count * DEVICE_THREADS_MULTIPLE_OF_DEVICES + 1, device_count, true, 
            SchemeType::BGV, 32, 35, 
            { 60, 40, 40, 60 }, true, 0x123, 0
        );
    }
    TEST(MultithreadTest, HostCKKSMultiDevices) {
        test_multi_devices(8, 4, false,
            SchemeType::CKKS, 32, 0, 
            { 60, 40, 40, 60 }, true, 0x123, 
            10, 1ull<<20, 1e-2
        );
    }
    TEST(MultithreadTest, DeviceCKKSMultiDevices) {
        SKIP_WHEN_NO_CUDA_DEVICE;
        int device_count;
        cudaError_t success = cudaGetDeviceCount(&device_count);
        if (success != cudaSuccess || device_count <= 1) {
            GTEST_SKIP_("No multiple devices available");
        }
        test_multi_devices(device_count * DEVICE_THREADS_MULTIPLE_OF_DEVICES + 1, device_count, true,
            SchemeType::CKKS, 32, 0, 
            { 60, 40, 40, 60 }, true, 0x123, 
            10, 1ull<<20, 1e-2
        );
    }

#undef IF_FALSE_RETURN
#undef IF_FALSE_PRINT_RETURN
#undef CHECKPOINT

}