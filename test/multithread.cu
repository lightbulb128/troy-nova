#include "test_adv.cuh"
#include <gtest/gtest.h>
#include <thread>
#include <future>

namespace multithread {

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
                ASSERT_TRUE(truth.near_equal(result, tolerance));

                GeneralVector message3 = context.random_simd_full();
                Plaintext encoded3 = context.encoder().encode_simd(message3, std::nullopt, relined.scale());
                Ciphertext encrypted3 = context.encryptor().encrypt_asymmetric_new(encoded3);
                multiplied = context.evaluator().multiply_new(multiplied, encrypted3);
                relined = context.evaluator().relinearize_new(multiplied, relin_keys);
                decrypted = context.decryptor().decrypt_new(relined);
                result = context.encoder().decode_simd(decrypted);
                truth = truth.mul(message3, t);
                ASSERT_TRUE(truth.near_equal(result, tolerance));

            }

        };

        vector<std::thread> thread_instances;
        for (size_t i = 0; i < threads; i++) {
            thread_instances.push_back(std::thread(test_thread, i));
        }

        for (size_t i = 0; i < threads; i++) {
            thread_instances[i].join();
        }

    }    
    
    TEST(MultithreadTest, HostSinglePoolMultiThread) {
        GeneralHeContext ghe(false, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
        test_single_pool_multi_thread(ghe, 64, 4);
    }
    TEST(MultithreadTest, DeviceSinglePoolMultiThread) {
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
        test_single_pool_multi_thread(ghe, 64, 4);
        utils::MemoryPool::Destroy();
    }

}