#include "test_adv.h"
#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <thread>
#include <future>

namespace temp {

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

    void test_single_pool_multi_thread(const GeneralHeContext& context, size_t threads, size_t repeat) {

        uint64_t t = context.t();
        double scale = context.scale();
        double tolerance = context.tolerance();

        GeneralVector m = context.random_simd_full();
        Plaintext p = context.encoder().encode_simd(m, std::nullopt, scale);
        std::cout << "p = " << p.data().slice(0, 4) << std::endl;

        auto test_thread = [t, scale, repeat, tolerance, &context, &m, &p](int thread) {

            for (size_t rep = 0; rep < repeat; rep++) {
                bool succ = true;

                Array<uint64_t> copied(32, true); copied.slice(0, p.coeff_count()).copy_from_slice(p.const_poly());

                {
                    Array<uint64_t> h = Array<uint64_t>::create_and_copy_from_slice(copied.const_slice(0, 1), false);
                    if (h[0] != 10572) {
                        std::cerr << "ckpt 3 h[0] = " << h[0] << std::endl;
                        succ = false;
                    }
                }

                auto decoded = context.encoder().batch().decode_new(p);
                if (!succ) {
                    return false;
                }
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
    
    TEST(Temp, Temp) {
        GeneralHeContext ghe(true, SchemeType::BFV, 32, 20, { 60, 40, 40, 60 }, false, 0x123, 0);
        test_single_pool_multi_thread(ghe, 64, 4);
        utils::MemoryPool::Destroy();
    }

}