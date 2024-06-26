#include "test_adv.cuh"
#include <gtest/gtest.h>
#include <thread>
#include <future>

namespace multithread {

    using namespace std;
    using namespace troy;
    using troy::utils::Array;

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

}