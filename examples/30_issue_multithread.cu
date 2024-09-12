
#include <future>
#include <iostream>
#include "examples.h"

#include "../src/utils/box.h"

constexpr int N = 1<<10;
constexpr int K = 64;
using namespace std;

using namespace troy::utils;

__global__ void repeat_set_value(uint64_t* target, size_t n, uint64_t value, size_t repeat) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t r = 0; r < repeat; r++) {
        for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
            target[i] = value;
        }
    }
}

__global__ void set_value(uint64_t* target, size_t n, uint64_t value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        target[i] = value;
    }
}

__global__ void wait_set_value(uint64_t* target, size_t n, uint64_t value, size_t wait_cycles) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    clock_t start = clock();
    while (start + static_cast<clock_t>(wait_cycles) > clock()) {
        // busy wait
    }
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        target[i] = value;
    }
}


// This function creates an array on device memory. It calls a kernel `wait_set_value` that waits some time before setting
// the array to 2. 
Array<uint64_t> thread_a(bool sync_after_kernel) {
    Array<uint64_t> arr = Array<uint64_t>::create_uninitialized(N * K, true);
    wait_set_value<<<1, K>>>(arr.raw_pointer(), N * K, 2, 100000000);
    if (sync_after_kernel) cudaStreamSynchronize(0);
    return arr;
}

// This function accepts an array and then set it to 1 immediately.
void thread_b(Array<uint64_t>& arr) {
    set_value<<<1, K>>>(arr.raw_pointer(), N * K, 1);
    
    // make sure all the kernels are finished
    cudaDeviceSynchronize();

    // check the value
    arr.to_host_inplace();
    bool flag = false;
    for (size_t i = 0; i < N * K; i++) {
        if (arr[i] != 1) {
            std::cerr << "Error at " << i << " " << arr[i] << std::endl;
            flag = true;
            break;
        }
    }
    if (!flag) {
        std::cerr << "No error" << std::endl;
    }
}

void example_issue_multithread() {

    print_example_banner("Issue of Multithread");

    if (troy::utils::device_count() == 0) {
        std::cerr << "No GPU device found" << std::endl;
        return;
    }

    // warm up.
    uint64_t* arr1;
    Array<uint64_t> arr1_array = Array<uint64_t>::create_uninitialized(N * K, true);
    arr1 = arr1_array.raw_pointer();
    set_value<<<1, K>>>(arr1, N * K, 0);
    cudaStreamSynchronize(0);
    
    // First we can see that, if thread a is not synced after kernel call, there will be problem.
    std::future<Array<uint64_t>> t1 = std::async(std::launch::async, thread_a, false);
    Array<uint64_t> arr = t1.get();
    thread_b(arr);
    
    cudaDeviceSynchronize();

    // If we sync after kernel call, the data racing is solved.
    t1 = std::async(std::launch::async, thread_a, true);
    arr = t1.get();
    thread_b(arr);
    
    cudaDeviceSynchronize();

}