#include "../src/troy.h"

using namespace troy;
using namespace troy::utils;
using namespace troy::bench;

/*
struct BinaryAdd {
    static __device__ __forceinline__ uint64_t run(uint64_t a, uint64_t b) {
        return a + b;
    }
};

template <typename F>
__global__ void binary_operator_block(const uint64_t* a, const uint64_t* b, size_t n, uint64_t* result) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        result[i] = F::run(a[i], b[i]);
    }
}

__global__ void simple_add(const uint64_t* a, const uint64_t* b, size_t n, uint64_t* result) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}

__global__ void simple_add_block(const uint64_t* a, const uint64_t* b, size_t n, uint64_t* result) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        result[i] = a[i] + b[i];
    }
}

void simple_add_host(const uint64_t* a, const uint64_t* b, size_t n, uint64_t* result) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void test2() {
    
    Timer timer;
    constexpr size_t n = 16384;
    constexpr size_t repeat = 10000;

    Array<uint64_t> arr(n, false);
    Array<uint64_t> result(n, false);

    for (size_t i = 0; i < n; i++) {
        arr[i] = i;
    }

    auto th_host = timer.register_timer("host");
    for (size_t i = 0; i < repeat; i++) {
        timer.tick(th_host);
        simple_add_host(arr.raw_pointer(), arr.raw_pointer(), n, result.raw_pointer());
        timer.tock(th_host);
    }

    arr.to_device_inplace();
    result.to_device_inplace();

    auto th_device = timer.register_timer("device");
    auto th_device_block = timer.register_timer("device_block");
    auto th_binary_operator = timer.register_timer("binary_operator");
    for (size_t i = 0; i < repeat; i++) {

        timer.tick(th_device_block);
        simple_add_block<<<1, 256>>>(arr.raw_pointer(), arr.raw_pointer(), n, result.raw_pointer());
        timer.tock(th_device_block);

        size_t block_count = ceil_div(n, KERNEL_THREAD_COUNT);
        timer.tick(th_device);
        simple_add<<<block_count, KERNEL_THREAD_COUNT>>>(arr.raw_pointer(), arr.raw_pointer(), n, result.raw_pointer());
        timer.tock(th_device);

        timer.tick(th_binary_operator);
        binary_operator_block<BinaryAdd><<<block_count, KERNEL_THREAD_COUNT>>>(arr.raw_pointer(), arr.raw_pointer(), n, result.raw_pointer());
        timer.tock(th_binary_operator);

    }

    timer.print();

}
*/

void test1(size_t n, size_t repeat) {
    
    Timer timer;

    std::cout << "n: " << n << std::endl;
    std::cout << "repeat: " << repeat << std::endl;
    
    auto th_host = timer.register_timer("host");
    auto th_device = timer.register_timer("device");
    
    EncryptionParameters params(SchemeType::BFV);
    params.set_poly_modulus_degree(n);
    params.set_plain_modulus(PlainModulus::batching(n, 30));
    params.set_coeff_modulus(CoeffModulus::create(n, {50, 50, 50}));
    HeContextPointer context = HeContext::create(params, true, SecurityLevel::Nil);
    BatchEncoder encoder(context);

    {
        auto th = th_host;
        KeyGenerator keygen(context);
        Encryptor encryptor(context); encryptor.set_public_key(keygen.create_public_key(false));
        Evaluator evaluator(context);
        Ciphertext c = encryptor.encrypt_asymmetric_new(encoder.encode_new({ 1, 2, 3, 4, 5 }));
        Plaintext p = encoder.encode_new({ 1, 2, 3, 4, 5 });
        Ciphertext r;

        std::cout << "running host ...\n";
        const size_t warm_up = 10;
        
        for (size_t i = 0; i < repeat + warm_up; i++) {
            if (i == warm_up) {
                timer.tick(th);
            }
            evaluator.multiply_plain(c, p, r);
            if (i == repeat + warm_up - 1) {
                timer.tock(th);
            }
        }
    }

    context->to_device_inplace();
    encoder.to_device_inplace();

    {
        auto th = th_device;
        KeyGenerator keygen(context);
        Encryptor encryptor(context); encryptor.set_public_key(keygen.create_public_key(false));
        Evaluator evaluator(context);
        Ciphertext c = encryptor.encrypt_asymmetric_new(encoder.encode_new({ 1, 2, 3, 4, 5 }));
        Plaintext p = encoder.encode_new({ 1, 2, 3, 4, 5 });
        Ciphertext r = c.clone();

        std::cout << "running device ...\n";
        const size_t warm_up = 10;
        
        for (size_t i = 0; i < repeat + warm_up; i++) {
            if (i == warm_up) {
                cudaStreamSynchronize(0);
                timer.tick(th);
            }
            evaluator.multiply_plain(c, p, r);
            if (i == repeat + warm_up - 1) {
                cudaStreamSynchronize(0);
                timer.tock(th);
            }
        }
    }
    
    timer.print_divided(repeat);

    auto durations = timer.get();
    float acc = static_cast<double>(durations[th_host].count()) / static_cast<double>(durations[th_device].count());
    std::cout << "host/device: " << acc << std::endl;

}

int main(int, char** argv) {
    int n = std::stoi(argv[1]);
    int repeat = std::stoi(argv[2]);
    test1(n, repeat);
    return 0;
}