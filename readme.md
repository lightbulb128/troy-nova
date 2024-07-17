# Troy-Nova

The new implementation of Troy, a CUDA based GPU parallelized implementation of RLWE homomorphic encryption schemes. We support RNS version of BFV, CKKS and BGV schemes. The implementation itself is inspired by the [Microsoft SEAL](https://github.com/microsoft/SEAL) library, so the interfaces are very similar to theirs. 

We also include some utilities for privacy computing, including the matrix multiplication (from [BumbleBee](https://eprint.iacr.org/2023/1678)), 2d convolution (from [Cheetah](https://www.usenix.org/system/files/sec22-huang-zhicong.pdf)) and LWE-ciphertext extraction and packing (from [Chen et al.](https://eprint.iacr.org/2020/015.pdf)).

# Build

* **Requirements**: CUDA 11.7 and CMake 3.18.
* **Tested environment**: Ubuntu 20.04 with CUDA 12.4, NVIDIA A100 and RTX 4090. g++ and gcc 11.4.

### Build C++/CUDA.

```
mkdir build
cd build
cmake ..
make troy
```

Note: You could set the CUDA architecture to suit your graphic card, by setting "CMAKE_CUDA_ARCHITECTURES" variable when calling cmake. For example, `cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;89"`

### Build python bindings

```
mkdir -p build     # ensure build folder exists.
cd pybind
bash develop.sh
```

You will get a `pytroy*.whl` which could be installed.

### Building examples

`cmake` with `TROY_EXAMPLES` set to true, and then make will give the examples in `build/examples`.

```
mkdir -p build
cd build
cmake .. -DTROY_EXAMPLES=ON
make troyexamples
./examples/troyexamples
```

## Testing and Benchmark

```
cd build
cmake .. -DTROY_TEST=ON -DTROY_BENCH=ON
make
cd test
ctest
./troybench
```

# Quickstart example

See `examples/99_quickstart.cu`.

```c++
#include "troy/troy.h"

using namespace troy;

int main() {
    
    // Setup encryption parameters.
    EncryptionParameters params(SchemeType::BFV);
    params.set_poly_modulus_degree(8192);
    params.set_coeff_modulus(CoeffModulus::create(8192, { 40, 40, 40 }));
    params.set_plain_modulus(PlainModulus::batching(8192, 20));

    // Create context and encoder
    HeContextPointer context = HeContext::create(params, true, SecurityLevel::Classical128);
    BatchEncoder encoder(context);

    // Convey them to the device memory.
    // The encoder must be conveyed to the device memory after creating it from a host-memory context.
    // i.e. you cannot create an encoder directly from a device-memory context.
    context->to_device_inplace();
    encoder.to_device_inplace();

    // Other utilities could directly be constructed from device-memory context.
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.create_public_key(false);
    Encryptor encryptor(context); encryptor.set_public_key(public_key);
    Decryptor decryptor(context, keygen.secret_key());
    Evaluator evaluator(context);
    // Alternatively, you can create all of these (keygen, encryptor, etc.) 
    // on host memory and then convey them all to device memory at once.

    // Create plaintexts
    std::vector<uint64_t> message1 = { 1, 2, 3, 4 };
    Plaintext plain1 = encoder.encode_new(message1);
    std::vector<uint64_t> message2 = { 5, 6, 7, 8 };
    Plaintext plain2 = encoder.encode_new(message2);

    // Encrypt. Since we only set the public key, we can only use asymmetric encryption.
    Ciphertext encrypted1 = encryptor.encrypt_asymmetric_new(plain1);
    Ciphertext encrypted2 = encryptor.encrypt_asymmetric_new(plain2);

    // Add
    Ciphertext encrypted_sum = evaluator.add_new(encrypted1, encrypted2);

    // Decrypt and decode
    Plaintext decrypted_sum = decryptor.decrypt_new(encrypted_sum);
    std::vector<uint64_t> result = encoder.decode_new(decrypted_sum);

    // Check good?
    result.resize(message1.size());
    if (result == std::vector<uint64_t>({ 6, 8, 10, 12 })) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }

    // Destroy global memory pool before the program exits.
    utils::MemoryPool::Destroy();

    return 0;

}
```

# Memory Pool

Troy uses memory pools to manage device memory, and some APIs take a memory pool handle as an argument. By default, you can omit this argument and Troy will use a default global memory pool for all those operations. This default memory pool will be created on device 0 (the first visible CUDA device). All memory pools are thread safe so a lazy user simply use defaults and be free of any concerns of memory pools. 

Note that if you use the default memory pool, it is recommented that you call `MemoryPool::Destroy` before the program exits to safely release the static memory pool. But not doing it may just still go smoothly.

You can disable the management of device memory by memory pools by applying `TROY_MEMORY_POOL=OFF` to cmake. If so, although memory pools can still be created, they do not manage the memory but simply call cudaMalloc and cudaFree whenever their methods are called. When memory pool is enabled, some users report unexpected exceptions complaining `"[MemoryPool::get] The singleton has been destroyed."` when using the library. One could check if `MemoryPool::Destroy()` has been called prematurely in your program to locate the problem, or one could simply provide a `TROY_MEMORY_POOL_UNSAFE=OFF` to try to avoid it. This `TROY_MEMORY_POOL_UNSAFE` is an experimental hacking solution (because I have not reproduced the error on my machine yet 😢), so if you still get the errors please file an issue.

## Multithreading with Memory Pools

Multithreaded programs **should** use different memory pools for each thread, since I have observed some issues when all threads use the same memory pool. You can create the context in the main thread with a single memory pool, but when you conduct HE computations on multiple threads, you should provide unique memory pools for each thread. Alternatively, you could turn on `TROY_STREAM_SYNC_AFTER_KERNEL_CALLS` in cmake, this guarantees any memory pool shared by multiple threads won't lead to racing conditions, but it could result in some performance drop.

Furthermore, if you wish to use **multiple GPUs, you must** create multiple memory pools to handle the memory for each device. You can create new instances of memory pools by calling `MemoryPool::Create`, which takes the device index as an argument. See [`memory_pool.h`](src/utils/memory_pool.h). For the usage multiple memory pools, see the example at `examples/20_memory_pools.cu`.


# Contribute
Feel free to fork / pull request.
Please cite this repository if you use it in your work.