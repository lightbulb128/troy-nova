# Troy-Nova

The new implementation of Troy, a CUDA based GPU parallelized implementation of RLWE homomorphic encryption schemes. We support RNS version of BFV, CKKS and BGV schemes. The implementation itself is inspired by the [Microsoft SEAL](https://github.com/microsoft/SEAL) library, so the interfaces are very similar to theirs. 

We also include some utilities for privacy computing, including the matrix multiplication (from [BumbleBee](https://eprint.iacr.org/2023/1678)), 2d convolution (from [Cheetah](https://www.usenix.org/system/files/sec22-huang-zhicong.pdf)) and LWE-ciphertext extraction and packing (from [Chen et al.](https://eprint.iacr.org/2020/015.pdf)).

# Build

* **Requirements**: CUDA 11.7 and CMake 3.18.
* **Tested environment**: Ubuntu 20.04 with CUDA 12.4, NVIDIA A100 and RTX 4090. g++ and gcc 11.4.

## Build C++/CUDA.

```
mkdir build
cd build
cmake ..
make troy
```

Note: You could set the CUDA architecture to suit your graphic card, by setting "CMAKE_CUDA_ARCHITECTURES" variable when calling cmake. For example, `cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;89"`

## Build python bindings

```
mkdir -p build     # ensure build folder exists.
cd pybind
bash develop.sh
```

You will get a `pytroy*.whl` which could be installed.

## Building examples

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

Troy uses memory pools to manage device memory, and some APIs take a memory pool handle as an argument. By default, you can omit this argument and Troy will use a default global memory pool for all those operations. This default memory pool will be created on device 0 (the first visible CUDA device). All memory pools are thread safe so a lazy user simply use defaults and be free of any concerns of memory pools. You can disable the management of device memory by memory pools by applying `TROY_MEMORY_POOL=OFF` to cmake. If so, although memory pools can still be created, they do not manage the memory but simply call cudaMalloc and cudaFree whenever their methods are called.

Note that if you use the default memory pool, it is recommented that you call `MemoryPool::Destroy` before the program exits to safely release the static memory pool. But not doing it may just still go smoothly.

## Multithreading or multiple devices usage with memory pools

Some programs could benefit from multithreading, but running multiple threads with memory pools could be tricky. You could implement your multithread application with the following paradigms, and for example you can see `examples/20_memory_pools.cu`.

### Multithreading issue

Please note that, all kernels in this library runs on the default stream, and by default we compile with `per-thread default streams`. That is, if two kernels is triggered from two different threads, one cannot guarantee their finish order is the same as launch order. It is **your duty** to ensure that the data racing is handled when using multithreading. We illustrate this issue by an example when this is not handled properly. See `examples/30_issue_multithread.cu`.

### Multithreading with a single device

1. **Lazy** - just use one global memory pool shared by all threads, that is, you don't bother to specify any memory pools in the API. When allocation and deallocation occurs, the memory pool will be locked to ensure thread safety. When a piece of allocated memory previously used by one thread is taken by another thread, a `cudaDeviceSynchronize` will be executed, to prevent data racing (because kernel calls by the previous thread might still not be finished, as the default stream of different threads are not synchronous). This might lead to minor efficiency drop.

    ```c++
        HeContextPointer he = HeContext::create(...);
        he->to_device_inplace();
        KeyGenerator keygen = ...;
        Encryptor encryptor = ...;
        auto thread_lambda = [&he, &encryptor](){
            // simply don't care about any memory pools
            Ciphertext c = encryptor.encrypt_zero_symmetric(...);
        };
        std::vector<std::thread> threads;
        for (size_t i = 0; i < thread_num; i++) {
            threads.push_back(std::thread(thread_lambda));
        }
    ```

2. **Recommended** - one memory pool for each thread. You could create a new memory pool (`MemoryPool::create`) at the start of each thread, and use that for all operations inside that thread. Note, you can create common objects (e.g. usually `HeContext` and utility classes like `Encryptor`, etc.) *in the main thread with the global pool*, and in spawned threads you just create keys, plaintexts and ciphertexts with the thread-unique pool. There is no need to create `HeContext` for each different thread. 


    ```c++
        HeContextPointer he = HeContext::create(...);
        // just use global memory pool to create the context
        he->to_device_inplace();
        KeyGenerator keygen = ...;
        Encryptor encryptor = ...;
        auto thread_lambda = [&he, &encryptor](){
            // create memory pool on the default device 
            MemoryPoolHandle pool = MemoryPool::create(0);
            // supply the pool for any operation that involves memory allocation
            Ciphertext c = encryptor.encrypt_zero_symmetric(..., pool);
        };
        std::vector<std::thread> threads;
        for (size_t i = 0; i < thread_num; i++) {
            threads.push_back(std::thread(thread_lambda));
        }
    ```

### Multiple devices

If you wish to use **multiple GPUs, you must** create multiple memory pools to handle the memory for each device. You can create new instances of memory pools by calling `MemoryPool::create`, which takes the device index as an argument. Objects stored in different device memories cannot interact (e.g. addition between a device-0 ciphertext and a device-1 ciphertext will give undefined results). If wished, you can convey objects to other devices by calling its `to_device, to_device_inplace, clone` methods, and provide a `MemoryPoolHandle` which is created on the destination device as an argument. You can directly create multiple contexts for different devices. However, if you wish these contexts to have the same secret key, you should convey the same secret key cloned to different devices to the `KeyGenerator` constructor. Below is an example.

```c++
    // First we create a context and a keygen on the default device
    HeContextPointer he = HeContext::create(...);
    he->to_device_inplace();
    KeyGenerator keygen = ...;
    // Obtain the secret key
    SecretKey secret_key = keygen.secret_key().clone();
    auto thread_lambda = [&secret_key](size_t device_index){
        // Create a handle for the memory pool on the specific device
        MemoryPoolHandle pool = MemoryPool::create(device_index);
        // Create context and convey to that device
        HeContextPointer he = HeContext::create(...); 
        he->to_device_inplace(pool);
        // Create keygen and set the same secret key
        KeyGenerator keygen(he, secret_key, pool);
        // Now you can create encryptor etc
        ...
    };
```


# Contribute
Feel free to fork / pull request.
Please cite this paper if you use it in your work!
```
@inproceedings{liu2024pencil,
  title={{Pencil: Private and Extensible Collaborative Learning without the Non-Colluding Assumption}},
  author={Liu, Xuanqi and Liu, Zhuotao and Li, Qi and Xu, Ke and Xu, Mingwei},
  journal={NDSS},
  year={2024}
}
```
