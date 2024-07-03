# Troy-Nova

The new implementation of Troy, a CUDA based GPU parallelized implementation of RLWE homomorphic encryption schemes. We support RNS version of BFV, CKKS and BGV schemes. The implementation itself is inspired by the [Microsoft SEAL](https://github.com/microsoft/SEAL) library, so the interfaces are very similar to theirs. 

We also include some utilities for privacy computing, including the matrix multiplication (from [BumbleBee](https://eprint.iacr.org/2023/1678)), 2d convolution (from [Cheetah](https://www.usenix.org/system/files/sec22-huang-zhicong.pdf)) and LWE-ciphertext extraction and packing (from [Chen et al.](https://eprint.iacr.org/2020/015.pdf)).

# Build

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

# Memory Pool

Troy uses memory pools to manage device memory, and some APIs take a memory pool handle as an argument. By default, you can omit this argument and Troy will use a default global memory pool for all those operations. This default memory pool will be created on device 0 (the first visible CUDA device). All memory pools are thread safe so a lazy user simply use defaults and be free of any concerns of memory pools. 

Note that if you use the default memory pool, it is recommented that you call `MemoryPool::Destroy` before the program exits to safely release the static memory pool. But not doing it may just still go smoothly.

Multithreaded programs *may* benefit from using one memory pool for each thread. Furthermore, if you wish to use **multiple GPUs, you must** create multiple memory pools to handle the memory for each device. You can create new instances of memory pools by calling `MemoryPool::Create`, which takes the device index as an argument. See [`memory_pool.cuh`](src/utils/memory_pool.cuh). For the usage multiple memory pools, see the example at `examples/20_memory_pools.cu`.

You can disable the management of device memory by memory pools by applying `TROY_MEMORY_POOL=OFF` to cmake. If so, although memory pools can still be created, they do not manage the memory but simply call cudaMalloc and cudaFree whenever their methods are called. When memory pool is enabled, some users report unexpected exceptions complaining `"[MemoryPool::get] The singleton has been destroyed."` when using the library. One could check if `MemoryPool::Destroy()` has been called prematurely in your program to locate the problem, or one could simply provide a `TROY_MEMORY_POOL_UNSAFE=OFF` to try to avoid it. This `TROY_MEMORY_POOL_UNSAFE` is an experimental hacking solution (because I have not reproduced the error on my machine yet 😢), so if you still get the errors please file an issue.

# Contribute
Feel free to fork / pull request.
Please cite this repository if you use it in your work.