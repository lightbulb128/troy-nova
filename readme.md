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

To prevent frequent allocation and freeing of device memory, we use a static MemoryPool to manage all allocated device memory (See [`memory_pool.cuh`](src/utils/memory_pool.cuh)). If you wish to disable this behavior you could provide `TROY_MEMORY_POOL=OFF` to cmake options. 

Some users report unexpected exceptions complaining `"[MemoryPool::get] The singleton has been destroyed."` when using the library. One could check if `MemoryPool::Destroy()` has been called prematurely in your program to locate the problem, or one could simply provide a `TROY_MEMORY_POOL_UNSAFE=OFF` to try to avoid it. This `TROY_MEMORY_POOL_UNSAFE` is an experimental hacking solution (because I have not reproduced the error on my machine yet 😢), so if you still get the errors please file an issue.

# Contribute
Feel free to fork / pull request.
Please cite this repository if you use it in your work.