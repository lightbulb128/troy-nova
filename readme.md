# Troy-Nova

The new implementation of Troy (CUDA-HE). We support RNS version of BFV, CKKS and BGV scheme. The implementation itself is inspired by the [Microsoft SEAL](https://github.com/microsoft/SEAL) library, so the interfaces are very similar to theirs. 

We also include some utilities for privacy computing, including the matrix multiplication (from [BumbleBee](https://eprint.iacr.org/2023/1678)), 2d convolution (from [Cheetah](https://www.usenix.org/system/files/sec22-huang-zhicong.pdf)) and LWE-ciphertext extraction and packing (from [Chen et al.](https://eprint.iacr.org/2020/015.pdf)).

# Build

Build C++/CUDA.

```
mkdir build
cd build
cmake ..
make troy
```

Build python bindings.

```
mkdir -p build     # ensure build folder exists.
cd pybind
bash develop.sh
```

You will get a `pytroy*.whl` which could be installed. 

# Testing and Benchmark

```
cd build
cmake .. -DTROY_TEST=ON -DTROY_BENCH=ON
make
cd test
ctest
./troybench
```

## Contribute
Feel free to fork / pull request.
Please cite this repository if you use it in your work.