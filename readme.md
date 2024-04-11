# Troy-Nova

The new implementation of Troy (CUDA-HE).

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