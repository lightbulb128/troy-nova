# This script tests the correctness of the 3 options of memory pool impls of Troy.


set -ex

cd build

cmake .. -DTROY_MEMORY_POOL=OFF
make troytest -j64
cd test
ctest
cd ..

cmake .. -DTROY_MEMORY_POOL=ON -DTROY_MEMORY_POOL_UNSAFE=ON
make troytest -j64
cd test
ctest
cd ..

cmake .. -DTROY_MEMORY_POOL=ON -DTROY_MEMORY_POOL_UNSAFE=OFF
make troytest -j64
cd test
ctest
cd ..

cd ..