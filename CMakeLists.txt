project(TroyNew LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
endif()

# if CMAKE_BUILD_TYPE is empty, set it to Release
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

cmake_minimum_required(VERSION 3.16)

option(TROY_TEST "Build tests" OFF)
option(TROY_BENCH "Build benchmarks" OFF)
option(TROY_MEMORY_POOL "Use memory pool" ON)
option(TROY_PYBIND "Build python encapsulation with Pybind11" ON)

add_subdirectory(src)

if(TROY_PYBIND)
  add_subdirectory(extern/pybind11)
  add_subdirectory(pybind)
endif()

add_subdirectory(test)