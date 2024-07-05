cmake_minimum_required(VERSION 3.18)

project(TroyNew LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread -Xcompiler -Wall -Xcompiler -Wextra")


# if cmake version > 3.18
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
endif()

# Print CUDA archs
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# if CMAKE_BUILD_TYPE is empty, set it to Release
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

cmake_minimum_required(VERSION 3.16)

option(TROY_TEST "Build tests" OFF)
option(TROY_BENCH "Build benchmarks" OFF)
option(TROY_MEMORY_POOL "Use memory pool" ON)
option(TROY_MEMORY_POOL_UNSAFE "Use unsafe memory pool" OFF)
option(TROY_PYBIND "Build python encapsulation with Pybind11" ON)
option(TROY_EXAMPLES "Build examples" OFF)

if (TROY_MEMORY_POOL_UNSAFE)
  # if TROY_MEMORY_POOL_UNSAFE is enabled, TROY_MEMORY_POOL must be enabled
  if(NOT TROY_MEMORY_POOL)
    message(WARNING "TROY_MEMORY_POOL_UNSAFE is enabled, but TROY_MEMORY_POOL is set to OFF.")
    message(WARNING "Setting TROY_MEMORY_POOL to ON.")
    set(TROY_MEMORY_POOL ON)
  endif()
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)

if(TROY_PYBIND)
  add_subdirectory(extern/pybind11)
  add_subdirectory(pybind)
endif()

if(TROY_EXAMPLES)
  add_subdirectory(examples)
endif()

add_subdirectory(test)