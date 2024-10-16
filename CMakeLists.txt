cmake_minimum_required(VERSION 3.16)

project(TroyNew LANGUAGES C CXX CUDA)

# ---------- Options ----------

option(TROY_TEST "Build tests" OFF)
option(TROY_BENCH "Build benchmarks" OFF)
option(TROY_MEMORY_POOL "Use memory pool" ON)
option(TROY_MEMORY_POOL_UNSAFE "Use unsafe memory pool" OFF)
option(TROY_PYBIND "Build python encapsulation with Pybind11" ON)
option(TROY_EXAMPLES "Build examples" OFF)
option(TROY_ZSTD "Use Zstandard for compression" ON)
option(TROY_STREAM_SYNC_AFTER_KERNEL_CALLS "Synchronize stream after each kernel call" OFF)
option(TROY_DEVICE_SYNCHRONIZE_ON_MULTIPLE_THREAD "Synchronize device when memory is allocated to another thread" ON)
option(TROY_STREAM_SYNC_ON_RETURN_TO_MEMORY_POOL "Synchronize stream when returning memory to pool" OFF)

if (TROY_MEMORY_POOL_UNSAFE)
  # if TROY_MEMORY_POOL_UNSAFE is enabled, TROY_MEMORY_POOL must be enabled
  if(NOT TROY_MEMORY_POOL)
    message(WARNING "TROY_MEMORY_POOL_UNSAFE is enabled, but TROY_MEMORY_POOL is set to OFF.")
    message(WARNING "Setting TROY_MEMORY_POOL to ON.")
    set(TROY_MEMORY_POOL ON)
  endif()
endif()

# ---------- Set flags ----------

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread")

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

# --------- Check dependencies --------

include(FetchContent)

function(directory_nonexistent_or_empty dir outvar)
  if(NOT EXISTS ${dir})
    set(${outvar} TRUE PARENT_SCOPE)
  else()
    file(GLOB files ${dir}/*)
    if(files)
      set(${outvar} FALSE PARENT_SCOPE)
    else()
      set(${outvar} TRUE PARENT_SCOPE)
    endif()
  endif()
endfunction()

set(TROY_ZSTD_GIT_HASH "ab02fd342f5c06be9fc309098f4bd54b0c1b1de9")

if(TROY_ZSTD)
  # first try to find package
  find_package(zstd CONFIG)
  if(zstd_FOUND)
    if(TARGET zstd::libzstd_static)
      message(STATUS "TROY_ZSTD is enabled and found in system.")
      set(TROY_ZSTD_STATIC_LINK_NAME "zstd::libzstd_static")
    elseif(TARGET libzstd)
      get_target_property(libzstd_type libzstd TYPE)
      if(libzstd_type STREQUAL "STATIC_LIBRARY")
        message(STATUS "TROY_ZSTD is enabled and found in system.")
        set(TROY_ZSTD_STATIC_LINK_NAME "libzstd")
      else()
        message(FATAL_ERROR "libzstd is not a static library.")
      endif()
    endif()
  endif()
  if(NOT DEFINED TROY_ZSTD_STATIC_LINK_NAME)
    directory_nonexistent_or_empty("${CMAKE_CURRENT_SOURCE_DIR}/extern/zstd" TROY_ZSTD_NOT_FOUND)
    if(TROY_ZSTD_NOT_FOUND)
      message(STATUS "TROY_ZSTD is enabled, but not found in extern/zstd. Try to git clone.")
      FetchContent_Declare(
        zstd
        GIT_REPOSITORY "https://github.com/facebook/zstd.git"
        GIT_TAG        ${TROY_ZSTD_GIT_HASH}
        SOURCE_SUBDIR build/cmake
      )
      FetchContent_MakeAvailable(zstd)
      set(TROY_ZSTD_STATIC_LINK_NAME "libzstd_static")
    else()
      add_subdirectory(extern/zstd/build/cmake)
      set(TROY_ZSTD_STATIC_LINK_NAME "libzstd_static")
    endif()
    unset(TROY_ZSTD_NOT_FOUND)
  endif()
endif()

cmake_minimum_required(VERSION 3.16)

if (TROY_MEMORY_POOL_UNSAFE)
  # if TROY_MEMORY_POOL_UNSAFE is enabled, TROY_MEMORY_POOL must be enabled
  if(NOT TROY_MEMORY_POOL)
    message(WARNING "TROY_MEMORY_POOL_UNSAFE is enabled, but TROY_MEMORY_POOL is set to OFF.")
    message(WARNING "Setting TROY_MEMORY_POOL to ON.")
    set(TROY_MEMORY_POOL ON)
  endif()
endif()

set(TROY_PYBIND11_GIT_HASH "a2e59f0e7065404b44dfe92a28aca47ba1378dc4")

if(TROY_PYBIND)
  directory_nonexistent_or_empty("${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11" TROY_PYBIND11_NOT_FOUND)
  if(TROY_PYBIND11_NOT_FOUND)
    message(STATUS "TROY_PYBIND is enabled, but not found in extern/pybind11. Try to git clone.")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
      GIT_TAG        ${TROY_PYBIND11_GIT_HASH}
    )
    FetchContent_MakeAvailable(pybind11)
  else()
    add_subdirectory(extern/pybind11)
  endif()
  unset(TROY_PYBIND11_NOT_FOUND)
endif()

set(TROY_GOOGLETEST_GIT_HASH "b10fad38c4026a29ea6561ab15fc4818170d1c10")

if(TROY_TEST)
  directory_nonexistent_or_empty("${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest" TROY_GOOGLETEST_NOT_FOUND)
  if(TROY_GOOGLETEST_NOT_FOUND)
    message(STATUS "TROY_TEST is enabled, but not found in extern/googletest. Try to git clone.")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY "https://github.com/google/googletest.git"
      GIT_TAG        ${TROY_GOOGLETEST_GIT_HASH}
    )
    FetchContent_MakeAvailable(googletest)
  else()
    add_subdirectory(extern/googletest)
  endif()
  unset(TROY_GOOGLETEST_NOT_FOUND)
endif()

# --------- Stard building -----------

if(TROY_STREAM_SYNC_AFTER_KERNEL_CALLS)
  add_compile_definitions(TROY_STREAM_SYNC_AFTER_KERNEL_CALLS)
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)

if(TROY_PYBIND)
  add_subdirectory(pybind)
endif()

if(TROY_EXAMPLES)
  add_subdirectory(examples)
endif()

add_subdirectory(test)