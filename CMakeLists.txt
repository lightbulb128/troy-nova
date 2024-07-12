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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread")

option(TROY_WARNINGS "Enable warnings" OFF)
if (TROY_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic,-Wno-comment")
endif()

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

function(fetch_if_not_populated name)
  FetchContent_GetProperties(${name})
  if(NOT ${name}_POPULATED)
    FetchContent_Populate(${name})
  endif()
endfunction()

# make sure ./extern exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/extern)

set(TROY_ZSTD_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/zstd")
set(TROY_ZSTD_GIT_HASH "ab02fd342f5c06be9fc309098f4bd54b0c1b1de9")

if(TROY_ZSTD)
  directory_nonexistent_or_empty("${TROY_ZSTD_DIRECTORY}" TROY_ZSTD_NOT_FOUND)
  if(TROY_ZSTD_NOT_FOUND)
    message(STATUS "TROY_ZSTD is enabled, but not found in extern/zstd. Try to git clone.")
    FetchContent_Declare(
      zstd
      GIT_REPOSITORY "https://github.com/facebook/zstd.git"
      GIT_TAG        ${TROY_ZSTD_GIT_HASH}
      SOURCE_DIR     ${TROY_ZSTD_DIRECTORY}
    )
    fetch_if_not_populated(zstd)
  endif()
  unset(TROY_ZSTD_NOT_FOUND)
endif()

set(TROY_PYBIND11_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11")
set(TROY_PYBIND11_GIT_HASH "8a099e44b3d5f85b20f05828d919d2332a8de841")

if(TROY_PYBIND)
  directory_nonexistent_or_empty("${TROY_PYBIND11_DIRECTORY}" TROY_PYBIND11_NOT_FOUND)
  if(TROY_PYBIND11_NOT_FOUND)
    message(STATUS "TROY_PYBIND is enabled, but not found in extern/pybind11. Try to git clone.")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
      GIT_TAG        ${TROY_PYBIND11_GIT_HASH}
      SOURCE_DIR     ${TROY_PYBIND11_DIRECTORY}
    )
    fetch_if_not_populated(pybind11)
  endif()
  unset(TROY_PYBIND11_NOT_FOUND)
endif()

set(TROY_GOOGLETEST_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest")
set(TROY_GOOGLETEST_GIT_HASH "b10fad38c4026a29ea6561ab15fc4818170d1c10")

if(TROY_TEST)
  directory_nonexistent_or_empty("${TROY_GOOGLETEST_DIRECTORY}" TROY_GOOGLETEST_NOT_FOUND)
  if(TROY_GOOGLETEST_NOT_FOUND)
    message(STATUS "TROY_TEST is enabled, but not found in extern/googletest. Try to git clone.")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY "https://github.com/google/googletest.git"
      GIT_TAG        ${TROY_GOOGLETEST_GIT_HASH}
      SOURCE_DIR     ${TROY_GOOGLETEST_DIRECTORY}
    )
    fetch_if_not_populated(googletest)
  endif()
  unset(TROY_GOOGLETEST_NOT_FOUND)
endif()

# --------- Stard building -----------

if(TROY_ZSTD)
  # Since zstd have warnings about "-Wmaybe-unitialized", we disable this warning first before including
  # the zstd library
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
  add_subdirectory(extern/zstd/build/cmake)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
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