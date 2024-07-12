#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "basics.h"
#include "box.h"

namespace troy { namespace reduction {

    template<typename T>
    T max(utils::ConstSlice<T> inputs, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    template<typename T>
    T min(utils::ConstSlice<T> inputs, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    template<typename T>
    T sum(utils::ConstSlice<T> inputs, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    template<typename T>
    size_t nonzero_count(utils::ConstSlice<T> inputs, MemoryPoolHandle pool = MemoryPool::GlobalPool());
    
}}