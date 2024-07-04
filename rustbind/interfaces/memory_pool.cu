#include "memory_pool.cuh"
#include "troy/troy.cuh"

namespace troy_rust {

    std::unique_ptr<MemoryPool> create_memory_pool(size_t device_index) {
        return std::make_unique<MemoryPool>(device_index);
    }

    MemoryPool::MemoryPool(size_t device_index) {
        pool = troy::MemoryPool::create(device_index);
    }
    
}