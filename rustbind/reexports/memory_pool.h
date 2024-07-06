#pragma once
#include <memory>
#include "cuda_runtime.h"
#include "troy/troy.cuh"

namespace troy_rust {

    class MemoryPool;
    typedef std::unique_ptr<MemoryPool> UpMemoryPool;

    class MemoryPool {
    private:
        troy::utils::MemoryPoolHandle pool;
    public:
        explicit inline MemoryPool(troy::utils::MemoryPoolHandle pool) : pool(pool) {}
        explicit inline MemoryPool(size_t device_index) {
            pool = troy::MemoryPool::create(device_index);
        }
        MemoryPool(const MemoryPool& pool) = default;
        MemoryPool& operator=(const MemoryPool& pool) = default;
        inline size_t handle_address() const {
            return (pool == nullptr) ? 0 : reinterpret_cast<size_t>(pool.get());
        }
        inline size_t device_index() const {
            return (pool == nullptr) ? 0 : pool->get_device();
        }
    };

    inline UpMemoryPool memory_pool_constructor(size_t device_index) {
        return std::make_unique<MemoryPool>(device_index);
    }
    inline UpMemoryPool memory_pool_constructor_copy(const MemoryPool& pool) {
        return std::make_unique<MemoryPool>(pool);
    }
    inline UpMemoryPool memory_pool_static_global_pool() {
        return std::make_unique<MemoryPool>(troy::MemoryPool::GlobalPool());
    }
    inline void memory_pool_static_destroy() {
        troy::MemoryPool::Destroy();
    }
    inline UpMemoryPool memory_pool_static_nullptr() {
        return std::make_unique<MemoryPool>(nullptr);
    }
    
}