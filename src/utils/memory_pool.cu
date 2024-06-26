#include "memory_pool.cuh"

namespace troy {

    namespace utils {

#ifdef TROY_MEMORY_POOL

        std::shared_ptr<MemoryPool> MemoryPool::global_pool = nullptr;
        std::mutex MemoryPool::global_pool_mutex = std::mutex();

#endif

    }
}