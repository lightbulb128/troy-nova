#include "memory_pool.cuh"

namespace troy::utils {

    std::shared_ptr<MemoryPool> MemoryPool::global_pool = nullptr;
    std::mutex MemoryPool::global_pool_mutex = std::mutex();

}

#ifdef TROY_MEMORY_POOL 

    #ifndef TROY_MEMORY_POOL_UNSAFE

        #include "memory_pool_safe.in"
    
    #else

        #include "memory_pool_unsafe.in"
    
    #endif

#else

    #include "memory_pool_none.in"

#endif