#include "memory_pool.h"

namespace troy::utils {

    std::shared_ptr<MemoryPool> MemoryPool::global_pool = nullptr;
    std::mutex MemoryPool::global_pool_mutex = std::mutex();
    bool MemoryPool::established = false;
    bool MemoryPool::has_device = false;

    void stream_sync() {
        #ifdef TROY_STREAM_SYNC_AFTER_KERNEL_CALLS
            cudaError_t status = cudaStreamSynchronize(0);
            if (status != cudaSuccess) {
                std::string msg = "[stream_sync] cudaStreamSynchronize failed: ";
                msg += cudaGetErrorString(status);
                throw std::runtime_error(msg);
            }
        #endif
    }

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