#include "cuda_runtime.h"
#include <memory>
#include "troy/troy.cuh"

namespace troy::utils {
    class MemoryPool;
    typedef std::shared_ptr<MemoryPool> MemoryPoolHandle;
}

namespace troy_rust {

    class MemoryPool {
    private:
        troy::utils::MemoryPoolHandle pool;
    public:
        MemoryPool(size_t device_index);
        
    };

    std::unique_ptr<MemoryPool> create_memory_pool(size_t device_index);
    
}