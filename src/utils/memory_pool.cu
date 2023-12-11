#include "memory_pool.cuh"

namespace troy {

    namespace utils {

#ifdef TROY_MEMORY_POOL

        MemoryPool MemoryPool::singleton;

#endif

    }
}