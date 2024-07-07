#include "wrapper.h"

namespace troy_wrapper {

    extern "C"
    void create_memory_pool_handle(size_t device_index, troy::MemoryPoolHandle* out) {
        *out = troy::MemoryPool::create(device_index);
    }

}