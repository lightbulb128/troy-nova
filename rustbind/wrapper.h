#include "troy/troy.h"

namespace troy_wrapper {
    // this namespace is for defining some additional
    // utility functions or workaround for shared_ptr.
    // bindgen cannot return c++ shared_ptr natively,
    // we must give it as a parameter func(shared_ptr<T>* out)
    // to obtain the result.
    // see https://github.com/rust-lang/rust-bindgen/issues/1509

    extern "C"
    void create_memory_pool_handle(size_t device_index, troy::MemoryPoolHandle* out);

}