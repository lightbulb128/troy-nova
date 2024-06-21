#pragma once
#include <map>
#include <stdexcept>
#include <iostream>
#include <set>

namespace troy {namespace utils {

#ifdef TROY_MEMORY_POOL

#ifdef TROY_MEMORY_POOL_UNSAFE


    // MemoryPool defined with `TROY_MEMORY_POOL` and `TROY_MEMORY_POOL_UNSAFE` manages
    // the device memory allocation and freeing. This `unsafe` implementation never
    // automatically frees the memory unless the user explicity calls `ReleaseUnused()`.
    class MemoryPool {

    private:

        static const int PRESERVED_MEMORY_BYTES = 1024 * 1024 * 32;

        static MemoryPool singleton;

        std::multimap<size_t, void*> unused;
        std::multimap<void*, size_t> allocated;
        size_t total_allocated;
        size_t total_memory;

        inline static void runtime_error(const char* prompt, cudaError_t status) {
            std::string msg = prompt;
            msg += ": ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }

        inline MemoryPool() {
            total_allocated = 0;
            cudaDeviceProp props;
            cudaError_t status = cudaGetDeviceProperties(&props, 0);
            if (status != cudaSuccess) {
                runtime_error("[MemoryPool::MemoryPool] cudaGetDeviceProperties failed.", status);
            }
            total_memory = props.totalGlobalMem;
        }

        inline ~MemoryPool() {
            // Do nothing when `unsafe`
        }

        inline void release_unused() {
            for (auto it = unused.begin(); it != unused.end();) {
                cudaError_t status = cudaFree(it->second);
                if (status != cudaSuccess) {
                    runtime_error("[MemoryPool::release_unused] cudaFree failed.", status);
                }
                // remove pointer from allocated
                auto it2 = allocated.find(it->second);
                if (it2 == allocated.end()) {
                    throw std::runtime_error("[MemoryPool::release_unused] The pointer is not in the allocated set.");
                }
                allocated.erase(it2);
                it = unused.erase(it);
            }
        }

        inline void* try_allocate(size_t required) {
            size_t free, total;
            cudaError_t status = cudaMemGetInfo(&free, &total);
            if (status != cudaSuccess) {
                runtime_error("[MemoryPool::try_allocate] cudaMemGetInfo failed", status);
            }
            if (free < required + PRESERVED_MEMORY_BYTES) {
                release_unused();
                // try again
                status = cudaMemGetInfo(&free, &total);
                if (status != cudaSuccess) {
                    runtime_error("[MemoryPool::try_allocate] cudaMemGetInfo failed", status);
                }
                if (free < required + PRESERVED_MEMORY_BYTES) {
                    throw std::runtime_error("[MemoryPool::try_allocate] Not enough memory.");
                }
            }
            void* ptr = nullptr;
            status = cudaMalloc(&ptr, required);
            if (status != cudaSuccess) {
                runtime_error("[MemoryPool::try_allocate] cudaMalloc failed", status);
            }
            total_allocated += required;
            allocated.insert(std::make_pair(ptr, required));
            return ptr;
        }

        inline void* get(size_t required) {
            auto iterator = unused.lower_bound(required);
            if (iterator == unused.end() || iterator->first >= required * 2) {
                return try_allocate(required);
            } else {
                void* ptr = iterator->second;
                unused.erase(iterator);
                return ptr;
            }
        }

        inline void give_back(void* ptr) {
            auto iterator = allocated.find(ptr);
            if (iterator == allocated.end()) {
                throw std::runtime_error("[MemoryPool::give_back] The pointer is not in the allocated set.");
            }
            size_t size = iterator->second;
            unused.insert(std::make_pair(size, ptr));
        }

    public:

        inline static void* Allocate(size_t required) {
            return singleton.get(required);
        }

        inline static void Free(void* ptr) {
            singleton.give_back(ptr);
        }

        inline static void ReleaseUnused() {
            singleton.release_unused();
        }

        inline static void Destroy() {
            // do nothing
        }

    };

#else // TROY_MEMORY_POOL_UNSAFE

    // MemoryPool defined with `TROY_MEMORY_POOL` and no `TROY_MEMORY_POOL_UNSAFE` manages
    // the device memory allocation and freeing. User should call
    // `Destroy()` before the program exit to free all the memory, and
    // all further operations after destruction will fail.
    class MemoryPool {

    private:

        static const int PRESERVED_MEMORY_BYTES = 1024 * 1024 * 32;

        static MemoryPool singleton;

        std::multimap<size_t, void*> unused;
        std::multimap<void*, size_t> allocated;
        std::set<void*> zombie;
        size_t total_allocated;
        size_t total_memory;
        bool destroyed;

        inline static void runtime_error(const char* prompt, cudaError_t status) {
            std::string msg = prompt;
            msg += ": ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }

        inline MemoryPool() {
            total_allocated = 0;
            cudaDeviceProp props;
            cudaError_t status = cudaGetDeviceProperties(&props, 0);
            if (status != cudaSuccess) {
                runtime_error("[MemoryPool::MemoryPool] cudaGetDeviceProperties failed.", status);
            }
            total_memory = props.totalGlobalMem;
            destroyed = false;
        }

        inline ~MemoryPool() {
            if (allocated.size() > 0 && !destroyed) {
                std::cerr << "[MemoryPool::~MemoryPool] The singleton was not destroyed before the program exit.\n";
            }
            // if (zombie.size() > 0) {
            //     std::cerr << "[MemoryPool::~MemoryPool] The zombie set is not empty. There may be memory leak.\n";
            // }
        }

        inline void release_unused() {
            for (auto it = unused.begin(); it != unused.end();) {
                cudaError_t status = cudaFree(it->second);
                if (status != cudaSuccess) {
                    runtime_error("[MemoryPool::release_unused] cudaFree failed.", status);
                }
                // remove pointer from allocated
                auto it2 = allocated.find(it->second);
                if (it2 == allocated.end()) {
                    throw std::runtime_error("[MemoryPool::release_unused] The pointer is not in the allocated set.");
                }
                allocated.erase(it2);
                it = unused.erase(it);
            }
        }

        inline void* try_allocate(size_t required) {
            size_t free, total;
            cudaError_t status = cudaMemGetInfo(&free, &total);
            if (status != cudaSuccess) {
                runtime_error("[MemoryPool::try_allocate] cudaMemGetInfo failed", status);
            }
            if (free < required + PRESERVED_MEMORY_BYTES) {
                release_unused();
                // try again
                status = cudaMemGetInfo(&free, &total);
                if (status != cudaSuccess) {
                    runtime_error("[MemoryPool::try_allocate] cudaMemGetInfo failed", status);
                }
                if (free < required + PRESERVED_MEMORY_BYTES) {
                    throw std::runtime_error("[MemoryPool::try_allocate] Not enough memory.");
                }
            }
            void* ptr = nullptr;
            status = cudaMalloc(&ptr, required);
            if (status != cudaSuccess) {
                runtime_error("[MemoryPool::try_allocate] cudaMalloc failed", status);
            }
            total_allocated += required;
            allocated.insert(std::make_pair(ptr, required));
            return ptr;
        }

        inline void* get(size_t required) {
            if (destroyed) {
                throw std::runtime_error("[MemoryPool::get] The singleton has been destroyed.");
            }
            auto iterator = unused.lower_bound(required);
            if (iterator == unused.end() || iterator->first >= required * 2) {
                return try_allocate(required);
            } else {
                void* ptr = iterator->second;
                unused.erase(iterator);
                return ptr;
            }
        }

        inline void give_back(void* ptr) {
            if (destroyed) {
                // the ptr given back should be in the zombie set
                auto iterator = zombie.find(ptr);
                if (iterator == zombie.end()) {
                    throw std::runtime_error("[MemoryPool::give_back] The pointer is not in the zombie set.");
                }
                zombie.erase(iterator);
                return;
            }
            auto iterator = allocated.find(ptr);
            if (iterator == allocated.end()) {
                throw std::runtime_error("[MemoryPool::give_back] The pointer is not in the allocated set.");
            }
            size_t size = iterator->second;
            unused.insert(std::make_pair(size, ptr));
        }

        inline void destroy() {
            // first release all unused
            for (auto it = unused.begin(); it != unused.end();) {
                cudaError_t status = cudaFree(it->second);
                if (status != cudaSuccess) {
                    runtime_error("[MemoryPool::destroy] cudaFree unused failed.", status);
                }
                // remove pointer from allocated
                auto it2 = allocated.find(it->second);
                if (it2 == allocated.end()) {
                    throw std::runtime_error("[MemoryPool::destroy] The pointer is not in the allocated set.");
                }
                allocated.erase(it2);
                it = unused.erase(it);
            }
            // for all the remaining in allocated, move them to zombies
            for (auto it = allocated.begin(); it != allocated.end();) {
                cudaError_t status = cudaFree(it->first);
                if (status != cudaSuccess) {
                    runtime_error("[MemoryPool::destroy] cudaFree allocated failed.", status);
                }
                zombie.insert(it->first);
                it = allocated.erase(it);
            }
            allocated.clear();
            unused.clear();
            total_allocated = 0;
            destroyed = true;
        }

    public:

        inline static void* Allocate(size_t required) {
            return singleton.get(required);
        }

        inline static void Free(void* ptr) {
            singleton.give_back(ptr);
        }

        inline static void ReleaseUnused() {
            singleton.release_unused();
        }

        inline static void Destroy() {
            singleton.destroy();
        }

        

    };

#endif // TROY_MEMORY_POOL_UNSAFE

#else

    class MemoryPool {

    public:

        inline static void* Allocate(size_t required) {
            void* ptr = nullptr;
            cudaError_t status = cudaMalloc(&ptr, required);
            if (status != cudaSuccess) {
                std::string msg = "[MemoryPool::Allocate] cudaMalloc failed: ";
                msg += cudaGetErrorString(status);
                throw std::runtime_error(msg);
            }
            return ptr;
        }

        inline static void Free(void* ptr) {
            cudaError_t status = cudaFree(ptr);
            if (status != cudaSuccess) {
                std::string msg = "[MemoryPool::Free] cudaFree failed: ";
                msg += cudaGetErrorString(status);
                throw std::runtime_error(msg);
            }
        }

        inline static void ReleaseUnused() {}

        inline static void Destroy() {}

    };

#endif

}}