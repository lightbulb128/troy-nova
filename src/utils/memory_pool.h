#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <iostream>
#include <set>

namespace troy {namespace utils {

    inline size_t device_count() {
        int count;
        cudaError_t status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) {
            std::string msg = "[device_count] cudaGetDeviceCount failed: ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }
        return count;
    }

    class MemoryPool;
    typedef std::shared_ptr<MemoryPool> MemoryPoolHandle;

    class MemoryPool {
    private:

        static std::shared_ptr<MemoryPool> global_pool;
        static std::mutex global_pool_mutex;
        bool denying = false;
        bool is_global_pool = false;
        size_t device_index;
        struct Impl;
        std::shared_ptr<Impl> impl_;
        inline static void ensure_global_pool() {
            std::unique_lock lock(global_pool_mutex);
            if (global_pool == nullptr) {
                global_pool = std::make_shared<MemoryPool>(0);
                global_pool->is_global_pool = true;
            }
        }
        inline static void runtime_error(const char* prompt, cudaError_t status) {
            std::string msg = prompt;
            msg += ": ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }
        void* try_allocate(size_t required);

    public:
        static int implementation_type();
        inline void set_device() {
            cudaError_t status = cudaSetDevice(device_index);
            if (status != cudaSuccess) {
                runtime_error("[MemoryPool::set_devicew] cudaSetDevice failed.", status);
            }
        }
        inline size_t get_device() {
            return device_index;
        }
        inline void deny(bool set = true) {
            denying = set;
        }
        MemoryPool(size_t device_index = 0);
        ~MemoryPool();
        void* allocate(size_t required);
        void release(void* ptr);
        void destroy();
        void release_unused();
        inline static void* Allocate(size_t required) {
            ensure_global_pool();
            return global_pool->allocate(required);
        }
        inline static void Free(void* ptr) {
            ensure_global_pool();
            global_pool->release(ptr);
        }
        inline static void ReleaseUnused() {
            ensure_global_pool();
            global_pool->release_unused();
        }
        inline static void Destroy() {
            ensure_global_pool();
            global_pool->destroy();
        }
        inline static MemoryPoolHandle GlobalPool() {
            ensure_global_pool();
            return global_pool;
        }
        inline static MemoryPoolHandle create(size_t device = 0) {
            return std::make_shared<MemoryPool>(device);
        }
    };

}

// make available in high level namespace
using utils::MemoryPool;
using utils::MemoryPoolHandle;

}