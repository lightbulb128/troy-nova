#pragma once
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <iostream>
#include <set>
#include <thread>

namespace troy {namespace utils {

    inline size_t device_count() {
        int count;
        cudaError_t status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) {
            if (
                status == cudaErrorNoDevice || 
                status == cudaErrorInitializationError || 
                status == cudaErrorInsufficientDriver || 
                status == cudaErrorNotSupported || 
                status == cudaErrorNotPermitted
            ) {
                return 0;
            }
            std::string msg = "[device_count] cudaGetDeviceCount failed: ";
            msg += cudaGetErrorString(status);
            msg += "; perhaps report this issue.";
            throw std::runtime_error(msg);
        }
        return count;
    }

    // If "TROY_STREAM_SYNC_AFTER_KERNEL_CALLS" is enabled,
    // this will call cudaStreamSynchronize(0). Otherwise, it does nothing.
    void stream_sync();

    inline void stream_sync_concrete() {
        cudaError_t status = cudaStreamSynchronize(0);
        if (status != cudaSuccess) {
            std::string msg = "[stream_sync_concrete] cudaStreamSynchronize failed: ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }
    }


    class MemoryPool;
    typedef std::shared_ptr<MemoryPool> MemoryPoolHandle;

    class MemoryPool {
    private:

        static std::shared_ptr<MemoryPool> global_pool;
        static std::mutex global_pool_mutex;
        static bool established;
        static bool has_device;
        bool denying = false;
        bool is_global_pool = false;
        size_t device_index;
        struct Impl;
        std::shared_ptr<Impl> impl_;
        inline static void ensure_global_pool() {
            // Don't directly exclusively lock but check established first.
            if (!established) {
                std::unique_lock lock(global_pool_mutex);
                if (global_pool != nullptr) {
                    return;
                }
                int count = device_count();
                has_device = count > 0;
                if (!has_device) {
                    return;
                }
                global_pool = std::make_shared<MemoryPool>(0);
                global_pool->is_global_pool = true;
                established = true;
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
        void force_set_thread_id(std::thread::id id);
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
            if (!has_device) {
                return;
            }
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

    inline void set_device(size_t device) {
        cudaError_t status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            std::string msg = "[set_device] cudaSetDevice failed: ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }
    }

}

// make available in high level namespace
using utils::MemoryPool;
using utils::MemoryPoolHandle;

}