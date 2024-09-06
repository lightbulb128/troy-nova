#pragma once
#include "cuda_runtime.h"
#include <stdexcept>
#include <cstring>
#include "utils/memory_pool.h"

namespace troy {

    namespace kernel_provider {

        inline void runtime_error(const char* prompt, cudaError_t status) {
            std::string msg = prompt;
            msg += ": ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }

        inline void initialize(int device = 0) {
            cudaError_t status = cudaSetDevice(device);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::initialize] cudaSetDevice failed", status);
            }
        }

        template <typename T>
        inline T* malloc(MemoryPool& pool, size_t length) {
            if (length == 0) return nullptr;
            T* ret = reinterpret_cast<T*>(pool.allocate(length * sizeof(T)));
            return ret;
        }

        template <typename T>
        inline void free(MemoryPool& pool, T* ptr) {
            pool.release(reinterpret_cast<void*>(ptr));
        }


        template <typename T>
        inline void copy_host_to_device(MemoryPool& pool, T* dst, const T* src, size_t length) {
            if (length == 0) return;
            pool.set_device();
            cudaError_t status = cudaMemcpyAsync(dst, src, length * sizeof(T), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::copy_host_to_device] cudaMemcpy host to device failed", status);
            }
        }

        template <typename T>
        inline void copy_device_to_host(MemoryPool& pool, T* dst, const T* src, size_t length) {
            if (length == 0) return;
            pool.set_device();
            cudaError_t status = cudaMemcpy(dst, src, length * sizeof(T), cudaMemcpyDeviceToHost);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::copy_device_to_host] cudaMemcpy device to host failed", status);
            }
        }

        template <typename T>
        inline void copy_device_to_device(MemoryPool& pool, T* dst, const T* src, size_t length) {
            if (length == 0) return;
            pool.set_device();
            cudaError_t status = cudaMemcpyAsync(dst, src, length * sizeof(T), cudaMemcpyDeviceToDevice);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::copy_device_to_device] cudaMemcpy device to device failed", status);
            }
            utils::stream_sync();
        }

        template <typename T>
        inline void memset(MemoryPool& pool, T* ptr, size_t length, int value) {
            if (length == 0) return;
            pool.set_device();
            cudaError_t status = cudaMemsetAsync(ptr, value, length * sizeof(T));
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::memset] cudaMemset failed", status);
            }
            utils::stream_sync();
        }

        template <typename T>
        inline void memset_zero(MemoryPool& pool, T* ptr, size_t length) {
            memset(pool, ptr, length, 0);
        }

    }

}