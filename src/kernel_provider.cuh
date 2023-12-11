#pragma once
#include <stdexcept>
#include <cstring>
#include "utils/memory_pool.cuh"

namespace troy {

    namespace kernel_provider {

        inline void runtime_error(const char* prompt, cudaError_t status) {
            std::string msg = prompt;
            msg += ": ";
            msg += cudaGetErrorString(status);
            throw std::runtime_error(msg);
        }

        inline void initialize() {
            cudaError_t status = cudaSetDevice(0);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::initialize] cudaSetDevice failed", status);
            }
        }

        template <typename T>
        inline T* malloc(size_t length) {
            T* ret = reinterpret_cast<T*>(utils::MemoryPool::Allocate(length * sizeof(T)));
            return ret;
        }

        template <typename T>
        inline void free(T* ptr) {
            utils::MemoryPool::Free(reinterpret_cast<void*>(ptr));
        }


        template <typename T>
        inline void copy_host_to_device(T* dst, const T* src, size_t length) {
            cudaError_t status = cudaMemcpy(dst, src, length * sizeof(T), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::copy_host_to_device] cudaMemcpy host to device failed", status);
            }
        }

        template <typename T>
        inline void copy_device_to_host(T* dst, const T* src, size_t length) {
            cudaError_t status = cudaMemcpy(dst, src, length * sizeof(T), cudaMemcpyDeviceToHost);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::copy_device_to_host] cudaMemcpy device to host failed", status);
            }
        }

        template <typename T>
        inline void copy_device_to_device(T* dst, const T* src, size_t length) {
            cudaError_t status = cudaMemcpy(dst, src, length * sizeof(T), cudaMemcpyDeviceToDevice);
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::copy_device_to_device] cudaMemcpy device to device failed", status);
            }
        }

        template <typename T>
        inline void memset(T* ptr, size_t length, int value) {
            cudaError_t status = cudaMemset(ptr, value, length * sizeof(T));
            if (status != cudaSuccess) {
                runtime_error("[kernel_provider::memset] cudaMemset failed", status);
            }
        }

        template <typename T>
        inline void memset_zero(T* ptr, size_t length) {
            memset(ptr, length, 0);
        }

    }

}