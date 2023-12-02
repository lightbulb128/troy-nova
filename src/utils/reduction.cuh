#pragma once
#include "basics.cuh"
#include "box.cuh"

namespace troy { namespace reduction {

    template<typename T>
    __global__ void kernel_find_max(utils::ConstSlice<T> input, utils::Slice<T> destination) {
        size_t di = blockIdx.x * blockDim.x + threadIdx.x;
        destination[di] = input[di];
        size_t i = di + destination.size();
        while (i < input.size()) {
            if (input[i] > destination[di]) {
                destination[di] = input[i];
            }
            i += destination.size();
        }
    }

    template<typename T>
    T max(utils::ConstSlice<T> inputs) {
        if (inputs.size() == 0) {
            return T();
        }
        if (!inputs.on_device()) {
            T max = inputs[0];
            for (size_t i = 1; i < inputs.size(); i++) {
                if (inputs[i] > max) {
                    max = inputs[i];
                }
            }
            return max;
        } else {
            size_t thread_count = utils::KERNEL_THREAD_COUNT;
            size_t block_count = utils::ceil_div(inputs.size(), thread_count);
            utils::Array<T> max_array(block_count, true);
            kernel_find_max<<<block_count, thread_count>>>(inputs, max_array.reference());
            max_array.to_host_inplace();
            return max(max_array.const_reference());
        }
    }

    template<typename T>
    __global__ void kernel_find_min(utils::ConstSlice<T> input, utils::Slice<T> destination) {
        size_t di = blockIdx.x * blockDim.x + threadIdx.x;
        destination[di] = input[di];
        size_t i = di + destination.size();
        while (i < input.size()) {
            if (input[i] < destination[di]) {
                destination[di] = input[i];
            }
            i += destination.size();
        }
    }

    template<typename T>
    T min(utils::ConstSlice<T> inputs) {
        if (inputs.size() == 0) {
            return T();
        }
        if (!inputs.on_device()) {
            T min = inputs[0];
            for (size_t i = 1; i < inputs.size(); i++) {
                if (inputs[i] < min) {
                    min = inputs[i];
                }
            }
            return min;
        } else {
            size_t thread_count = utils::KERNEL_THREAD_COUNT;
            size_t block_count = utils::ceil_div(inputs.size(), thread_count);
            utils::Array<T> min_array(block_count, true);
            kernel_find_min<<<block_count, thread_count>>>(inputs, min_array.reference());
            min_array.to_host_inplace();
            return min(min_array.const_reference());
        }
    }

    template<typename T>
    __global__ void kernel_sum(utils::ConstSlice<T> input, utils::Slice<T> destination) {
        size_t di = blockIdx.x * blockDim.x + threadIdx.x;
        destination[di] = input[di];
        size_t i = di + destination.size();
        while (i < input.size()) {
            destination[di] += input[i];
            i += destination.size();
        }
    }

    template<typename T>
    T sum(utils::ConstSlice<T> inputs) {
        if (inputs.size() == 0) {
            return T();
        }
        if (!inputs.on_device()) {
            T sum = inputs[0];
            for (size_t i = 1; i < inputs.size(); i++) {
                sum += inputs[i];
            }
            return sum;
        } else {
            size_t thread_count = utils::KERNEL_THREAD_COUNT;
            size_t block_count = utils::ceil_div(inputs.size(), thread_count);
            utils::Array<T> sum_array(block_count, true);
            kernel_sum<<<block_count, thread_count>>>(inputs, sum_array.reference());
            sum_array.to_host_inplace();
            return sum(sum_array.const_reference());
        }
    }

    template<typename T>
    __global__ void kernel_nonzero_count(utils::ConstSlice<T> input, utils::Slice<size_t> destination) {
        size_t di = blockIdx.x * blockDim.x + threadIdx.x;
        destination[di] = input[di] != 0;
        size_t i = di + destination.size();
        while (i < input.size()) {
            destination[di] += static_cast<size_t>(input[i] != 0);
            i += destination.size();
        }
    }

    template<typename T>
    size_t nonzero_count(utils::ConstSlice<T> inputs) {
        if (inputs.size() == 0) {
            return 0;
        }
        if (!inputs.on_device()) {
            size_t count = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                count += static_cast<size_t>(inputs[i] != 0);
            }
            return count;
        } else {
            size_t thread_count = utils::KERNEL_THREAD_COUNT;
            size_t block_count = utils::ceil_div(inputs.size(), thread_count);
            utils::Array<T> count_array(block_count, true);
            kernel_nonzero_count<<<block_count, thread_count>>>(inputs, count_array.reference());
            count_array.to_host_inplace();
            return sum(count_array.const_reference());
        }
    }
    
}}