#include "reduction.h"

namespace troy { namespace reduction {

    template<typename T>
    __global__ void kernel_max(utils::ConstSlice<T> input, utils::Slice<T> destination) {
        size_t di = threadIdx.x;
        T target = input[di];
        size_t i = di + destination.size();
        while (i < input.size()) {
            if (input[i] > target) {
                target = input[i];
            }
            i += destination.size();
        }
        destination[di] = target;
    }

    template<typename T>
    T max(utils::ConstSlice<T> inputs, MemoryPoolHandle pool) {
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
            utils::Array<T> max_array(thread_count, true, pool);
            utils::set_device(max_array.device_index());
            kernel_max<<<1, thread_count>>>(inputs, max_array.reference());
            utils::stream_sync();
            max_array.to_host_inplace();
            return max(max_array.const_reference());
        }
    }

    template<typename T>
    __global__ void kernel_min(utils::ConstSlice<T> input, utils::Slice<T> destination) {
        size_t di = threadIdx.x;
        T target = input[di];
        size_t i = di + destination.size();
        while (i < input.size()) {
            if (input[i] < target) {
                target = input[i];
            }
            i += destination.size();
        }
        destination[di] = target;
    }

    template<typename T>
    T min(utils::ConstSlice<T> inputs, MemoryPoolHandle pool) {
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
            utils::Array<T> min_array(thread_count, true, pool);
            utils::set_device(min_array.device_index());
            kernel_min<<<1, thread_count>>>(inputs, min_array.reference());
            utils::stream_sync();
            min_array.to_host_inplace();
            return min(min_array.const_reference());
        }
    }

    template<typename T>
    __global__ void kernel_sum(utils::ConstSlice<T> input, utils::Slice<T> destination) {
        size_t di = threadIdx.x;
        T target = input[di];
        size_t i = di + destination.size();
        while (i < input.size()) {
            target += input[i];
            i += destination.size();
        }
        destination[di] = target;
    }

    template<typename T>
    T sum(utils::ConstSlice<T> inputs, MemoryPoolHandle pool) {
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
            utils::Array<T> sum_array(thread_count, true, pool);
            utils::set_device(sum_array.device_index());
            kernel_sum<<<1, thread_count>>>(inputs, sum_array.reference());
            utils::stream_sync();
            sum_array.to_host_inplace();
            return sum(sum_array.const_reference());
        }
    }

    template<typename T>
    __global__ void kernel_nonzero_count(utils::ConstSlice<T> input, utils::Slice<size_t> destination) {
        size_t di = threadIdx.x;
        size_t target = input[di] != 0;
        size_t i = di + destination.size();
        while (i < input.size()) {
            target += input[i] != 0;
            i += destination.size();
        }
        destination[di] = target;
    }

    template<typename T>
    size_t nonzero_count(utils::ConstSlice<T> inputs, MemoryPoolHandle pool) {
        if (inputs.size() == 0) {
            return 0;
        }
        if (!inputs.on_device()) {
            size_t count = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                count += inputs[i] != 0;
            }
            return count;
        } else {
            size_t thread_count = utils::KERNEL_THREAD_COUNT;
            utils::Array<size_t> count_array(thread_count, true, pool);
            utils::set_device(count_array.device_index());
            kernel_nonzero_count<<<1, thread_count>>>(inputs, count_array.reference());
            utils::stream_sync();
            count_array.to_host_inplace();
            return sum(count_array.const_reference());
        }
    }

    #define IMPL_REDUCTION_FUNCTIONS(TYPE) \
        template TYPE max(utils::ConstSlice<TYPE> inputs, MemoryPoolHandle pool); \
        template TYPE min(utils::ConstSlice<TYPE> inputs, MemoryPoolHandle pool); \
        template TYPE sum(utils::ConstSlice<TYPE> inputs, MemoryPoolHandle pool); \
        template size_t nonzero_count(utils::ConstSlice<TYPE> inputs, MemoryPoolHandle pool)
    
    IMPL_REDUCTION_FUNCTIONS(float);
    IMPL_REDUCTION_FUNCTIONS(double);
    IMPL_REDUCTION_FUNCTIONS(uint32_t);
    IMPL_REDUCTION_FUNCTIONS(uint64_t);
    IMPL_REDUCTION_FUNCTIONS(__uint128_t);


    
}}