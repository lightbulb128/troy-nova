#include "random_generator.cuh"

namespace troy {namespace utils {

    __global__ static void kernel_init_curand_states(Slice<curandState_t> states, size_t count, uint64_t seed) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            curand_init(idx + seed, 0, 0, &states[idx]);
        }
    }

    void RandomGenerator::init_curand_states(size_t count) {
        this->curand_states = Array<curandState_t>(count, true);
        size_t block_count = utils::ceil_div(count, utils::KERNEL_THREAD_COUNT);
        kernel_init_curand_states<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
            this->curand_states.reference(), count, this->seed);
    }

    __global__ static void kernel_fill_bytes(
        Slice<uint8_t> bytes, Slice<curandState_t> states
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < states.size()) {
            curandState_t& state = states[idx];
            for (size_t index = idx; index < bytes.size(); index += states.size()) {
                bytes[index] = static_cast<uint8_t>(curand(&state));
            }
        }
    }
    
    void RandomGenerator::fill_bytes(Slice<uint8_t> bytes) {
        if (bytes.on_device()) {
            size_t block_count = utils::ceil_div(this->curand_states.size(), utils::KERNEL_THREAD_COUNT);
            kernel_fill_bytes<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                bytes, this->curand_states.reference()
            );
        } else {
            blake2xb(
                bytes.raw_pointer(), bytes.size(),
                &this->counter, sizeof(this->counter),
                &this->seed, sizeof(this->seed)
            );
            this->counter++;
        }
    }

    __device__ inline
    uint64_t generate_uint64(curandState_t& state) {
        return (static_cast<uint64_t>(curand(&state)))
            + (static_cast<uint64_t>(curand(&state)) << 32);
    }

    __global__ static void kernel_fill_uint64s(
        Slice<uint64_t> uint64s, Slice<curandState_t> states
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < states.size()) {
            curandState_t& state = states[idx];
            for (size_t index = idx; index < uint64s.size(); index += states.size()) {
                uint64s[index] = generate_uint64(state);
            }
        }
    }

    void RandomGenerator::fill_uint64s(Slice<uint64_t> uint64s) {
        if (uint64s.on_device()) {
            size_t block_count = utils::ceil_div(this->curand_states.size(), utils::KERNEL_THREAD_COUNT);
            kernel_fill_uint64s<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                uint64s, this->curand_states.reference()
            );
        } else {
            blake2xb(
                uint64s.raw_pointer(), uint64s.size() * sizeof(uint64_t),
                &this->counter, sizeof(this->counter),
                &this->seed, sizeof(this->seed)
            );
            this->counter++;
        }
    }

    __global__ static void kernel_sample_poly_ternary(
        Slice<uint64_t> destination, size_t degree, Slice<curandState_t> states, ConstSlice<Modulus> moduli
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < degree) {
            curandState_t& state = states[idx];
            uint8_t r = curand(&state) % 3;
            size_t i = 0;
            for (size_t index = idx; index < destination.size(); index += degree) {
                if (r == 2) {
                    destination[index] = moduli[i].value() - 1;
                } else {
                    destination[index] = r;
                }
                i++;
            }
        }
    }
    
    void RandomGenerator::sample_poly_ternary(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli) {
        bool device = destination.on_device();
        if (device != moduli.on_device()) {
            throw std::runtime_error("[RandomGenerator::sample_poly_ternary] destination and modulus must be on the same device");
        }
        if (!device) {
            Array<uint8_t> buffer(degree, false); 
            this->fill_bytes(buffer.reference());
            for (size_t j = 0; j < degree; j++) {
                uint8_t r = buffer[j] % 3;
                for (size_t i = 0; i < moduli.size(); i++) {
                    size_t index = i * degree + j;
                    if (r == 2) {
                        destination[index] = moduli[i].value() - 1;
                    } else {
                        destination[index] = r;
                    }
                }
            }
        } else {
            if (degree != this->curand_states.size()) {
                throw std::runtime_error("[RandomGenerator::sample_poly_ternary] degree must be equal to the number of curand states");
            }
            size_t block_count = utils::ceil_div(degree, utils::KERNEL_THREAD_COUNT);
            kernel_sample_poly_ternary<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                destination, degree, this->curand_states.reference(), moduli
            );
        }
    }

    __host__ __device__
    static int uint64_to_cbd(uint64_t x_uint64) {
        uint8_t* x = reinterpret_cast<uint8_t*>(&x_uint64);
        x[2] &= 0x1f; x[5] &= 0x1f;
        return 
            utils::hamming_weight(x[0])
            + utils::hamming_weight(x[1])
            + utils::hamming_weight(x[2])
            - utils::hamming_weight(x[3])
            - utils::hamming_weight(x[4])
            - utils::hamming_weight(x[5]);
    }

    __global__ static void kernel_sample_poly_centered_binomial(
        Slice<uint64_t> destination, size_t degree, Slice<curandState_t> states, ConstSlice<Modulus> moduli
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < degree) {
            curandState_t& state = states[idx];
            int r = uint64_to_cbd(generate_uint64(state));
            size_t i = 0;
            for (size_t index = idx; index < destination.size(); index += degree) {
                if (r >= 0) {
                    destination[index] = r;
                } else {
                    destination[index] = moduli[i].value() + r;
                }
                i++;
            }
        }
    }

    void RandomGenerator::sample_poly_centered_binomial(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli) {
        bool device = destination.on_device();
        if (device != moduli.on_device()) {
            throw std::runtime_error("[RandomGenerator::sample_poly_centered_binomial] destination and modulus must be on the same device");
        }
        if (!device) {
            Array<uint64_t> buffer(degree, false);
            this->fill_uint64s(buffer.reference());
            for (size_t j = 0; j < degree; j++) {
                int r = uint64_to_cbd(buffer[j]);
                for (size_t i = 0; i < moduli.size(); i++) {
                    size_t index = i * degree + j;
                    if (r >= 0) {
                        destination[index] = r;
                    } else {
                        destination[index] = moduli[i].value() + r;
                    }
                }
            }
        } else {
            if (degree != this->curand_states.size()) {
                throw std::runtime_error("[RandomGenerator::sample_poly_centered_binomial] degree must be equal to the number of curand states");
            }
            size_t block_count = utils::ceil_div(degree, utils::KERNEL_THREAD_COUNT);
            kernel_sample_poly_centered_binomial<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                destination, degree, this->curand_states.reference(), moduli
            );
        }
    }

    void RandomGenerator::sample_poly_uniform(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli) {
        bool device = destination.on_device();
        if (device != moduli.on_device()) {
            throw std::runtime_error("[RandomGenerator::sample_poly_uniform] destination and modulus must be on the same device");
        }
        if (device) {
            if (degree != this->curand_states.size()) {
                throw std::runtime_error("[RandomGenerator::sample_poly_uniform] degree must be equal to the number of curand states");
            }
        }
        this->fill_uint64s(destination);
        utils::modulo_inplace_p(destination, degree, moduli);
    }

}}