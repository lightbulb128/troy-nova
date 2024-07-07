#include "random_generator.h"

#include "aes_impl.inc"

#define TROY_AES_DEVICE
#include "aes_impl.inc"
#undef TROY_AES_DEVICE

namespace troy {namespace utils {

    __global__ static void kernel_fill_uint128s(Slice<uint8_t> bytes, ruint128_t seed, ruint128_t counter) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < bytes.size() / sizeof(ruint128_t)) {
            ruint128_t value = counter.add(idx);
            aes::device::encrypt(value.as_bytes(), seed.as_bytes());
            reinterpret_cast<ruint128_t*>(bytes.raw_pointer())[idx] = value;
        }
    }

    void fill_uint128s(Slice<uint8_t> bytes, ruint128_t seed, ruint128_t& counter) {
#if DEBUG
        if (bytes.size() % sizeof(ruint128_t) != 0) {
            throw std::runtime_error("[fill_uint128s] bytes.len() must be a multiple of sizeof(ruint128_t)");
        }
#endif
        size_t n = bytes.size() / sizeof(ruint128_t);
        if (!bytes.on_device()) {
            ruint128_t* pointer = reinterpret_cast<ruint128_t*>(bytes.raw_pointer());
            for (size_t i = 0; i < n; i++) {
                *pointer = counter;
                aes::host::encrypt(pointer->as_bytes(), seed.as_bytes());
                counter.increase_inplace();
                pointer++;
            }
        } else {
            size_t block_count = utils::ceil_div(n, utils::KERNEL_THREAD_COUNT);
            kernel_fill_uint128s<<<block_count, utils::KERNEL_THREAD_COUNT>>>(bytes, seed, counter);
            cudaStreamSynchronize(0);
            counter = counter.add(n);
        }
    }

    ruint128_t host_generate_uint128(ruint128_t seed, ruint128_t& counter) {
        ruint128_t value = counter;
        aes::host::encrypt(value.as_bytes(), seed.as_bytes());
        counter.increase_inplace();
        return value;
    }

    __device__ ruint128_t device_generate_uint128(const ruint128_t& seed, const ruint128_t& counter) {
        ruint128_t value = counter;
        aes::device::encrypt(value.as_bytes(), seed.as_bytes());
        return value;
    }
    
    void RandomGenerator::fill_bytes(Slice<uint8_t> bytes) {
        size_t main = bytes.size() / sizeof(ruint128_t);
        size_t tail = bytes.size() % sizeof(ruint128_t);
        if (main > 0) {
            fill_uint128s(bytes.slice(0, main * sizeof(ruint128_t)), this->seed, this->counter);
        }
        if (tail > 0) {
            ruint128_t value = host_generate_uint128(this->seed, this->counter);
            Slice<uint8_t> tail_slice = bytes.slice(main * sizeof(ruint128_t), bytes.size());
            tail_slice.copy_from_slice(ConstSlice<uint8_t>(reinterpret_cast<uint8_t*>(&value), tail, false, nullptr));
        }
    }

    void RandomGenerator::fill_uint64s(Slice<uint64_t> uint64s) {
        this->fill_bytes(
            Slice<uint8_t>(
                reinterpret_cast<uint8_t*>(uint64s.raw_pointer()), 
                uint64s.size() * sizeof(uint64_t), 
                uint64s.on_device(),
                uint64s.pool()
            )
        );
    }
    
    uint64_t RandomGenerator::sample_uint64() {
        ruint128_t value = host_generate_uint128(this->seed, this->counter);
        return value.low;
    }

    __global__ static void kernel_sample_poly_ternary(
        Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < degree) {
            uint8_t r = device_generate_uint128(seed, counter.add(idx)).low % 3;
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
        if (!utils::device_compatible(destination, moduli)) {
            throw std::runtime_error("[RandomGenerator::sample_poly_ternary] destination and modulus must be on the same device");
        }
        if (!device) {
            Array<uint8_t> buffer(degree, false, nullptr); 
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
            size_t block_count = utils::ceil_div(degree, utils::KERNEL_THREAD_COUNT);
            kernel_sample_poly_ternary<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                destination, degree, moduli,
                this->seed, this->counter
            );
            cudaStreamSynchronize(0);
            this->counter = this->counter.add(degree);
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
        Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < degree) {
            int r = uint64_to_cbd(device_generate_uint128(seed, counter.add(idx)).low);
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
        if (!utils::device_compatible(destination, moduli)) {
            throw std::runtime_error("[RandomGenerator::sample_poly_centered_binomial] destination and modulus must be on the same device");
        }
        if (!device) {
            Array<uint64_t> buffer(degree, false, nullptr);
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
            size_t block_count = utils::ceil_div(degree, utils::KERNEL_THREAD_COUNT);
            kernel_sample_poly_centered_binomial<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                destination, degree, moduli, this->seed, this->counter
            );
            cudaStreamSynchronize(0);
            this->counter = this->counter.add(degree);
        }
    }

    void RandomGenerator::sample_poly_uniform(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli) {
        if (!utils::device_compatible(destination, moduli)) {
            throw std::runtime_error("[RandomGenerator::sample_poly_uniform] destination and modulus must be on the same device");
        }
        this->fill_uint64s(destination);
        utils::modulo_inplace_p(destination, degree, moduli);
    }

}}