#include "constants.h"
#include "random_generator.h"

#include "aes_impl.inc"

#define TROY_AES_DEVICE
#include "aes_impl.inc"
#undef TROY_AES_DEVICE

namespace troy {namespace utils {

    __device__ static void device_fill_uint128s(Slice<uint8_t> bytes, ruint128_t seed, ruint128_t counter) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < bytes.size() / sizeof(ruint128_t)) {
            ruint128_t value = counter.add(idx);
            aes::device::encrypt(value.as_bytes(), seed.as_bytes());
            reinterpret_cast<ruint128_t*>(bytes.raw_pointer())[idx] = value;
        }
    }

    __global__ static void kernel_fill_uint128s(Slice<uint8_t> bytes, ruint128_t seed, ruint128_t counter) {
        device_fill_uint128s(bytes, seed, counter);
    }

    __global__ static void kernel_fill_uint128s_batched(SliceArrayRef<uint8_t> bytes, ruint128_t seed, ruint128_t counter, bool skip_one) {
        size_t idx = blockIdx.y;
        ruint128_t ci = counter.add(idx * (bytes[idx].size() / sizeof(ruint128_t) + static_cast<size_t>(skip_one)));
        device_fill_uint128s(bytes[idx], seed, ci);
    }
    
    __global__ static void kernel_fill_uint128s_many(SliceArrayRef<uint8_t> bytes, ConstSlice<uint64_t> seed) {
        size_t idx = blockIdx.y;
        ruint128_t this_seed = ruint128_t(seed[idx], 0);
        device_fill_uint128s(bytes[idx], this_seed, 0);
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
            utils::set_device(bytes.device_index());
            kernel_fill_uint128s<<<block_count, utils::KERNEL_THREAD_COUNT>>>(bytes, seed, counter);
            utils::stream_sync();
            counter = counter.add(n);
        }
    }

    void fill_uint128s_batched(const SliceVec<uint8_t>& bytes, ruint128_t seed, ruint128_t& counter, MemoryPoolHandle pool, bool skip_one) {
        if (bytes.size() == 0) return;
        size_t length = bytes[0].size();
        for (size_t i = 1; i < bytes.size(); i++) {
            if (bytes[i].size() != length) {
                throw std::runtime_error("[fill_uint128s_batched] all elements of bytes must have the same size");
            }
        }
        if (bytes.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < bytes.size(); i++) {
                fill_uint128s(bytes[i], seed, counter);
            }
        } else {
            size_t block_count = utils::ceil_div(length, utils::KERNEL_THREAD_COUNT);
            auto bytes_batched = construct_batch(bytes, pool, bytes[0]);
            utils::set_device(bytes[0].device_index());
            dim3 block_dims(block_count, bytes.size());
            kernel_fill_uint128s_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                bytes_batched, seed, counter, skip_one
            );
            utils::stream_sync();
            counter = counter.add(bytes.size() * (length / sizeof(ruint128_t) + static_cast<size_t>(skip_one)));
        }
    }

    void fill_uint128s_many(const SliceVec<uint8_t>& bytes, ConstSlice<uint64_t> seed, MemoryPoolHandle pool) {
        if (bytes.size() == 0) return;
        size_t length = bytes[0].size();
        for (size_t i = 1; i < bytes.size(); i++) {
            if (bytes[i].size() != length) {
                throw std::runtime_error("[fill_uint128s_batched] all elements of bytes must have the same size");
            }
        }
        if (bytes.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < bytes.size(); i++) {
                ruint128_t counter;
                fill_uint128s(bytes[i], ruint128_t(seed[i], 0), counter);
            }
        } else {
            size_t block_count = utils::ceil_div(length, utils::KERNEL_THREAD_COUNT);
            auto bytes_batched = construct_batch(bytes, pool, bytes[0]);
            utils::set_device(bytes[0].device_index());
            dim3 block_dims(block_count, bytes.size());
            kernel_fill_uint128s_many<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                bytes_batched, seed
            );
            utils::stream_sync();
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

    __global__ static void kernel_fill_bytes_batched_tail(SliceArrayRef<uint8_t> bytes, ruint128_t seed, ruint128_t counter) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < bytes.size()) {
            Slice<uint8_t> t = bytes[idx];
            size_t interval = t.size() / sizeof(ruint128_t) + 1;
            ruint128_t value = counter.add(interval * (idx + 1) - 1);
            aes::device::encrypt(value.as_bytes(), seed.as_bytes());
            size_t tail = t.size() % sizeof(ruint128_t);
            size_t offset = t.size() - tail;
            for (size_t i = 0; i < tail; i++) {
                t[i + offset] = value.byte_at(i);
            }
        }
    }
    
    __global__ static void kernel_fill_bytes_many_tail(SliceArrayRef<uint8_t> bytes, ConstSlice<uint64_t> seeds) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < bytes.size()) {
            Slice<uint8_t> t = bytes[idx];
            size_t interval = t.size() / sizeof(ruint128_t) + 1;
            ruint128_t value = ruint128_t(0).add(interval - 1);
            ruint128_t seed = ruint128_t(seeds[idx], 0);
            aes::device::encrypt(value.as_bytes(), seed.as_bytes());
            size_t tail = t.size() % sizeof(ruint128_t);
            size_t offset = t.size() - tail;
            for (size_t i = 0; i < tail; i++) {
                t[i + offset] = value.byte_at(i);
            }
        }
    }
    
    void RandomGenerator::fill_bytes_batched(const SliceVec<uint8_t>& bytes, MemoryPoolHandle pool) {
        if (bytes.size() == 0) return;
        size_t length = bytes[0].size();
        for (size_t i = 1; i < bytes.size(); i++) {
            if (bytes[i].size() != length) {
                throw std::runtime_error("[RandomGenerator::fill_bytes_batched] all elements of bytes must have the same size");
            }
        }
        if (!bytes[0].on_device() || bytes.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < bytes.size(); i++) {
                this->fill_bytes(bytes[i]);
            }
            return;
        }
        size_t main = length / sizeof(ruint128_t);
        size_t tail = length % sizeof(ruint128_t);
        auto old_counter = this->counter;
        if (main > 0) {
            SliceVec<uint8_t> bytes_main; bytes_main.reserve(bytes.size());
            for (size_t i = 0; i < bytes.size(); i++) {
                bytes_main.push_back(Slice<uint8_t>(bytes[i].raw_pointer(), main * sizeof(ruint128_t), bytes[i].on_device(), bytes[i].pool()));
            }
            fill_uint128s_batched(bytes_main, this->seed, this->counter, pool, tail > 0);
            this->counter = old_counter;
        }
        if (tail > 0) {
            size_t block_count = utils::ceil_div(bytes.size(), utils::KERNEL_THREAD_COUNT);
            auto bytes_batched = construct_batch(bytes, pool, bytes[0]);
            utils::set_device(bytes[0].device_index());
            kernel_fill_bytes_batched_tail<<<block_count, utils::KERNEL_THREAD_COUNT>>>(bytes_batched, this->seed, this->counter);
            utils::stream_sync();
            this->counter = old_counter;
        }
        this->counter = this->counter.add(bytes.size() * (main + static_cast<size_t>(tail > 0)));
    }

    
    void RandomGenerator::fill_bytes_many(const ConstSlice<uint64_t> seeds, const SliceVec<uint8_t>& bytes, MemoryPoolHandle pool) {
        if (bytes.size() == 0) return;
        size_t length = bytes[0].size();
        for (size_t i = 1; i < bytes.size(); i++) {
            if (bytes[i].size() != length) {
                throw std::runtime_error("[RandomGenerator::fill_bytes_batched] all elements of bytes must have the same size");
            }
        }
        if (!bytes[0].on_device() || bytes.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < bytes.size(); i++) {
                RandomGenerator rng(seeds[i]);
                rng.fill_bytes(bytes[i]);
            }
            return;
        }
        size_t main = length / sizeof(ruint128_t);
        size_t tail = length % sizeof(ruint128_t);
        if (main > 0) {
            SliceVec<uint8_t> bytes_main; bytes_main.reserve(bytes.size());
            for (size_t i = 0; i < bytes.size(); i++) {
                bytes_main.push_back(Slice<uint8_t>(bytes[i].raw_pointer(), main * sizeof(ruint128_t), bytes[i].on_device(), bytes[i].pool()));
            }
            fill_uint128s_many(bytes_main, seeds, pool);
        }
        if (tail > 0) {
            size_t block_count = utils::ceil_div(bytes.size(), utils::KERNEL_THREAD_COUNT);
            auto bytes_batched = construct_batch(bytes, pool, bytes[0]);
            utils::set_device(bytes[0].device_index());
            kernel_fill_bytes_many_tail<<<block_count, utils::KERNEL_THREAD_COUNT>>>(bytes_batched, seeds);
            utils::stream_sync();
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
    
    void RandomGenerator::fill_uint64s_batched(const SliceVec<uint64_t>& uint64s, MemoryPoolHandle pool) {
        if (uint64s.size() == 0) return;
        SliceVec<uint8_t> reinterpreted; reinterpreted.reserve(uint64s.size());
        for (size_t i = 0; i < uint64s.size(); i++) {
            reinterpreted.push_back(Slice<uint8_t>(
                reinterpret_cast<uint8_t*>(uint64s[i].raw_pointer()), 
                uint64s[i].size() * sizeof(uint64_t), 
                uint64s[i].on_device(),
                uint64s[i].pool()
            ));
        }
        this->fill_bytes_batched(reinterpreted, pool);
    }

    void RandomGenerator::fill_uint64s_many(const ConstSlice<uint64_t> seeds, const SliceVec<uint64_t>& uint64s, MemoryPoolHandle pool) {
        if (uint64s.size() == 0) return;
        SliceVec<uint8_t> reinterpreted; reinterpreted.reserve(uint64s.size());
        for (size_t i = 0; i < uint64s.size(); i++) {
            reinterpreted.push_back(Slice<uint8_t>(
                reinterpret_cast<uint8_t*>(uint64s[i].raw_pointer()), 
                uint64s[i].size() * sizeof(uint64_t), 
                uint64s[i].on_device(),
                uint64s[i].pool()
            ));
        }
        RandomGenerator::fill_bytes_many(seeds, reinterpreted, pool);
    }
    
    uint64_t RandomGenerator::sample_uint64() {
        ruint128_t value = host_generate_uint128(this->seed, this->counter);
        return value.low;
    }

    __device__ static void device_sample_poly_ternary(
        Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < degree) {
            size_t counter_offset = idx / 16;
            uint8_t r = device_generate_uint128(seed, counter.add(counter_offset)).byte_at(idx & 15) % 3;
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

    __global__ static void kernel_sample_poly_ternary(
        Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        device_sample_poly_ternary(destination, degree, moduli, seed, counter);
    }

    __global__ static void kernel_sample_poly_ternary_batched(
        SliceArrayRef<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        size_t idx = blockIdx.y;
        ruint128_t ci = counter.add((degree + 15ul) / 16ul * idx);
        device_sample_poly_ternary(destination[idx], degree, moduli, seed, ci);
    }
    
    void RandomGenerator::sample_poly_ternary(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli) {
        bool device = destination.on_device();
        if (!utils::device_compatible(destination, moduli)) {
            throw std::runtime_error("[RandomGenerator::sample_poly_ternary] destination and modulus must be on the same device");
        }
        if (!device) {
            size_t byte_at = 0;
            ruint128_t full_word;
            for (size_t j = 0; j < degree; j++) {
                if (byte_at == 0) {
                    full_word = host_generate_uint128(this->seed, this->counter);
                }
                uint8_t r = full_word.byte_at(byte_at) % 3;
                byte_at = (byte_at + 1) & 15;
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
            utils::set_device(destination.device_index());
            kernel_sample_poly_ternary<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                destination, degree, moduli,
                this->seed, this->counter
            );
            utils::stream_sync();
            this->counter = this->counter.add(utils::ceil_div(degree, 16ul));
        }
    }
    
    void RandomGenerator::sample_poly_ternary_batched(const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool) {
        if (destination.size() == 0) return;
        if (!moduli.on_device() || destination.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < destination.size(); i++) {
                this->sample_poly_ternary(destination[i], degree, moduli);
            }
        } else {
            size_t block_count = utils::ceil_div(degree, utils::KERNEL_THREAD_COUNT);
            auto destination_batched = construct_batch(destination, pool, moduli);
            utils::set_device(moduli.device_index());
            dim3 block_dims(block_count, destination.size());
            kernel_sample_poly_ternary_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                destination_batched, degree, moduli,
                this->seed, this->counter
            );
            utils::stream_sync();
            this->counter = this->counter.add(utils::ceil_div(degree, 16ul) * destination.size());
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

    __device__ static void device_sample_poly_centered_binomial(
        Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < degree) {
            size_t counter_offset = idx / 2;
            ruint128_t rf = device_generate_uint128(seed, counter.add(counter_offset));
            int r = uint64_to_cbd((idx & 1) ? (rf.high) : (rf.low));
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

    __global__ static void kernel_sample_poly_centered_binomial(
        Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        device_sample_poly_centered_binomial(destination, degree, moduli, seed, counter);
    }

    __global__ static void kernel_sample_poly_centered_binomial_batched(
        SliceArrayRef<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli,
        ruint128_t seed, ruint128_t counter
    ) {
        size_t idx = blockIdx.y;
        ruint128_t ci = counter.add((degree + 1ul) / 2ul * idx);
        device_sample_poly_centered_binomial(destination[idx], degree, moduli, seed, ci);
    }

    void RandomGenerator::sample_poly_centered_binomial(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli) {
        bool device = destination.on_device();
        if (!utils::device_compatible(destination, moduli)) {
            throw std::runtime_error("[RandomGenerator::sample_poly_centered_binomial] destination and modulus must be on the same device");
        }
        if (!device) {
            ruint128_t full_word;
            for (size_t j = 0; j < degree; j++) {
                if (!(j & 1)) {
                    full_word = host_generate_uint128(this->seed, this->counter);
                }
                int r = uint64_to_cbd((j & 1) ? (full_word.high) : (full_word.low));
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
            utils::set_device(destination.device_index());
            kernel_sample_poly_centered_binomial<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                destination, degree, moduli, this->seed, this->counter
            );
            utils::stream_sync();
            this->counter = this->counter.add(utils::ceil_div(degree, 2ul));
        }
    }

    void RandomGenerator::sample_poly_centered_binomial_batched(const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool) {
        if (destination.size() == 0) return;
        if (!moduli.on_device() || destination.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < destination.size(); i++) {
                this->sample_poly_centered_binomial(destination[i], degree, moduli);
            }
        } else {
            size_t block_count = utils::ceil_div(degree, utils::KERNEL_THREAD_COUNT);
            auto destination_batched = construct_batch(destination, pool, moduli);
            utils::set_device(moduli.device_index());
            dim3 block_dims(block_count, destination.size());
            kernel_sample_poly_centered_binomial_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(
                destination_batched, degree, moduli,
                this->seed, this->counter
            );
            utils::stream_sync();
            this->counter = this->counter.add(utils::ceil_div(degree, 2ul) * destination.size());
        }
    }

    void RandomGenerator::sample_poly_uniform(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli) {
        if (!utils::device_compatible(destination, moduli)) {
            throw std::runtime_error("[RandomGenerator::sample_poly_uniform] destination and modulus must be on the same device");
        }
        this->fill_uint64s(destination);
        utils::modulo_inplace_p(destination, degree, moduli);
    }

    void RandomGenerator::sample_poly_uniform_batched(const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool) {
        if (!moduli.on_device() ||  destination.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < destination.size(); i++) {
                this->sample_poly_uniform(destination[i], degree, moduli);
            }
        } else {
            this->fill_uint64s_batched(destination, pool);
            utils::modulo_inplace_bp(destination, degree, moduli, pool);
        }
    }
    
    void RandomGenerator::sample_poly_uniform_many(const ConstSlice<uint64_t> seeds, const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool) {
        if (!moduli.on_device() || destination.size() < BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < destination.size(); i++) {
                RandomGenerator rng(seeds[i]);
                rng.sample_poly_uniform(destination[i], degree, moduli);
            }
        } else {
            fill_uint64s_many(seeds, destination, pool);
            utils::modulo_inplace_bp(destination, degree, moduli, pool);
        }
    }

}}