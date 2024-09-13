#pragma once
#include "box.h"
#include "basics.h"
#include "../modulus.h"
#include "poly_small_mod.h"

namespace troy { namespace utils {

    struct ruint128_t {
        uint64_t low;
        uint64_t high;
        __host__ __device__ inline ruint128_t(uint64_t low, uint64_t high): low(low), high(high) {}
        __host__ __device__ inline ruint128_t(__uint128_t value): low(static_cast<uint64_t>(value)), high(static_cast<uint64_t>(value >> 64)) {}
        __host__ __device__ inline ruint128_t(): low(0), high(0) {}
        __host__ inline __uint128_t to_uint128() const {
            return (static_cast<__uint128_t>(this->high) << 64) + this->low;
        }
        __host__ __device__ inline ruint128_t increase() const {
            return ruint128_t(this->low + 1, this->high + (this->low == UINT64_MAX));
        }
        __host__ __device__ inline void increase_inplace() {
            this->low++;
            if (this->low == 0) {
                this->high++;
            }
        }
        __host__ __device__ inline uint8_t* as_bytes() {
            return reinterpret_cast<uint8_t*>(this);
        }
        __host__ __device__ inline const uint8_t* as_bytes() const {
            return reinterpret_cast<const uint8_t*>(this);
        }
        __host__ __device__ inline ruint128_t add(size_t value) const {
            uint64_t low = this->low + value;
            return ruint128_t(low, this->high + (low < this->low));
        }
        __host__ __device__ inline uint8_t byte_at(size_t index) const {
            return (index < 8) ? static_cast<uint8_t>(this->low >> (index * 8)) : static_cast<uint8_t>(this->high >> ((index - 8) * 8));
        }
    };

    class RandomGenerator {

    private:
        ruint128_t seed;
        ruint128_t counter;

    public:

        // remove all copy and move constructors
        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator(RandomGenerator&&) = default;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
        RandomGenerator& operator=(RandomGenerator&&) = default;

        inline RandomGenerator(__uint128_t seed): seed(seed), counter(0) {}
        inline RandomGenerator(): RandomGenerator(0) {}

        inline void reset_seed(__uint128_t seed) {
            this->seed = seed;
            this->counter = 0;
        }

        inline void set_counter(__uint128_t counter) {
            this->counter = counter;
        }
        inline __uint128_t get_counter() const {
            return this->counter.to_uint128();
        }

        void fill_bytes(Slice<uint8_t> bytes);
        void fill_bytes_batched(const SliceVec<uint8_t>& bytes, MemoryPoolHandle pool);
        /*
          Diff between batched and many
          batched: use the same seed, so the counter is added batch_size * length
          many: use different seeds for each item in batch, counter is starting from 0
        */
        static void fill_bytes_many(const ConstSlice<uint64_t> seeds, const SliceVec<uint8_t>& destination, MemoryPoolHandle pool);
        void fill_uint64s(Slice<uint64_t> uint64s);
        void fill_uint64s_batched(const SliceVec<uint64_t>& uint64s, MemoryPoolHandle pool);
        static void fill_uint64s_many(const ConstSlice<uint64_t> seeds, const SliceVec<uint64_t>& destination, MemoryPoolHandle pool);
        uint64_t sample_uint64();

        void sample_poly_ternary(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli);
        void sample_poly_centered_binomial(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli);
        void sample_poly_uniform(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli);
        
        void sample_poly_ternary_batched(const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool());
        void sample_poly_centered_binomial_batched(const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool());
        void sample_poly_uniform_batched(const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool());

        static void sample_poly_uniform_many(const ConstSlice<uint64_t> seeds, const SliceVec<uint64_t>& destination, size_t degree, ConstSlice<Modulus> moduli, MemoryPoolHandle pool = MemoryPool::GlobalPool());

    };

}}