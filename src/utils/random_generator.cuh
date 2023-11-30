#pragma once
#include <curand_kernel.h>
#include "box.cuh"
#include "basics.cuh"
#include "blake2/blake2.h"
#include "../modulus.cuh"
#include "poly_small_mod.cuh"

namespace troy { namespace utils {

    class RandomGenerator {

    private:
        uint64_t seed;
        uint64_t counter;
        Array<curandState_t> curand_states;

    public:

        // remove all copy and move constructors
        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator(RandomGenerator&&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
        RandomGenerator& operator=(RandomGenerator&&) = delete;

        inline RandomGenerator(uint64_t seed): seed(seed), counter(0), curand_states() {}
        inline RandomGenerator(): RandomGenerator(0) {}

        void init_curand_states(size_t count);

        inline void reset_seed(uint64_t seed) {
            this->seed = seed;
            this->counter = 0;
            if (this->curand_states.size() > 0) {
                this->init_curand_states(this->curand_states.size());
            }
        }

        void fill_bytes(Slice<uint8_t> bytes);
        void fill_uint64s(Slice<uint64_t> uint64s);

        void sample_poly_ternary(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli);
        void sample_poly_centered_binomial(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli);
        void sample_poly_uniform(Slice<uint64_t> destination, size_t degree, ConstSlice<Modulus> moduli);

    };

}}