#include "number_theory.h"
#include "uint_small_mod.h"

namespace troy {namespace utils {

    std::vector<int> naf(int value) {
        std::vector<int> res;
        bool sign = value < 0;
        value = std::abs(value);
        int i = 0;
        while (value > 0) {
            int zi = ((value & 1) != 0) ? (2 - (value & 3)) : 0;
            value = (value - zi) >> 1;
            if (zi != 0) {
                res.push_back((sign ? -zi : zi) << i);
            }
            i++;
        }
        return res;
    }
    
    std::vector<Modulus> get_primes(uint64_t factor, size_t bit_size, size_t count) {
        std::vector<Modulus> ret;
        // Start with (2^bit_size - 1) / factor * factor + 1
        uint64_t value = ((static_cast<uint64_t>(1) << bit_size) - 1) / factor * factor + 1;
        uint64_t lower_bound = static_cast<uint64_t>(1) << (bit_size - 1);
        while (count > 0 && value > lower_bound) {
            Modulus modulus(value);
            if (is_prime(modulus)) {
                ret.push_back(modulus);
                count--;
            }
            value -= factor;
        }
        if (count > 0) {
            throw std::logic_error("[get_primes] Failed to find enough qualifying primes.");
        }
        return ret;
    }
    
    bool try_primitive_root(uint64_t degree, const Modulus& modulus, uint64_t& destination) {
        // We need to divide modulus-1 by degree to get the size of the quotient group
        uint64_t size_entire_group = modulus.value() - 1;
        // Compute size of quotient group
        uint64_t size_quotient_group = size_entire_group / degree;
        // size_entire_group must be divisible by degree, or otherwise the primitive root does not
        // exist in integers modulo modulus
        if (size_entire_group - size_quotient_group * degree != 0) {
            return false;
        }
        size_t attempt_counter = 0;
        while (true) {
            attempt_counter += 1;
            // Set destination to be a random number modulo modulus.
            destination = modulus.reduce(rand());
            // Raise the random number to power the size of the quotient
            // to get rid of irrelevant part
            destination = exponentiate_uint64_mod(destination, size_quotient_group, modulus);
            // Stop condition
            bool cond = !is_primitive_root(destination, degree, modulus) && (attempt_counter < TRY_PRIMITIVE_ROOT_NUM_ROUNDS);
            if (!cond) {
                break;
            }
        }
        return is_primitive_root(destination, degree, modulus);
    }

    bool try_minimal_primitive_root(uint64_t degree, const Modulus& modulus, uint64_t& destination) {
        uint64_t root = 0;
        if (!try_primitive_root(degree, modulus, root)) {
            return false;
        }
        uint64_t current_generator = root;
        // destination is going to always contain the smallest generator found
        uint64_t generator_sq = multiply_uint64_mod(root, root, modulus);
        for (size_t i = 0; i < (degree + 1) / 2; i++) {
            // If our current generator is strictly smaller than destination,
            // update
            if (current_generator < root) {
                root = current_generator;
            }
            // Then move on to the next generator
            current_generator = multiply_uint64_mod(current_generator, generator_sq, modulus);
        }
        destination = root;
        return true;
    }

}}