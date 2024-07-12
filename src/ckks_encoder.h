#pragma once
#include <complex>
#include <cmath>
#include "he_context.h"
#include "plaintext.h"
#include "utils/reduction.h"

namespace troy {

    class CKKSEncoder {

    private:
        HeContextPointer context_;
        size_t slots_;
        bool device;
        utils::Array<std::complex<double>> root_powers_;
        utils::Array<std::complex<double>> inv_root_powers_;
        utils::Array<size_t> matrix_reps_index_map;

        inline bool pool_compatible(MemoryPoolHandle pool) const {
            if (this->on_device()) {
                return pool != nullptr && pool->get_device() == this->device_index();
            } else {
                return true;
            }
        }
        
        void encode_internal_complex_simd_slice(utils::ConstSlice<std::complex<double>> values, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const;
        inline void encode_internal_complex_simd(const std::vector<std::complex<double>>& values, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const {
            encode_internal_complex_simd_slice(utils::ConstSlice<std::complex<double>>(values.data(), values.size(), false, nullptr), parms_id, scale, destination, pool);
        }
        void encode_internal_double_polynomial_slice(utils::ConstSlice<double> values, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const;
        inline void encode_internal_double_polynomial(const std::vector<double>& values, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const {
            encode_internal_double_polynomial_slice(utils::ConstSlice<double>(values.data(), values.size(), false, nullptr), parms_id, scale, destination, pool);
        }
        void encode_internal_double_single(double value, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const;
        void encode_internal_integer_polynomial_slice(utils::ConstSlice<int64_t> values, ParmsID parms_id, Plaintext& destination, MemoryPoolHandle pool) const;
        inline void encode_internal_integer_polynomial(const std::vector<int64_t>& values, ParmsID parms_id, Plaintext& destination, MemoryPoolHandle pool) const {
            encode_internal_integer_polynomial_slice(utils::ConstSlice<int64_t>(values.data(), values.size(), false, nullptr), parms_id, destination, pool);
        }
        void encode_internal_integer_single(int64_t value, ParmsID parms_id, Plaintext& destination, MemoryPoolHandle pool) const;
        inline void encode_internal_complex_single(std::complex<double> value, ParmsID parms_id, double scale, Plaintext& destination, MemoryPoolHandle pool) const {
            std::vector<std::complex<double>> repeated(this->slot_count());
            for (size_t i = 0; i < this->slot_count(); i++) {
                repeated[i] = value;
            }
            encode_internal_complex_simd(repeated, parms_id, scale, destination, pool);
        }

        void decode_internal_simd_slice(const Plaintext& plain, utils::Slice<std::complex<double>> destination, MemoryPoolHandle pool) const;
        inline void decode_internal_simd(const Plaintext& plain, std::vector<std::complex<double>>& destination, MemoryPoolHandle pool) const {
            destination.resize(slot_count());
            decode_internal_simd_slice(plain, utils::Slice<std::complex<double>>(destination.data(), destination.size(), false, nullptr), pool);
        }
        void decode_internal_polynomial_slice(const Plaintext& plain, utils::Slice<double> destination, MemoryPoolHandle pool) const;
        inline void decode_internal_polynomial(const Plaintext& plain, std::vector<double>& destination, MemoryPoolHandle pool) const {
            destination.resize(slot_count() * 2);
            decode_internal_polynomial_slice(plain, utils::Slice<double>(destination.data(), destination.size(), false, nullptr), pool);
        }

    public:

        CKKSEncoder(HeContextPointer context); 

        inline bool on_device() const noexcept { return device; }
        inline size_t device_index() const {
            return matrix_reps_index_map.device_index();
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            if (device) return;
            device = true;
            root_powers_.to_device_inplace(pool);
            inv_root_powers_.to_device_inplace(pool);
            matrix_reps_index_map.to_device_inplace(pool);
        }

        inline size_t slot_count() const noexcept { return slots_; }

        inline size_t polynomial_modulus_degree() const noexcept { return slots_ * 2; }

        inline HeContextPointer context() const noexcept { return context_; }

        inline void encode_complex64_simd(
            const std::vector<std::complex<double>>& values, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_complex_simd(values, p, scale, destination, pool);
        }

        inline void encode_complex64_simd_slice(
            utils::ConstSlice<std::complex<double>> values, ParmsID parms_id, double scale, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            encode_internal_complex_simd_slice(values, parms_id, scale, destination, pool);
        }

        inline Plaintext encode_complex64_simd_new(
            const std::vector<std::complex<double>>& values,
            std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_complex64_simd(values, parms_id, scale, destination, pool);
            return destination;
        }

        inline Plaintext encode_complex64_simd_slice_new(
            utils::ConstSlice<std::complex<double>> values, ParmsID parms_id, double scale,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_complex64_simd_slice(values, parms_id, scale, destination, pool);
            return destination;
        }

        // polynomial with constant term as value; equivalent to simd with all slots equal to value
        inline void encode_float64_single(
            double value, std::optional<ParmsID> parms_id, double scale, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_double_single(value, p, scale, destination, pool);
        }

        inline Plaintext encode_float64_single_new(
            double value, std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_float64_single(value, parms_id, scale, destination, pool);
            return destination;
        }

        inline void encode_float64_polynomial(
            const std::vector<double>& values, std::optional<ParmsID> parms_id, double scale, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_double_polynomial(values, p, scale, destination, pool);
        }

        inline void encode_float64_polynomial_slice(
            utils::ConstSlice<double> values, ParmsID parms_id, double scale, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            encode_internal_double_polynomial_slice(values, parms_id, scale, destination, pool);
        }

        inline Plaintext encode_float64_polynomial_new(
            const std::vector<double>& values, std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_float64_polynomial(values, parms_id, scale, destination, pool);
            return destination;
        }

        inline Plaintext encode_float64_polynomial_slice_new(
            utils::ConstSlice<double> values, ParmsID parms_id, double scale,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_float64_polynomial_slice(values, parms_id, scale, destination, pool);
            return destination;
        }

        // simd with all slots equal to value
        inline void encode_complex64_single(
            std::complex<double> value, std::optional<ParmsID> parms_id, double scale, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_complex_single(value, p, scale, destination, pool);
        }

        inline Plaintext encode_complex64_single_new(
            std::complex<double> value, std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_complex64_single(value, parms_id, scale, destination, pool);
            return destination;
        }

        // polynomial with constant term as value; equivalent to simd with all slots equal to value
        inline void encode_integer64_single(
            int64_t value, std::optional<ParmsID> parms_id, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_integer_single(value, p, destination, pool);
        }

        inline Plaintext encode_integer64_single_new(
            int64_t value, std::optional<ParmsID> parms_id,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_integer64_single(value, parms_id, destination, pool);
            return destination;
        }

        inline void encode_integer64_polynomial(
            const std::vector<int64_t>& values, std::optional<ParmsID> parms_id, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_integer_polynomial(values, p, destination, pool);
        }

        inline void encode_integer64_polynomial_slice(
            utils::ConstSlice<int64_t> values, ParmsID parms_id, Plaintext& destination,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            encode_internal_integer_polynomial_slice(values, parms_id, destination, pool);
        }

        inline Plaintext encode_integer64_polynomial_new(
            const std::vector<int64_t>& values, std::optional<ParmsID> parms_id,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_integer64_polynomial(values, parms_id, destination, pool);
            return destination;
        }

        inline Plaintext encode_integer64_polynomial_slice_new(
            utils::ConstSlice<int64_t> values, ParmsID parms_id,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ) const {
            Plaintext destination;
            encode_integer64_polynomial_slice(values, parms_id, destination, pool);
            return destination;
        }

        inline void decode_complex64_simd(const Plaintext& plain, std::vector<std::complex<double>>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            decode_internal_simd(plain, destination, pool);
        }

        inline void decode_complex64_simd_slice(const Plaintext& plain, utils::Slice<std::complex<double>> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (destination.size() != slot_count()) {
                throw std::invalid_argument("[ckks_encoder::decode_complex64_simd_slice] destination size must be equal to slot_count.");
            }
            decode_internal_simd_slice(plain, destination, pool);
        }

        inline std::vector<std::complex<double>> decode_complex64_simd_new(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<std::complex<double>> destination;
            decode_complex64_simd(plain, destination, pool);
            return destination;
        }

        inline utils::Array<std::complex<double>> decode_complex64_simd_slice_new(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            utils::Array<std::complex<double>> destination(slot_count(), on_device(), pool);
            decode_complex64_simd_slice(plain, destination.reference(), pool);
            return destination;
        }

        inline void decode_float64_polynomial(const Plaintext& plain, std::vector<double>& destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            decode_internal_polynomial(plain, destination, pool);
        }

        inline void decode_float64_polynomial_slice(const Plaintext& plain, utils::Slice<double> destination, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (destination.size() != slot_count() * 2) {
                throw std::invalid_argument("[ckks_encoder::decode_float64_polynomial_slice] destination size must be equal to slot_count * 2.");
            }
            decode_internal_polynomial_slice(plain, destination, pool);
        }

        inline std::vector<double> decode_float64_polynomial_new(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<double> destination;
            decode_float64_polynomial(plain, destination, pool);
            return destination;
        }

        inline utils::Array<double> decode_float64_polynomial_slice_new(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            utils::Array<double> destination(slot_count() * 2, on_device(), pool);
            decode_float64_polynomial_slice(plain, destination.reference(), pool);
            return destination;
        }

    };

}