#pragma once
#include <complex>
#include <cmath>
#include "he_context.cuh"
#include "plaintext.cuh"
#include "utils/reduction.cuh"

namespace troy {

    class CKKSEncoder {

    private:
        HeContextPointer context_;
        size_t slots_;
        bool device;
        utils::Array<std::complex<double>> root_powers_;
        utils::Array<std::complex<double>> inv_root_powers_;
        utils::Array<size_t> matrix_reps_index_map;

        void encode_internal_complex_simd(const std::vector<std::complex<double>>& values, ParmsID parms_id, double scale, Plaintext& destination) const;
        void encode_internal_double_polynomial(const std::vector<double>& values, ParmsID parms_id, double scale, Plaintext& destination) const;
        void encode_internal_double_single(double value, ParmsID parms_id, double scale, Plaintext& destination) const;
        void encode_internal_integer_polynomial(const std::vector<int64_t>& values, ParmsID parms_id, Plaintext& destination) const;
        void encode_internal_integer_single(int64_t value, ParmsID parms_id, Plaintext& destination) const;
        inline void encode_internal_complex_single(std::complex<double> value, ParmsID parms_id, double scale, Plaintext& destination) const {
            std::vector<std::complex<double>> repeated(this->slot_count());
            for (size_t i = 0; i < this->slot_count(); i++) {
                repeated[i] = value;
            }
            encode_internal_complex_simd(repeated, parms_id, scale, destination);
        }

        void decode_internal_simd(const Plaintext& plain, std::vector<std::complex<double>>& destination) const;
        void decode_internal_polynomial(const Plaintext& plain, std::vector<double>& destination) const;

    public:

        CKKSEncoder(HeContextPointer context); 

        inline bool on_device() const noexcept { return device; }

        inline void to_device_inplace() {
            if (device) return;
            device = true;
            root_powers_.to_device_inplace();
            inv_root_powers_.to_device_inplace();
            matrix_reps_index_map.to_device_inplace();
        }

        inline size_t slot_count() const noexcept { return slots_; }

        inline size_t polynomial_modulus_degree() const noexcept { return slots_ * 2; }

        inline HeContextPointer context() const noexcept { return context_; }

        inline void encode_complex64_simd(
            const std::vector<std::complex<double>>& values, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& destination
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_complex_simd(values, p, scale, destination);
        }

        inline Plaintext encode_complex64_simd_new(
            const std::vector<std::complex<double>>& values,
            std::optional<ParmsID> parms_id, double scale
        ) const {
            Plaintext destination;
            encode_complex64_simd(values, parms_id, scale, destination);
            return destination;
        }

        // polynomial with constant term as value; equivalent to simd with all slots equal to value
        inline void encode_float64_single(
            double value, std::optional<ParmsID> parms_id, double scale, Plaintext& destination
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_double_single(value, p, scale, destination);
        }

        inline Plaintext encode_float64_single_new(
            double value, std::optional<ParmsID> parms_id, double scale
        ) const {
            Plaintext destination;
            encode_float64_single(value, parms_id, scale, destination);
            return destination;
        }

        inline void encode_float64_polynomial(
            const std::vector<double>& values, std::optional<ParmsID> parms_id, double scale, Plaintext& destination
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_double_polynomial(values, p, scale, destination);
        }

        inline Plaintext encode_float64_polynomial_new(
            const std::vector<double>& values, std::optional<ParmsID> parms_id, double scale
        ) const {
            Plaintext destination;
            encode_float64_polynomial(values, parms_id, scale, destination);
            return destination;
        }

        // simd with all slots equal to value
        inline void encode_complex64_single(
            std::complex<double> value, std::optional<ParmsID> parms_id, double scale, Plaintext& destination
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_complex_single(value, p, scale, destination);
        }

        inline Plaintext encode_complex64_single_new(
            std::complex<double> value, std::optional<ParmsID> parms_id, double scale
        ) const {
            Plaintext destination;
            encode_complex64_single(value, parms_id, scale, destination);
            return destination;
        }

        // polynomial with constant term as value; equivalent to simd with all slots equal to value
        inline void encode_integer64_single(
            int64_t value, std::optional<ParmsID> parms_id, Plaintext& destination
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_integer_single(value, p, destination);
        }

        inline Plaintext encode_integer64_single_new(
            int64_t value, std::optional<ParmsID> parms_id
        ) const {
            Plaintext destination;
            encode_integer64_single(value, parms_id, destination);
            return destination;
        }

        inline void encode_integer64_polynomial(
            const std::vector<int64_t>& values, std::optional<ParmsID> parms_id, Plaintext& destination
        ) const {
            ParmsID p = parms_id.value_or(context_->first_parms_id());
            encode_internal_integer_polynomial(values, p, destination);
        }

        inline Plaintext encode_integer64_polynomial_new(
            const std::vector<int64_t>& values, std::optional<ParmsID> parms_id
        ) const {
            Plaintext destination;
            encode_integer64_polynomial(values, parms_id, destination);
            return destination;
        }

        inline void decode_complex64_simd(const Plaintext& plain, std::vector<std::complex<double>>& destination) const {
            decode_internal_simd(plain, destination);
        }

        inline std::vector<std::complex<double>> decode_complex64_simd_new(const Plaintext& plain) const {
            std::vector<std::complex<double>> destination;
            decode_complex64_simd(plain, destination);
            return destination;
        }

        inline void decode_float64_polynomial(const Plaintext& plain, std::vector<double>& destination) const {
            decode_internal_polynomial(plain, destination);
        }

        inline std::vector<double> decode_float64_polynomial_new(const Plaintext& plain) const {
            std::vector<double> destination;
            decode_float64_polynomial(plain, destination);
            return destination;
        }

    };

}