#pragma once

#include "cuda_runtime.h"
#include "../src/he_context.h"
#include "../src/batch_encoder.h"
#include "../src/ckks_encoder.h"
#include "../src/evaluator.h"
#include "../src/encryptor.h"
#include "../src/key_generator.h"
#include "../src/decryptor.h"
#include "../src/app/bfv_ring2k.h"

namespace tool {

    using uint128_t = __uint128_t;
    using namespace troy;
    using troy::utils::ConstSlice;
    using troy::utils::Slice;
    using troy::linear::PolynomialEncoderRing2k;
    using std::vector;
    using std::optional;
    using std::complex;

    inline uint128_t random_uint128() {
        return (static_cast<uint128_t>(rand()) << 64) | rand();
    }

    template<typename T>
    ConstSlice<T> sfv(const vector<T> &vec) {
        return ConstSlice<T>(vec.data(), vec.size(), false, nullptr);
    }

    class GeneralVector {
    private:
        optional<vector<complex<double>>> complexes_;
        optional<vector<uint64_t>> integers_;
        optional<vector<double>> doubles_;
        optional<vector<uint32_t>> uint32s_;
        optional<vector<uint64_t>> uint64s_;
        optional<vector<uint128_t>> uint128s_;
    public:

        inline GeneralVector() {}
        inline GeneralVector(vector<complex<double>>&& complexes): complexes_(std::move(complexes)) {}
        inline GeneralVector(vector<uint64_t>&& integers, bool is_ring) {
            if (is_ring) {
                uint64s_ = std::move(integers);
            } else {
                integers_ = std::move(integers);
            }
        }
        inline GeneralVector(vector<double>&& doubles): doubles_(std::move(doubles)) {}
        inline GeneralVector(vector<uint32_t>&& uint32s): uint32s_(std::move(uint32s)) {}
        inline GeneralVector(vector<uint128_t>&& uint128s): uint128s_(std::move(uint128s)) {}

        void resize(size_t size) {
            if (complexes_) {
                complexes_->resize(size);
            } else if (integers_) {
                integers_->resize(size);
            } else if (doubles_) {
                doubles_->resize(size);
            } else if (uint32s_) {
                uint32s_->resize(size);
            } else if (uint64s_) {
                uint64s_->resize(size);
            } else if (uint128s_) {
                uint128s_->resize(size);
            } else {
                throw std::invalid_argument("[GeneralVector::resize] Cannot resize empty vector");
            }
        }

        inline static GeneralVector random_complexes(size_t size, double component_max_absolute) {
            vector<complex<double>> vec(size);
            for (size_t i = 0; i < size; i++) {
                double real = (double)rand() / (double)RAND_MAX * 2 * component_max_absolute - component_max_absolute;
                double imag = (double)rand() / (double)RAND_MAX * 2 * component_max_absolute - component_max_absolute;
                vec[i] = complex<double>(real, imag);
            }
            return GeneralVector(std::move(vec));
        }

        inline static GeneralVector random_complex_repeated(size_t size, double component_max_absolute) {
            // random sample one and repeat size
            double real = (double)rand() / (double)RAND_MAX * 2 * component_max_absolute - component_max_absolute;
            double imag = (double)rand() / (double)RAND_MAX * 2 * component_max_absolute - component_max_absolute;
            return GeneralVector(vector<complex<double>>(size, complex<double>(real, imag)));
        }

        inline static GeneralVector random_integers(size_t size, uint64_t modulus) {
            vector<uint64_t> vec(size);
            for (size_t i = 0; i < size; i++) {
                vec[i] = rand() % modulus;
            }
            return GeneralVector(std::move(vec), false);
        }

        inline static GeneralVector random_integer_repeated(size_t size, uint64_t modulus) {
            // random sample one and repeat size
            uint64_t value = rand() % modulus;
            return GeneralVector(vector<uint64_t>(size, value), false);
        }

        inline static GeneralVector random_doubles(size_t size, double max_absolute) {
            vector<double> vec(size);
            for (size_t i = 0; i < size; i++) {
                vec[i] = (double)rand() / (double)RAND_MAX * 2 * max_absolute - max_absolute;
            }
            return GeneralVector(std::move(vec));
        }

        inline static GeneralVector random_double_repeated(size_t size, double max_absolute) {
            // random sample one and repeat size
            double value = (double)rand() / (double)RAND_MAX * 2 * max_absolute - max_absolute;
            return GeneralVector(vector<double>(size, value));
        }
        
        inline static GeneralVector random_uint32s(size_t size, uint32_t mask) {
            vector<uint32_t> vec(size);
            for (size_t i = 0; i < size; i++) {
                vec[i] = rand() & mask;
            }
            return GeneralVector(std::move(vec));
        }

        inline static GeneralVector random_uint32_repeated(size_t size, uint32_t mask) {
            // random sample one and repeat size
            uint32_t value = rand() & mask;
            return GeneralVector(vector<uint32_t>(size, value));
        }

        inline static GeneralVector random_uint64s(size_t size, uint64_t mask) {
            vector<uint64_t> vec(size);
            for (size_t i = 0; i < size; i++) {
                vec[i] = rand() & mask;
            }
            return GeneralVector(std::move(vec), true);
        }

        inline static GeneralVector random_uint64_repeated(size_t size, uint64_t mask) {
            // random sample one and repeat size
            uint64_t value = rand() & mask;
            return GeneralVector(vector<uint64_t>(size, value), true);
        }

        inline static GeneralVector random_uint128s(size_t size, uint128_t mask) {
            vector<uint128_t> vec(size);
            for (size_t i = 0; i < size; i++) {
                vec[i] = random_uint128() & mask;
            }
            return GeneralVector(std::move(vec));
        }

        inline static GeneralVector random_uint128_repeated(size_t size, uint128_t mask) {
            // random sample one and repeat size
            uint128_t value = random_uint128() & mask;
            return GeneralVector(vector<uint128_t>(size, value));
        }

        inline static GeneralVector zeros_double(size_t size) {
            return GeneralVector(vector<double>(size, 0));
        }

        inline static GeneralVector zeros_integer(size_t size) {
            return GeneralVector(vector<uint64_t>(size, 0), false);
        }

        inline static GeneralVector zeros_complex(size_t size) {
            return GeneralVector(vector<complex<double>>(size, 0));
        }

        inline static GeneralVector zeros_uint32(size_t size) {
            return GeneralVector(vector<uint32_t>(size, 0));
        }

        inline static GeneralVector zeros_uint64(size_t size) {
            return GeneralVector(vector<uint64_t>(size, 0), true);
        }

        inline static GeneralVector zeros_uint128(size_t size) {
            return GeneralVector(vector<uint128_t>(size, 0));
        }

        inline static GeneralVector zeros_like(const GeneralVector& vec, size_t size) {
            if (vec.is_complexes()) {
                return zeros_complex(size);
            } else if (vec.is_integers()) {
                return zeros_integer(size);
            } else if (vec.is_doubles()) {
                return zeros_double(size);
            } else if (vec.is_uint32s()) {
                return zeros_uint32(size);
            } else if (vec.is_uint64s()) {
                return zeros_uint64(size);
            } else if (vec.is_uint128s()) {
                return zeros_uint128(size);
            } else {
                throw std::invalid_argument("[GeneralVector::zeros_like] Cannot zeros_like empty vector");
            }
        }

        inline GeneralVector subvector(size_t low, size_t high) const {
            if (complexes_) {
                return GeneralVector(vector<complex<double>>(complexes_->begin() + low, complexes_->begin() + high));
            } else if (integers_) {
                return GeneralVector(vector<uint64_t>(integers_->begin() + low, integers_->begin() + high), false);
            } else if (doubles_) {
                return GeneralVector(vector<double>(doubles_->begin() + low, doubles_->begin() + high));
            } else if (uint32s_) {
                return GeneralVector(vector<uint32_t>(uint32s_->begin() + low, uint32s_->begin() + high));
            } else if (uint64s_) {
                return GeneralVector(vector<uint64_t>(uint64s_->begin() + low, uint64s_->begin() + high), true);
            } else if (uint128s_) {
                return GeneralVector(vector<uint128_t>(uint128s_->begin() + low, uint128s_->begin() + high));
            } else {
                throw std::invalid_argument("[GeneralVector::subvector] Cannot subvector empty vector");
            }
        }

        inline GeneralVector element(size_t index) const {
            return subvector(index, index + 1);
        }

        inline GeneralVector get(size_t index) const {
            return element(index);
        }

        inline void set(size_t index, const GeneralVector& element) {
            if (element.size() != 1) throw std::invalid_argument("[GeneralVector::set] element.size() != 1");
            if (complexes_) {
                complexes_->at(index) = element.complexes().at(0);
            } else if (integers_) {
                integers_->at(index) = element.integers().at(0);
            } else if (doubles_) {
                doubles_->at(index) = element.doubles().at(0);
            } else if (uint32s_) {
                uint32s_->at(index) = element.uint32s().at(0);
            } else if (uint64s_) {
                uint64s_->at(index) = element.uint64s().at(0);
            } else if (uint128s_) {
                uint128s_->at(index) = element.uint128s().at(0);
            } else {
                throw std::invalid_argument("[GeneralVector::set] Cannot set empty vector");
            }
        }

        inline bool is_complexes() const {
            return complexes_.has_value();
        }
        inline bool is_integers() const {
            return integers_.has_value();
        }
        inline bool is_doubles() const {
            return doubles_.has_value();
        }
        inline bool is_uint32s() const {
            return uint32s_.has_value();
        }
        inline bool is_uint64s() const {
            return uint64s_.has_value();
        }
        inline bool is_uint128s() const {
            return uint128s_.has_value();
        }
        inline size_t size() const {
            if (complexes_) {
                return complexes_->size();
            } else if (integers_) {
                return integers_->size();
            } else if (doubles_) {
                return doubles_->size();
            } else if (uint32s_) {
                return uint32s_->size();
            } else if (uint64s_) {
                return uint64s_->size();
            } else if (uint128s_) {
                return uint128s_->size();
            } else {
                throw std::invalid_argument("[GeneralVector::size] Cannot get size of empty vector");
            }
        }

        inline vector<complex<double>>& complexes() {
            return *complexes_;
        }
        inline vector<uint64_t>& integers() {
            return *integers_;
        }
        inline vector<double>& doubles() {
            return *doubles_;
        }
        inline vector<uint32_t>& uint32s() {
            return *uint32s_;
        }
        inline vector<uint64_t>& uint64s() {
            return *uint64s_;
        }
        inline vector<uint128_t>& uint128s() {
            return *uint128s_;
        }
        
        inline const vector<complex<double>>& complexes() const {
            return *complexes_;
        }
        inline const vector<uint64_t>& integers() const {
            return *integers_;
        }
        inline const vector<double>& doubles() const {
            return *doubles_;
        }
        inline const vector<uint32_t>& uint32s() const {
            return *uint32s_;
        }
        inline const vector<uint64_t>& uint64s() const {
            return *uint64s_;
        }
        inline const vector<uint128_t>& uint128s() const {
            return *uint128s_;
        }

        inline GeneralVector negate(uint64_t modulus) const {
            if (complexes_) {
                vector<complex<double>> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = -complexes_->at(i);
                }
                return GeneralVector(std::move(vec));
            } else if (integers_) {
                vector<uint64_t> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = integers_->at(i) == 0 ? 0 : modulus - integers_->at(i);
                }
                return GeneralVector(std::move(vec), false);
            } else if (doubles_) {
                vector<double> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = -doubles_->at(i);
                }
                return GeneralVector(std::move(vec));
            } else {
                throw std::invalid_argument("[GeneralVector::negate] Cannot negate empty vector");
            }
        }

        template <typename T>
        void ring2k_check() const {
            if constexpr (std::is_same_v<T, uint32_t>) {
                if (!uint32s_) {
                    throw std::invalid_argument("[GeneralVector::ring2k_check] Empty uint32_t vector");
                }
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                if (!uint64s_) {
                    throw std::invalid_argument("[GeneralVector::ring2k_check] Empty uint64_t vector");
                }
            } else if constexpr (std::is_same_v<T, uint128_t>) {
                if (!uint128s_) {
                    throw std::invalid_argument("[GeneralVector::ring2k_check] Empty uint128_t vector");
                }
            } else {
                throw std::invalid_argument("[GeneralVector::ring2k_check] Unsupported type");
            }
        }

        template <typename T>
        inline T ring2k_at(size_t index) const {
            ring2k_check<T>();
            if constexpr (std::is_same_v<T, uint32_t>) {
                return uint32s_->at(index);
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                return uint64s_->at(index);
            } else if constexpr (std::is_same_v<T, uint128_t>) {
                return uint128s_->at(index);
            } else {
                throw std::invalid_argument("[GeneralVector::ring2k_at] Unsupported type");
            }
        }

        template <typename T>
        inline GeneralVector ring2k_negate(T mask) const {
            ring2k_check<T>();
            vector<T> vec(size());
            for (size_t i = 0; i < size(); i++) {
                vec[i] = (-ring2k_at<T>(i)) & mask;
            }
            if constexpr (std::is_same_v<T, uint64_t>) {
                return GeneralVector(std::move(vec), true);
            } else {
                return GeneralVector(std::move(vec));
            }
        }

        inline GeneralVector add(const GeneralVector& other, uint64_t modulus) const {
            if (complexes_) {
                vector<complex<double>> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = complexes_->at(i) + other.complexes_->at(i);
                }
                return GeneralVector(std::move(vec));
            } else if (integers_) {
                vector<uint64_t> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = (integers_->at(i) + other.integers_->at(i)) % modulus;
                }
                return GeneralVector(std::move(vec), false);
            } else if (doubles_) {
                vector<double> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = doubles_->at(i) + other.doubles_->at(i);
                }
                return GeneralVector(std::move(vec));
            } else {
                throw std::invalid_argument("[GeneralVector::add] Cannot add empty vector");
            }
        }

        template <typename T>
        inline GeneralVector ring2k_add(const GeneralVector& other, T mask) const {
            ring2k_check<T>();
            vector<T> vec(size());
            for (size_t i = 0; i < size(); i++) {
                vec[i] = (ring2k_at<T>(i) + other.ring2k_at<T>(i)) & mask;
            }
            if constexpr (std::is_same_v<T, uint64_t>) {
                return GeneralVector(std::move(vec), true);
            } else {
                return GeneralVector(std::move(vec));
            }
        }

        inline GeneralVector sub(const GeneralVector& other, uint64_t modulus) const {
            return this->add(other.negate(modulus), modulus);
        }

        template <typename T>
        inline GeneralVector ring2k_sub(const GeneralVector& other, T mask) const {
            return this->ring2k_add(other.ring2k_negate(mask), mask);
        }

        inline GeneralVector mul(const GeneralVector& other, uint64_t modulus) const {
            if (complexes_) {
                vector<complex<double>> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = complexes_->at(i) * other.complexes_->at(i);
                }
                return GeneralVector(std::move(vec));
            } else if (integers_) {
                vector<uint64_t> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    // switch to u128 before mod
                    __uint128_t tmp = static_cast<__uint128_t>(integers_->at(i)) * static_cast<__uint128_t>(other.integers_->at(i));
                    vec[i] = static_cast<uint64_t>(tmp % modulus);
                }
                return GeneralVector(std::move(vec), false);
            } else if (doubles_) {
                vector<double> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = doubles_->at(i) * other.doubles_->at(i);
                }
                return GeneralVector(std::move(vec));
            } else {
                throw std::invalid_argument("[GeneralVector::mul] Cannot mul empty vector");
            }
        }

        inline GeneralVector mul_poly(const GeneralVector& other, size_t degree, uint64_t modulus) const {
            if (size() > degree || other.size() > degree) {
                throw std::invalid_argument("[GeneralVector::mul_poly] Degree too small");
            }
            if (integers_) {
                vector<uint64_t> vec(std::min(degree, size() + other.size() - 1), 0);
                for (size_t i = 0; i < size(); i++) for (size_t j = 0; j < other.size(); j++) {
                    // switch to u128 before mod
                    __uint128_t tmp = static_cast<__uint128_t>(integers_->at(i)) * static_cast<__uint128_t>(other.integers_->at(j));
                    uint64_t v = static_cast<uint64_t>(tmp % modulus);
                    if (i + j >= degree) v = (modulus - v) % modulus;
                    vec[(i + j) % degree] = (vec[(i + j) % degree] + v) % modulus;
                }
                return GeneralVector(std::move(vec), false);
            } else if (doubles_) {
                vector<double> vec(std::min(degree, size() + other.size() - 1), 0);
                for (size_t i = 0; i < size(); i++) for (size_t j = 0; j < other.size(); j++) {
                    double v = doubles_->at(i) * other.doubles_->at(j);
                    if (i + j >= degree) v = -v;
                    vec[(i + j) % degree] += v;
                }
                return GeneralVector(std::move(vec));
            } else {
                throw std::invalid_argument("[GeneralVector::mul_poly] Cannot mul_poly empty vector");
            }
        }

        template <typename T>
        inline GeneralVector ring2k_mul(const GeneralVector& other, T mask) const {
            ring2k_check<T>();
            vector<T> vec(size());
            for (size_t i = 0; i < size(); i++) {
                T tmp = ring2k_at<T>(i) * other.ring2k_at<T>(i);
                vec[i] = static_cast<T>(tmp & mask);
            }
            if constexpr (std::is_same_v<T, uint64_t>) {
                return GeneralVector(std::move(vec), true);
            } else {
                return GeneralVector(std::move(vec));
            }
        }

        template <typename T>
        inline GeneralVector ring2k_mul_poly(const GeneralVector& other, size_t degree, T mask) const {
            ring2k_check<T>();
            if (size() > degree || other.size() > degree) {
                throw std::invalid_argument("[GeneralVector::ring2k_mul_poly] Degree too small");
            }
            vector<T> vec(std::min(degree, size() + other.size() - 1), 0);
            for (size_t i = 0; i < size(); i++) for (size_t j = 0; j < other.size(); j++) {
                T tmp = ring2k_at<T>(i) * other.ring2k_at<T>(j);
                T v = static_cast<T>(tmp & mask);
                if (i + j >= degree) v = (-v) & mask;
                vec[(i + j) % degree] = (vec[i] + v) & mask;
            }
            if constexpr (std::is_same_v<T, uint64_t>) {
                return GeneralVector(std::move(vec), true);
            } else {
                return GeneralVector(std::move(vec));
            }
        }

        inline GeneralVector square(uint64_t modulus) const {
            return this->mul(*this, modulus);
        }

        template <typename T>
        inline GeneralVector ring2k_square(T mask) const {
            return this->ring2k_mul(*this, mask);
        }

        inline bool near_equal(const GeneralVector& other, double tolerance) const {
            if (complexes_) {
                for (size_t i = 0; i < size(); i++) {
                    if (std::abs(complexes_->at(i) - other.complexes_->at(i)) > tolerance) {
                        return false;
                    }
                }
                return true;
            } else if (integers_) {
                for (size_t i = 0; i < size(); i++) {
                    if (integers_->at(i) != other.integers_->at(i)) {
                        return false;
                    }
                }
                return true;
            } else if (doubles_) {
                for (size_t i = 0; i < size(); i++) {
                    if (std::abs(doubles_->at(i) - other.doubles_->at(i)) > tolerance) {
                        return false;
                    }
                }
                return true;
            } else if (uint32s_) {
                for (size_t i = 0; i < size(); i++) {
                    if (uint32s_->at(i) != other.uint32s_->at(i)) {
                        return false;
                    }
                }
                return true;
            } else if (uint64s_) {
                for (size_t i = 0; i < size(); i++) {
                    if (uint64s_->at(i) != other.uint64s_->at(i)) {
                        return false;
                    }
                }
                return true;
            } else if (uint128s_) {
                for (size_t i = 0; i < size(); i++) {
                    if (uint128s_->at(i) != other.uint128s_->at(i)) {
                        return false;
                    }
                }
                return true;
            } else {
                throw std::invalid_argument("[GeneralVector::near_equal] Cannot compare empty vector");
            }
        }

        inline GeneralVector rotate(int step) const {
            if (complexes_) {
                vector<complex<double>> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = complexes_->at((i + step) % size());
                }
                return GeneralVector(std::move(vec));
            } else if (integers_) {
                vector<uint64_t> vec(size());
                size_t half = size() / 2;
                // rotate by halves
                for (size_t i = 0; i < half; i++) {
                    vec[i] = integers_->at((i + step) % half);
                    vec[i + half] = integers_->at((i + half + step) % half + half);
                }
                return GeneralVector(std::move(vec), false);
            } else {
                throw std::invalid_argument("[GeneralVector::rotate] Cannot rotate double");
            }
        }

        inline GeneralVector conjugate() const {
            if (complexes_) {
                vector<complex<double>> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = std::conj(complexes_->at(i));
                }
                return GeneralVector(std::move(vec));
            } else if (integers_) {
                // swap halves
                vector<uint64_t> vec(size());
                size_t half = size() / 2;
                for (size_t i = 0; i < half; i++) {
                    vec[i] = integers_->at(i + half);
                    vec[i + half] = integers_->at(i);
                }
                return GeneralVector(std::move(vec), false);
            } else {
                throw std::invalid_argument("[GeneralVector::conjugate] Cannot conjugate double");
            }
        }

    };

    inline std::ostream& operator << (std::ostream& os, const GeneralVector& vec) {
        if (vec.is_complexes()) {
            os << sfv(vec.complexes());
        } else if (vec.is_integers()) {
            os << sfv(vec.integers());
        } else if (vec.is_doubles()) {
            os << sfv(vec.doubles());
        } else if (vec.is_uint32s()) {
            os << sfv(vec.uint32s());
        } else if (vec.is_uint64s()) {
            os << sfv(vec.uint64s());
        } else if (vec.is_uint128s()) {
            os << "[";
            for (size_t i = 0; i < vec.size(); i++) {
                uint128_t value = vec.uint128s()[i];
                os << "0x" << std::hex << static_cast<uint64_t>(value >> 64) << std::hex << static_cast<uint64_t>(value);
                if (i != vec.size() - 1) {
                    os << ", ";
                }
            }
            os << "]";
        } else {
            os << "[]";
        }
        return os;
    }

    class GeneralEncoder {
    private:
        optional<BatchEncoder> batch_;
        optional<CKKSEncoder> ckks_;
        optional<PolynomialEncoderRing2k<uint32_t>> poly32_;
        optional<PolynomialEncoderRing2k<uint64_t>> poly64_;
        optional<PolynomialEncoderRing2k<uint128_t>> poly128_;
    public: 
        inline GeneralEncoder(BatchEncoder&& batch): batch_(std::move(batch)) {}
        inline GeneralEncoder(CKKSEncoder&& ckks): ckks_(std::move(ckks)) {}
        inline GeneralEncoder(PolynomialEncoderRing2k<uint32_t>&& poly32): poly32_(std::move(poly32)) {}
        inline GeneralEncoder(PolynomialEncoderRing2k<uint64_t>&& poly64): poly64_(std::move(poly64)) {}
        inline GeneralEncoder(PolynomialEncoderRing2k<uint128_t>&& poly128): poly128_(std::move(poly128)) {}

        inline bool is_batch() const {
            return batch_.has_value();
        }
        inline bool is_ckks() const {
            return ckks_.has_value();
        }
        inline bool is_ring32() const {
            return poly32_.has_value();
        }
        inline bool is_ring64() const {
            return poly64_.has_value();
        }
        inline bool is_ring128() const {
            return poly128_.has_value();
        }

        inline const BatchEncoder& batch() const {
            return *batch_;
        }
        inline const CKKSEncoder& ckks() const {
            return *ckks_;
        }
        inline const PolynomialEncoderRing2k<uint32_t>& poly32() const {
            return *poly32_;
        }
        inline const PolynomialEncoderRing2k<uint64_t>& poly64() const {
            return *poly64_;
        }
        inline const PolynomialEncoderRing2k<uint128_t>& poly128() const {
            return *poly128_;
        }
        template <typename T>
        inline const PolynomialEncoderRing2k<T>& poly() const {
            if constexpr (std::is_same_v<T, uint32_t>) {
                return poly32();
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                return poly64();
            } else if constexpr (std::is_same_v<T, uint128_t>) {
                return poly128();
            } else {
                throw std::invalid_argument("[GeneralEncoder::poly] Unsupported type");
            }
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            if (batch_) {
                batch_->to_device_inplace(pool);
            }
            if (ckks_) {
                ckks_->to_device_inplace(pool);
            }
            if (poly32_) {
                poly32_->to_device_inplace(pool);
            }
            if (poly64_) {
                poly64_->to_device_inplace(pool);
            }
            if (poly128_) {
                poly128_->to_device_inplace(pool);
            }
        }

        inline size_t slot_count() const {
            if (batch_) {
                return batch_->slot_count();
            } else if (ckks_) {
                return ckks_->slot_count();
            } else if (poly32_) {
                return poly32_->slot_count();
            } else if (poly64_) {
                return poly64_->slot_count();
            } else if (poly128_) {
                return poly128_->slot_count();
            } else {
                throw std::invalid_argument("[GeneralEncoder::slot_count] Encoder not initialized");
            }
        }

        inline Plaintext encode_simd(const GeneralVector& vec, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (vec.is_complexes()) {
                return this->ckks().encode_complex64_simd_new(vec.complexes(), parms_id, scale, pool);
            } else if (vec.is_integers()) {
                return this->batch().encode_new(vec.integers(), pool);
            } else {
                throw std::invalid_argument("[GeneralEncoder::encode] Cannot encode SIMD for double");
            }
        }

        inline std::vector<Plaintext> batch_encode_simd(const std::vector<GeneralVector>& vecs, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Plaintext> result;
            for (const auto& vec : vecs) {
                result.push_back(encode_simd(vec, parms_id, scale, pool));
            }
            return result;
        }

        inline Plaintext encode_polynomial(const GeneralVector& vec, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (vec.is_doubles()) {
                return this->ckks().encode_float64_polynomial_new(vec.doubles(), parms_id, scale, pool);
            } else if (vec.is_integers()) {
                return this->batch().encode_polynomial_new(vec.integers(), pool);
            } else if (vec.is_uint32s()) {
                return this->poly32().scale_up_new(vec.uint32s(), parms_id, pool);
            } else if (vec.is_uint64s()) {
                return this->poly64().scale_up_new(vec.uint64s(), parms_id, pool);
            } else if (vec.is_uint128s()) {
                return this->poly128().scale_up_new(vec.uint128s(), parms_id, pool);
            } else {
                throw std::invalid_argument("[GeneralEncoder::encode] Cannot encode polynomial for complex");
            }
        }

        inline std::vector<Plaintext> batch_encode_polynomial(const std::vector<GeneralVector>& vecs, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Plaintext> result;
            for (const auto& vec : vecs) {
                result.push_back(encode_polynomial(vec, parms_id, scale, pool));
            }
            return result;
        }

        inline Plaintext encode_polynomial_centralized(const GeneralVector& vec, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (vec.is_uint32s()) {
                return this->poly32().centralize_new(vec.uint32s(), parms_id, pool);
            } else if (vec.is_uint64s()) {
                return this->poly64().centralize_new(vec.uint64s(), parms_id, pool);
            } else if (vec.is_uint128s()) {
                return this->poly128().centralize_new(vec.uint128s(), parms_id, pool);
            } else {
                throw std::invalid_argument("[GeneralEncoder::encode_polynomial_centralized] Cannot encode polynomial");
            }
        }

        inline std::vector<Plaintext> batch_encode_polynomial_centralized(const std::vector<GeneralVector>& vecs, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<Plaintext> result;
            for (const auto& vec : vecs) {
                result.push_back(encode_polynomial_centralized(vec, parms_id, scale, pool));
            }
            return result;
        }

        inline GeneralVector decode_simd(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (batch_) {
                return GeneralVector(batch_->decode_new(plain, pool), false);
            } else if (ckks_) {
                return GeneralVector(ckks_->decode_complex64_simd_new(plain, pool));
            } else {
                throw std::invalid_argument("[GeneralEncoder::decode] Encoder not initialized");
            }
        }

        inline std::vector<GeneralVector> batch_decode_simd(const std::vector<Plaintext>& plains, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<GeneralVector> result;
            for (const auto& plain : plains) {
                result.push_back(decode_simd(plain, pool));
            }
            return result;
        }

        inline GeneralVector decode_polynomial(const Plaintext& plain, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            if (batch_) {
                return GeneralVector(batch_->decode_polynomial_new(plain), false);
            } else if (ckks_) {
                return GeneralVector(ckks_->decode_float64_polynomial_new(plain, pool));
            } else if (poly32_) {
                return GeneralVector(poly32_->scale_down_new(plain, pool));
            } else if (poly64_) {
                return GeneralVector(poly64_->scale_down_new(plain, pool), true);
            } else if (poly128_) {
                return GeneralVector(poly128_->scale_down_new(plain, pool));
            } else {
                throw std::invalid_argument("[GeneralEncoder::decode] Encoder not initialized");
            }
        }

        inline std::vector<GeneralVector> batch_decode_polynomial(const std::vector<Plaintext>& plains, MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            std::vector<GeneralVector> result;
            for (const auto& plain : plains) {
                result.push_back(decode_polynomial(plain, pool));
            }
            return result;
        }
        
        inline GeneralVector random_simd(size_t size, uint64_t t, double max) const {
            if (batch_) {
                return GeneralVector::random_integers(size, t);
            } else if (ckks_) {
                return GeneralVector::random_complexes(size, max);
            } else {
                throw std::invalid_argument("[GeneralEncoder::random_simd] Encoder not initialized");
            }
        }

        inline GeneralVector random_simd_full(uint64_t t, double max) {
            return this->random_simd(this->slot_count(), t, max);
        }

        inline GeneralVector random_polynomial(size_t size, uint64_t t, double max) const {
            if (batch_) {
                return GeneralVector::random_integers(size, t);
            } else if (ckks_) {
                return GeneralVector::random_doubles(size, max);
            } else if (poly32_) {
                return GeneralVector::random_uint32s(size, poly32_->t_mask());
            } else if (poly64_) {
                return GeneralVector::random_uint64s(size, poly64_->t_mask());
            } else if (poly128_) {
                return GeneralVector::random_uint128s(size, poly128_->t_mask());
            } else {
                throw std::invalid_argument("[GeneralEncoder::random_polynomial] Encoder not initialized");
            }
        }

        inline GeneralVector random_polynomial_full(uint64_t t, double max) {
            if (batch_) {
                return GeneralVector::random_integers(this->slot_count(), t);
            } else if (ckks_) {
                return GeneralVector::random_doubles(this->slot_count() * 2, max);
            } else if (poly32_) {
                return GeneralVector::random_uint32s(this->slot_count(), poly32_->t_mask());
            } else if (poly64_) {
                return GeneralVector::random_uint64s(this->slot_count(), poly64_->t_mask());
            } else if (poly128_) {
                return GeneralVector::random_uint128s(this->slot_count(), poly128_->t_mask());
            } else {
                throw std::invalid_argument("[GeneralEncoder::random_polynomial_full] Encoder not initialized");
            }
        }

        inline GeneralVector random_coefficient_repeated(size_t size, uint64_t t, double max) {
            if (batch_) {
                return GeneralVector::random_integer_repeated(size, t);
            } else if (ckks_) {
                return GeneralVector::random_double_repeated(size, max);
            } else if (poly32_) {
                return GeneralVector::random_uint32_repeated(size, poly32_->t_mask());
            } else if (poly64_) {
                return GeneralVector::random_uint64_repeated(size, poly64_->t_mask());
            } else if (poly128_) {
                return GeneralVector::random_uint128_repeated(size, poly128_->t_mask());
            } else {
                throw std::invalid_argument("[GeneralEncoder::random_coefficient_repeated] Encoder not initialized");
            }
        }

        inline GeneralVector random_coefficient_repeated_full(uint64_t t, double max) {
            return this->random_coefficient_repeated(ckks_ ? this->slot_count() * 2 : this->slot_count(), t, max);
        }

        inline GeneralVector random_slot_repeated(size_t size, uint64_t t, double max) {
            if (batch_) {
                return GeneralVector::random_integer_repeated(size, t);
            } else if (ckks_) {
                return GeneralVector::random_complex_repeated(size, max);
            } else {
                throw std::invalid_argument("[GeneralEncoder::random_slot_repeated] Encoder not initialized");
            }
        }

        inline GeneralVector random_slot_repeated_full(uint64_t t, double max) {
            return this->random_slot_repeated(this->slot_count(), t, max);
        }

        inline GeneralVector zeros_polynomial() {
            if (batch_) {
                return GeneralVector::zeros_integer(this->slot_count());
            } else if (ckks_) {
                return GeneralVector::zeros_double(this->slot_count() * 2);
            } else if (poly32_) {
                return GeneralVector::zeros_uint32(this->slot_count());
            } else if (poly64_) {
                return GeneralVector::zeros_uint64(this->slot_count());
            } else if (poly128_) {
                return GeneralVector::zeros_uint128(this->slot_count());
            } else {
                throw std::invalid_argument("[GeneralEncoder::zeros_polynomial] Encoder not initialized");
            }
        }

        inline GeneralVector zeros_simd() {
            if (batch_) {
                return GeneralVector::zeros_integer(this->slot_count());
            } else if (ckks_) {
                return GeneralVector::zeros_complex(this->slot_count());
            } else {
                throw std::invalid_argument("[GeneralEncoder::zeros_simd] Encoder not initialized");
            }
        }

        inline size_t coeff_count() {
            if (batch_) {
                return batch_->slot_count();
            } else if (ckks_) {
                return ckks_->slot_count() * 2;
            } else if (poly32_) {
                return poly32_->slot_count();
            } else if (poly64_) {
                return poly64_->slot_count();
            } else if (poly128_) {
                return poly128_->slot_count();
            } else {
                throw std::invalid_argument("[GeneralEncoder::coeff_count] Encoder not initialized");
            }
        }

    };

    struct GeneralHeContextParameters {
        bool device; 
        SchemeType scheme; 
        size_t n; 
        size_t simd_log_t; 
        vector<size_t> log_qi;
        bool expand_mod_chain;
        uint64_t seed;
        double input_max;
        double scale = 0;
        double tolerance = 1e-4;
        bool to_device_after_keygeneration;
        bool use_special_prime_for_encryption;
        MemoryPoolHandle pool;
        size_t ring2k_log_t;

        inline GeneralHeContextParameters() {}

        inline GeneralHeContextParameters(
            bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, 
            bool expand_mod_chain, uint64_t seed, double input_max = 0, double scale = 0, double tolerance = 1e-4,
            bool to_device_after_keygeneration = false, bool use_special_prime_for_encryption = false,
            MemoryPoolHandle pool = MemoryPool::GlobalPool(), size_t ring_log_t = 0
        ): 
            device(device), scheme(scheme), n(n), simd_log_t(log_t), log_qi(log_qi), expand_mod_chain(expand_mod_chain), seed(seed),
            input_max(input_max), scale(scale), tolerance(tolerance), to_device_after_keygeneration(to_device_after_keygeneration),
            use_special_prime_for_encryption(use_special_prime_for_encryption), pool(pool), ring2k_log_t(ring_log_t) {}
    };

    class GeneralHeContext {
    private:
        SchemeType scheme_;
        HeContextPointer he_context_;
        Encryptor* encryptor_;
        Evaluator* evaluator_;
        Decryptor* decryptor_;
        KeyGenerator* key_generator_;
        GeneralEncoder* encoder_;
        EncryptionParameters params_host_;
        uint64_t t_;
        double input_max_;
        double scale_;
        double tolerance_;
        MemoryPoolHandle pool_;
        uint128_t ring_mask_;

    public:

        GeneralHeContext(GeneralHeContextParameters args);

        inline GeneralHeContext(bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, 
            bool expand_mod_chain, uint64_t seed, double input_max = 0, double scale = 0, double tolerance = 1e-4,
            bool to_device_after_keygeneration = false, bool use_special_prime_for_encryption = false,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ): GeneralHeContext(GeneralHeContextParameters(device, scheme, n, log_t, log_qi, expand_mod_chain, seed, input_max, scale, tolerance, to_device_after_keygeneration, use_special_prime_for_encryption, pool)) {}

        ~GeneralHeContext();
        GeneralHeContext(const GeneralHeContext&) = delete;
        GeneralHeContext& operator=(const GeneralHeContext&) = delete;
        GeneralHeContext(GeneralHeContext&&);
        GeneralHeContext& operator=(GeneralHeContext&&) = delete;

        inline MemoryPoolHandle pool() const {
            return pool_;
        }

        inline SchemeType scheme() const {
            return scheme_;
        }
        inline HeContextPointer context() const {
            return he_context_;
        }
        inline const EncryptionParameters& params_host() const {
            return params_host_;
        }
        inline const Evaluator& evaluator() const {
            return *evaluator_;
        }
        inline const Encryptor& encryptor() const {
            return *encryptor_;
        }
        inline const Decryptor& decryptor() const {
            return *decryptor_;
        }
        inline const KeyGenerator& key_generator() const {
            return *key_generator_;
        }
        inline const GeneralEncoder& encoder() const {
            return *encoder_;
        }
        inline uint64_t t() const {
            return t_;
        }
        inline uint64_t simd_t() const {
            return t_;
        }
        inline uint128_t ring_t_mask() const {
            return ring_mask_;
        }
        inline double input_max() const {
            return input_max_;
        }
        inline double scale() const {
            return scale_;
        }
        inline double tolerance() const {
            return tolerance_;
        }

        // create vectors

        inline GeneralVector random_simd(size_t size) const {
            return encoder_->random_simd(size, t_, input_max_);
        }
        inline GeneralVector random_simd_full() const {
            return encoder_->random_simd_full(t_, input_max_);
        }
        inline GeneralVector random_polynomial(size_t size) const {
            return encoder_->random_polynomial(size, t_, input_max_);
        }
        inline GeneralVector random_polynomial_full() const {
            return encoder_->random_polynomial_full(t_, input_max_);
        }
        inline GeneralVector random_coefficient_repeated(size_t size) const {
            return encoder_->random_coefficient_repeated(size, t_, input_max_);
        }
        inline GeneralVector random_coefficient_repeated_full() const {
            return encoder_->random_coefficient_repeated_full(t_, input_max_);
        }
        inline GeneralVector random_slot_repeated(size_t size) const {
            return encoder_->random_slot_repeated(size, t_, input_max_);
        }
        inline GeneralVector random_slot_repeated_full() const {
            return encoder_->random_slot_repeated_full(t_, input_max_);
        }
        inline GeneralVector zeros_polynomial() const {
            return encoder_->zeros_polynomial();
        }
        inline GeneralVector zeros_simd() const {
            return encoder_->zeros_simd();
        }

        // batch create vectors
        inline std::vector<GeneralVector> batch_random_simd(size_t batch, size_t size) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_simd(size));
            return ret;
        }
        inline std::vector<GeneralVector> batch_random_simd_full(size_t batch) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_simd_full());
            return ret;
        }
        inline std::vector<GeneralVector> batch_random_polynomial(size_t batch, size_t size) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_polynomial(size));
            return ret;
        }
        inline std::vector<GeneralVector> batch_random_polynomial_full(size_t batch) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_polynomial_full());
            return ret;
        }
        inline std::vector<GeneralVector> batch_random_coefficient_repeated(size_t batch, size_t size) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_coefficient_repeated(size));
            return ret;
        }
        inline std::vector<GeneralVector> batch_random_coefficient_repeated_full(size_t batch) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_coefficient_repeated_full());
            return ret;
        }
        inline std::vector<GeneralVector> batch_random_slot_repeated(size_t batch, size_t size) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_slot_repeated(size));
            return ret;
        }
        inline std::vector<GeneralVector> batch_random_slot_repeated_full(size_t batch) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(random_slot_repeated_full());
            return ret;
        }
        inline std::vector<GeneralVector> batch_zeros_polynomial(size_t batch) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(zeros_polynomial());
            return ret;
        }
        inline std::vector<GeneralVector> batch_zeros_simd(size_t batch) const {
            std::vector<GeneralVector> ret; ret.reserve(batch);
            for (size_t i = 0; i < batch; i++) ret.push_back(zeros_simd());
            return ret;
        }

        inline std::vector<Ciphertext> batch_encrypt_symmetric(const std::vector<Plaintext>& vecs, bool save_seed) const {
            std::vector<Ciphertext> ret; ret.reserve(vecs.size());
            for (const auto& vec: vecs) ret.push_back(encryptor_->encrypt_symmetric_new(vec, save_seed));
            return ret;
        }

        inline std::vector<Ciphertext> batch_encrypt_asymmetric(const std::vector<Plaintext>& vecs) const {
            std::vector<Ciphertext> ret; ret.reserve(vecs.size());
            for (const auto& vec: vecs) ret.push_back(encryptor_->encrypt_asymmetric_new(vec));
            return ret;
        }

        inline std::vector<Plaintext> batch_decrypt(const std::vector<Ciphertext>& vecs) const {
            std::vector<Plaintext> ret; ret.reserve(vecs.size());
            for (const auto& vec: vecs) ret.push_back(decryptor_->decrypt_new(vec));
            return ret;
        }

        // arithmetic

        inline GeneralVector negate(const GeneralVector& vec) const {
            if (ring_mask_ == 0) return vec.negate(t_);
            else {
                if (vec.is_uint32s()) {
                    return vec.ring2k_negate<uint32_t>(ring_mask_);
                } else if (vec.is_uint64s()) {
                    return vec.ring2k_negate<uint64_t>(ring_mask_);
                } else if (vec.is_uint128s()) {
                    return vec.ring2k_negate<uint128_t>(ring_mask_);
                } else {
                    throw std::invalid_argument("[GeneralHeContext::negate] Unsupported type");
                }
            }
        }
        inline std::vector<GeneralVector> batch_negate(const std::vector<GeneralVector>& vecs) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs.size());
            for (const auto& vec: vecs) ret.push_back(negate(vec));
            return ret;
        }
        inline GeneralVector add(const GeneralVector& vec1, const GeneralVector& vec2) const {
            if (ring_mask_ == 0) return vec1.add(vec2, t_);
            else {
                if (vec1.is_uint32s()) {
                    return vec1.ring2k_add<uint32_t>(vec2, ring_mask_);
                } else if (vec1.is_uint64s()) {
                    return vec1.ring2k_add<uint64_t>(vec2, ring_mask_);
                } else if (vec1.is_uint128s()) {
                    return vec1.ring2k_add<uint128_t>(vec2, ring_mask_);
                } else {
                    throw std::invalid_argument("[GeneralHeContext::add] Unsupported type");
                }
            }
        }
        inline std::vector<GeneralVector> batch_add(const std::vector<GeneralVector>& vecs1, const std::vector<GeneralVector>& vecs2) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs1.size());
            for (size_t i = 0; i < vecs1.size(); i++) ret.push_back(add(vecs1[i], vecs2[i]));
            return ret;
        }
        inline GeneralVector sub(const GeneralVector& vec1, const GeneralVector& vec2) const {
            if (ring_mask_ == 0) return vec1.sub(vec2, t_);
            else {
                if (vec1.is_uint32s()) {
                    return vec1.ring2k_sub<uint32_t>(vec2, ring_mask_);
                } else if (vec1.is_uint64s()) {
                    return vec1.ring2k_sub<uint64_t>(vec2, ring_mask_);
                } else if (vec1.is_uint128s()) {
                    return vec1.ring2k_sub<uint128_t>(vec2, ring_mask_);
                } else {
                    throw std::invalid_argument("[GeneralHeContext::sub] Unsupported type");
                }
            }
        }
        inline std::vector<GeneralVector> batch_sub(const std::vector<GeneralVector>& vecs1, const std::vector<GeneralVector>& vecs2) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs1.size());
            for (size_t i = 0; i < vecs1.size(); i++) ret.push_back(sub(vecs1[i], vecs2[i]));
            return ret;
        }
        inline GeneralVector mul(const GeneralVector& vec1, const GeneralVector& vec2) const {
            if (ring_mask_ == 0) return vec1.mul(vec2, t_);
            else {
                if (vec1.is_uint32s()) {
                    return vec1.ring2k_mul<uint32_t>(vec2, ring_mask_);
                } else if (vec1.is_uint64s()) {
                    return vec1.ring2k_mul<uint64_t>(vec2, ring_mask_);
                } else if (vec1.is_uint128s()) {
                    return vec1.ring2k_mul<uint128_t>(vec2, ring_mask_);
                } else {
                    throw std::invalid_argument("[GeneralHeContext::mul] Unsupported type");
                }
            }
        }
        inline std::vector<GeneralVector> batch_mul(const std::vector<GeneralVector>& vecs1, const std::vector<GeneralVector>& vecs2) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs1.size());
            for (size_t i = 0; i < vecs1.size(); i++) ret.push_back(mul(vecs1[i], vecs2[i]));
            return ret;
        }
        inline GeneralVector mul_poly(const GeneralVector& vec1, const GeneralVector& vec2) const {
            size_t degree = params_host_.poly_modulus_degree();
            if (ring_mask_ == 0) return vec1.mul_poly(vec2, degree, t_);
            else {
                if (vec1.is_uint32s()) {
                    return vec1.ring2k_mul_poly<uint32_t>(vec2, degree, ring_mask_);
                } else if (vec1.is_uint64s()) {
                    return vec1.ring2k_mul_poly<uint64_t>(vec2, degree, ring_mask_);
                } else if (vec1.is_uint128s()) {
                    return vec1.ring2k_mul_poly<uint128_t>(vec2, degree, ring_mask_);
                } else {
                    throw std::invalid_argument("[GeneralHeContext::mul_poly] Unsupported type");
                }
            }
        }
        inline std::vector<GeneralVector> batch_mul_poly(const std::vector<GeneralVector>& vecs1, const std::vector<GeneralVector>& vecs2) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs1.size());
            for (size_t i = 0; i < vecs1.size(); i++) ret.push_back(mul_poly(vecs1[i], vecs2[i]));
            return ret;
        }
        inline GeneralVector square(const GeneralVector& vec) const {
            if (ring_mask_ == 0) return vec.square(t_);
            else {
                if (vec.is_uint32s()) {
                    return vec.ring2k_square<uint32_t>(ring_mask_);
                } else if (vec.is_uint64s()) {
                    return vec.ring2k_square<uint64_t>(ring_mask_);
                } else if (vec.is_uint128s()) {
                    return vec.ring2k_square<uint128_t>(ring_mask_);
                } else {
                    throw std::invalid_argument("[GeneralHeContext::square] Unsupported type");
                }
            }
        }
        inline std::vector<GeneralVector> batch_square(const std::vector<GeneralVector>& vecs) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs.size());
            for (const auto& vec: vecs) ret.push_back(square(vec));
            return ret;
        }
        inline std::vector<GeneralVector> batch_rotate(const std::vector<GeneralVector>& vecs, int step) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs.size());
            for (const auto& vec: vecs) ret.push_back(vec.rotate(step));
            return ret;
        }
        inline std::vector<GeneralVector> batch_conjugate(const std::vector<GeneralVector>& vecs) const {
            std::vector<GeneralVector> ret; ret.reserve(vecs.size());
            for (const auto& vec: vecs) ret.push_back(vec.conjugate());
            return ret;
        }
        inline bool near_equal(const GeneralVector& vec1, const GeneralVector& vec2) const {
            return vec1.near_equal(vec2, tolerance_);
        }
        inline std::vector<bool> batch_near_equal_every(const std::vector<GeneralVector>& vecs1, const std::vector<GeneralVector>& vecs2) const {
            std::vector<bool> ret; ret.reserve(vecs1.size());
            for (size_t i = 0; i < vecs1.size(); i++) ret.push_back(near_equal(vecs1[i], vecs2[i]));
            return ret;
        }
        inline bool batch_near_equal(const std::vector<GeneralVector>& vecs1, const std::vector<GeneralVector>& vecs2) const {
            for (size_t i = 0; i < vecs1.size(); i++) if (!near_equal(vecs1[i], vecs2[i])) return false;
            return true;
        }

        inline size_t coeff_count() const {
            return encoder_->coeff_count();
        }
        
    };

    

}