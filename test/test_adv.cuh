#include "../src/he_context.cuh"
#include "../src/batch_encoder.cuh"
#include "../src/ckks_encoder.cuh"
#include "../src/evaluator.cuh"
#include "../src/encryptor.cuh"
#include "../src/key_generator.cuh"
#include "../src/decryptor.cuh"

namespace tool {

    using namespace troy;
    using troy::utils::ConstSlice;
    using troy::utils::Slice;
    using std::vector;
    using std::optional;
    using std::complex;

    template<typename T>
    ConstSlice<T> sfv(const vector<T> &vec) {
        return ConstSlice<T>(vec.data(), vec.size(), false, nullptr);
    }

    class GeneralVector {
    private:
        optional<vector<complex<double>>> complexes_;
        optional<vector<uint64_t>> integers_;
        optional<vector<double>> doubles_;
    public:

        inline GeneralVector(vector<complex<double>>&& complexes): complexes_(std::move(complexes)) {}
        inline GeneralVector(vector<uint64_t>&& integers): integers_(std::move(integers)) {}
        inline GeneralVector(vector<double>&& doubles): doubles_(std::move(doubles)) {}

        inline static GeneralVector random_complexes(size_t size, double component_max_absolute) {
            vector<complex<double>> vec(size);
            for (size_t i = 0; i < size; i++) {
                double real = (double)rand() / (double)RAND_MAX * 2 * component_max_absolute - component_max_absolute;
                double imag = (double)rand() / (double)RAND_MAX * 2 * component_max_absolute - component_max_absolute;
                vec[i] = complex<double>(real, imag);
            }
            return GeneralVector(std::move(vec));
        }

        inline static GeneralVector random_integers(size_t size, uint64_t modulus) {
            vector<uint64_t> vec(size);
            for (size_t i = 0; i < size; i++) {
                vec[i] = rand() % modulus;
            }
            return GeneralVector(std::move(vec));
        }

        inline static GeneralVector random_doubles(size_t size, double max_absolute) {
            vector<double> vec(size);
            for (size_t i = 0; i < size; i++) {
                vec[i] = (double)rand() / (double)RAND_MAX * 2 * max_absolute - max_absolute;
            }
            return GeneralVector(std::move(vec));
        }

        inline GeneralVector subvector(size_t low, size_t high) {
            if (complexes_) {
                return GeneralVector(vector<complex<double>>(complexes_->begin() + low, complexes_->begin() + high));
            } else if (integers_) {
                return GeneralVector(vector<uint64_t>(integers_->begin() + low, integers_->begin() + high));
            } else {
                return GeneralVector(vector<double>(doubles_->begin() + low, doubles_->begin() + high));
            }
        }

        inline GeneralVector element(size_t index) {
            return subvector(index, index + 1);
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
        inline size_t size() const {
            return complexes_ ? complexes_->size() : integers_ ? integers_->size() : doubles_->size();
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
        inline const vector<complex<double>>& complexes() const {
            return *complexes_;
        }
        inline const vector<uint64_t>& integers() const {
            return *integers_;
        }
        inline const vector<double>& doubles() const {
            return *doubles_;
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
                return GeneralVector(std::move(vec));
            } else {
                vector<double> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = -doubles_->at(i);
                }
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
                return GeneralVector(std::move(vec));
            } else {
                vector<double> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = doubles_->at(i) + other.doubles_->at(i);
                }
                return GeneralVector(std::move(vec));
            }
        }

        inline GeneralVector sub(const GeneralVector& other, uint64_t modulus) const {
            return this->add(other.negate(modulus), modulus);
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
                return GeneralVector(std::move(vec));
            } else {
                vector<double> vec(size());
                for (size_t i = 0; i < size(); i++) {
                    vec[i] = doubles_->at(i) * other.doubles_->at(i);
                }
                return GeneralVector(std::move(vec));
            }
        }

        inline GeneralVector square(uint64_t modulus) const {
            return this->mul(*this, modulus);
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
            } else {
                for (size_t i = 0; i < size(); i++) {
                    if (std::abs(doubles_->at(i) - other.doubles_->at(i)) > tolerance) {
                        return false;
                    }
                }
                return true;
            }
        }

        inline GeneralVector rotate(int step) {
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
                return GeneralVector(std::move(vec));
            } else {
                throw std::invalid_argument("[GeneralVector::rotate] Cannot rotate double");
            }
        }

        inline GeneralVector conjugate() {
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
                return GeneralVector(std::move(vec));
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
        } else {
            os << sfv(vec.doubles());
        }
        return os;
    }

    class GeneralEncoder {
    private:
        optional<BatchEncoder> batch_;
        optional<CKKSEncoder> ckks_;
    public: 
        inline GeneralEncoder(BatchEncoder&& batch): batch_(std::move(batch)) {}
        inline GeneralEncoder(CKKSEncoder&& ckks): ckks_(std::move(ckks)) {}

        inline const BatchEncoder& batch() const {
            return *batch_;
        }
        inline const CKKSEncoder& ckks() const {
            return *ckks_;
        }

        inline void to_device_inplace() {
            if (batch_) {
                batch_->to_device_inplace();
            }
            if (ckks_) {
                ckks_->to_device_inplace();
            }
        }

        inline size_t slot_count() const {
            return batch_ ? batch_->slot_count() : ckks_->slot_count();
        }

        inline Plaintext encode_simd(const GeneralVector& vec, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20) const {
            if (vec.is_complexes()) {
                return this->ckks().encode_complex64_simd_new(vec.complexes(), parms_id, scale);
            } else if (vec.is_integers()) {
                return this->batch().encode_new(vec.integers());
            } else {
                throw std::invalid_argument("[GeneralEncoder::encode] Cannot encode SIMD for double");
            }
        }

        inline Plaintext encode_polynomial(const GeneralVector& vec, std::optional<ParmsID> parms_id = std::nullopt, double scale = 1<<20) const {
            if (vec.is_doubles()) {
                return this->ckks().encode_float64_polynomial_new(vec.doubles(), parms_id, scale);
            } else if (vec.is_integers()) {
                return this->batch().encode_polynomial_new(vec.integers());
            } else {
                throw std::invalid_argument("[GeneralEncoder::encode] Cannot encode polynomial for complexes");
            }
        }

        inline GeneralVector decode_simd(const Plaintext& plain) const {
            if (batch_) {
                return GeneralVector(batch_->decode_new(plain));
            } else if (ckks_) {
                return GeneralVector(ckks_->decode_complex64_simd_new(plain));
            } else {
                throw std::invalid_argument("[GeneralEncoder::decode] Encoder not initialized");
            }
        }

        inline GeneralVector decode_polynomial(const Plaintext& plain) const {
            if (batch_) {
                return GeneralVector(batch_->decode_polynomial_new(plain));
            } else if (ckks_) {
                return GeneralVector(ckks_->decode_float64_polynomial_new(plain));
            } else {
                throw std::invalid_argument("[GeneralEncoder::decode] Encoder not initialized");
            }
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
            } else {
                throw std::invalid_argument("[GeneralEncoder::random_polynomial] Encoder not initialized");
            }
        }

        inline GeneralVector random_polynomial_full(uint64_t t, double max) {
            if (batch_) {
                return GeneralVector::random_integers(this->slot_count(), t);
            } else if (ckks_) {
                return GeneralVector::random_doubles(this->slot_count() * 2, max);
            } else {
                throw std::invalid_argument("[GeneralEncoder::random_polynomial_full] Encoder not initialized");
            }
        }

    };

    class GeneralHeContext {
    private:
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

    public:
        GeneralHeContext(bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, 
            bool expand_mod_chain, uint64_t seed, double input_max = 0, double scale = 0, double tolerance = 1e-4,
            bool to_device_after_keygeneration = false, bool use_special_prime_for_encryption = false
        );
        ~GeneralHeContext();

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
        inline double input_max() const {
            return input_max_;
        }
        inline double scale() const {
            return scale_;
        }
        inline double tolerance() const {
            return tolerance_;
        }
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
        inline GeneralVector negate(const GeneralVector& vec) const {
            return vec.negate(t_);
        }
        inline GeneralVector add(const GeneralVector& vec1, const GeneralVector& vec2) const {
            return vec1.add(vec2, t_);
        }
        inline GeneralVector sub(const GeneralVector& vec1, const GeneralVector& vec2) const {
            return vec1.sub(vec2, t_);
        }
        inline GeneralVector mul(const GeneralVector& vec1, const GeneralVector& vec2) const {
            return vec1.mul(vec2, t_);
        }
        inline bool near_equal(const GeneralVector& vec1, const GeneralVector& vec2) const {
            return vec1.near_equal(vec2, tolerance_);
        }
        
    };

    

}