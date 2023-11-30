#pragma once
#include "utils/hash.h"
#include "modulus.cuh"

namespace troy {

    enum SchemeType {
        None = 0,
        BFV,
        CKKS,
        BGV
    };

    using ParmsID = utils::HashFunction::HashBlock;
    extern const ParmsID parms_id_zero;

    class EncryptionParameters {
    private:
        bool device;
        SchemeType scheme_;
        ParmsID parms_id_;
        size_t poly_modulus_degree_;
        utils::Array<Modulus> coeff_modulus_;
        utils::Box<Modulus> plain_modulus_;

        void compute_parms_id();

    public:

        EncryptionParameters(SchemeType scheme_type) : scheme_(scheme_type), device(false) {}
        
        inline SchemeType scheme() const noexcept {
            return scheme_;
        }

        inline ParmsID parms_id() const noexcept {
            return parms_id_;
        }

        inline size_t poly_modulus_degree() const noexcept {
            return poly_modulus_degree_;
        }

        inline utils::ConstSlice<Modulus> coeff_modulus() const noexcept {
            return coeff_modulus_.const_reference();
        }

        inline utils::ConstPointer<Modulus> plain_modulus() const noexcept {
            return plain_modulus_.as_const_pointer();
        }

        inline bool on_device() const noexcept {
            return device;
        }

        inline void set_poly_modulus_degree(size_t poly_modulus_degree) {
            if (this->on_device()) {
                throw std::invalid_argument("[EncryptionParameters::set_poly_modulus_degree] Can only set poly_modulus_degree on host");
            }
            poly_modulus_degree_ = poly_modulus_degree;
            compute_parms_id();
        }

        inline void set_coeff_modulus(utils::ConstSlice<Modulus> coeff_modulus) {
            if (this->on_device() || coeff_modulus.on_device()) {
                throw std::invalid_argument("[EncryptionParameters::set_coeff_modulus] Can only set coeff_modulus on host");
            }
            coeff_modulus_.copy_from_slice(coeff_modulus);
            compute_parms_id();
        }

        inline void set_plain_modulus(Modulus plain_modulus) {
            if (this->on_device()) {
                throw std::invalid_argument("[EncryptionParameters::set_plain_modulus] Can only set plain_modulus on host");
            }
            utils::Box<Modulus> new_t = utils::Box<Modulus>(Modulus(plain_modulus));
            plain_modulus_ = std::move(new_t);
            compute_parms_id();
        }

        inline EncryptionParameters clone() const {
            EncryptionParameters new_t = EncryptionParameters(this->scheme());
            new_t.poly_modulus_degree_ = this->poly_modulus_degree_;
            new_t.coeff_modulus_ = this->coeff_modulus_.clone();
            new_t.plain_modulus_ = this->plain_modulus_.clone();
            new_t.parms_id_ = this->parms_id_;
            new_t.device = this->device;
            return new_t;
        }

        inline void to_device_inplace() {
            if (this->on_device()) {
                return;
            }
            this->coeff_modulus_.to_device_inplace();
            this->plain_modulus_.to_device_inplace();
            this->device = true;
        }

        inline EncryptionParameters to_device() const {
            EncryptionParameters cloned = this->clone();
            cloned.to_device_inplace();
            return cloned;
        }

    };

    enum SecurityLevel {
        None = 0,
        Classical128 = 128,
        Classical192 = 192,
        Classical256 = 256,
    };

    enum ErrorType {
        None = -1,
        Success = 0,
        InvalidScheme,
        InvalidCoeffModulusSize,
        InvalidCoeffModulusBitCount,
        InvalidCoeffModulusNoNTT,
        InvalidPolyModulusDegree,
        InvalidPlyModulusDegreeNonPowerOfTwo,
        InvalidParametersTooLarge,
        InvalidParametersInsecure,
        FailedCreatingRNSBase,
        InvalidPlainModulusBitCount,
        InvalidPlainModulusCoprimality,
        InvalidPlainModulusTooLarge,
        InvalidPlainModulusNonZero,
        FailedCreatingRNSTool,
    };

    struct EncryptionParameterQualifiers {
        ErrorType parameter_error;
        bool using_fft;
        bool using_ntt;
        bool using_batching;
        bool using_fast_plain_lift;
        bool using_descending_modulus_chain;
        SecurityLevel security_level;

        bool parameters_set() const noexcept {
            return parameter_error == ErrorType::Success;
        }
    };


}