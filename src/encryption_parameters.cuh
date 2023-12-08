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
        Modulus plain_modulus_host_;

        void compute_parms_id();

    public:

        inline EncryptionParameters(SchemeType scheme_type) : 
            scheme_(scheme_type), device(false), plain_modulus_(new Modulus(0), false),
            plain_modulus_host_(0)
        {}

        inline EncryptionParameters(const EncryptionParameters& parms):
            scheme_(parms.scheme_),
            parms_id_(parms.parms_id_),
            poly_modulus_degree_(parms.poly_modulus_degree_),
            coeff_modulus_(parms.coeff_modulus_.clone()),
            plain_modulus_(parms.plain_modulus_.clone()),
            plain_modulus_host_(parms.plain_modulus_host_),
            device(parms.device) {}
            
        EncryptionParameters(EncryptionParameters&& parms) = default;

        // copy assignment
        inline EncryptionParameters& operator=(const EncryptionParameters& parms) {
            if (this == &parms) {
                return *this;
            }
            scheme_ = parms.scheme_;
            parms_id_ = parms.parms_id_;
            poly_modulus_degree_ = parms.poly_modulus_degree_;
            coeff_modulus_ = parms.coeff_modulus_.clone();
            plain_modulus_ = parms.plain_modulus_.clone();
            plain_modulus_host_ = parms.plain_modulus_host_;
            device = parms.device;
            return *this;
        }
        // move assignment
        inline EncryptionParameters& operator=(EncryptionParameters&& parms) = default;

        
        inline SchemeType scheme() const noexcept {
            return scheme_;
        }

        inline const ParmsID& parms_id() const noexcept {
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

        inline const Modulus& plain_modulus_host() const noexcept {
            return plain_modulus_host_;
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
            coeff_modulus_ = utils::Array<Modulus>::create_and_copy_from_slice(coeff_modulus);
            compute_parms_id();
        }

        inline void set_coeff_modulus(const utils::Array<Modulus>& coeff_modulus) {
            set_coeff_modulus(coeff_modulus.const_reference());
        }

        inline void set_coeff_modulus(const std::vector<Modulus>& coeff_modulus) {
            utils::Array<Modulus> array(coeff_modulus.size(), false);
            for (size_t i = 0; i < coeff_modulus.size(); i++) {
                array[i] = coeff_modulus[i];
            }
            set_coeff_modulus(array.const_reference());
        }

        inline void set_coeff_modulus(const std::vector<uint64_t>& coeff_modulus) {
            utils::Array<Modulus> array(coeff_modulus.size(), false);
            for (size_t i = 0; i < coeff_modulus.size(); i++) {
                array[i] = Modulus(coeff_modulus[i]);
            }
            set_coeff_modulus(array.const_reference());
        }

        inline void set_plain_modulus(const Modulus& plain_modulus) {
            if (this->on_device()) {
                throw std::invalid_argument("[EncryptionParameters::set_plain_modulus] Can only set plain_modulus on host");
            }
            utils::Box<Modulus> new_t = utils::Box<Modulus>(new Modulus(plain_modulus), false);
            plain_modulus_ = std::move(new_t);
            plain_modulus_host_ = plain_modulus;
            compute_parms_id();
        }

        inline void set_plain_modulus(uint64_t plain_modulus) {
            set_plain_modulus(Modulus(plain_modulus));
        }

        inline EncryptionParameters clone() const {
            return EncryptionParameters(*this);
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

    enum class SecurityLevel {
        None = 0,
        Classical128 = 128,
        Classical192 = 192,
        Classical256 = 256,
    };

    enum class EncryptionParameterErrorType {
        None = -1,
        Success = 0,
        CreatedFromDeviceParms,
        InvalidScheme,
        InvalidCoeffModulusSize,
        InvalidCoeffModulusBitCount,
        InvalidCoeffModulusNoNTT,
        InvalidPolyModulusDegree,
        InvalidPolyModulusDegreeNonPowerOfTwo,
        InvalidParametersTooLarge,
        InvalidParametersInsecure,
        FailedCreatingRNSBase,
        InvalidPlainModulusBitCount,
        InvalidPlainModulusCoprimality,
        InvalidPlainModulusTooLarge,
        InvalidPlainModulusNonZero,
        FailedCreatingRNSTool,
        FailedCreatingGaloisTool,
    };

    struct EncryptionParameterQualifiers {
        EncryptionParameterErrorType parameter_error = EncryptionParameterErrorType::None;
        bool using_fft = false;
        bool using_ntt = false;
        bool using_batching = false;
        bool using_fast_plain_lift = false;
        bool using_descending_modulus_chain = false;
        SecurityLevel security_level = SecurityLevel::None;

        bool parameters_set() const noexcept {
            return parameter_error == EncryptionParameterErrorType::Success;
        }
    };

}

namespace std {

    struct TroyHashParmsID {
        std::size_t operator()(const troy::ParmsID &parms_id) const {
            std::uint64_t result = 17;
            result = 31 * result + parms_id[0];
            result = 31 * result + parms_id[1];
            result = 31 * result + parms_id[2];
            result = 31 * result + parms_id[3];
            return static_cast<std::size_t>(result);
        }
    };

    template <>
    struct hash<troy::EncryptionParameters>
    {
        std::size_t operator()(const troy::EncryptionParameters &parms) const
        {
            TroyHashParmsID parms_id_hash;
            return parms_id_hash(parms.parms_id());
        }
    };

}