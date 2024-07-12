#include "context_data.h"
#include "he_context.h"

namespace troy {

    using namespace utils;

    void ContextData::set_context(std::shared_ptr<HeContext> context) {
        this->context_ = std::optional(std::weak_ptr<HeContext>(context));
    }

    std::optional<std::shared_ptr<HeContext>> ContextData::context() const {
        if (this->context_.has_value()) {
            // try get a shared pointer from the weak pointer
            std::shared_ptr<HeContext> context = this->context_.value().lock();
            if (context) {
                return std::optional(context);
            } else {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    }

    std::shared_ptr<HeContext> ContextData::context_pointer() const {
        if (this->context_.has_value()) {
            return this->context_.value().lock();
        } else {
            return nullptr;
        }
    }

    void ContextData::to_device_inplace(MemoryPoolHandle pool) {

        if (this->on_device()) {
            return;
        }

        this->parms_.to_device_inplace(pool);

        if (this->rns_tool_.has_value()) {
            this->rns_tool_->to_device_inplace(pool);
        }

        for (size_t i = 0; i < small_ntt_tables_.size(); i++) {
            this->small_ntt_tables_[i].to_device_inplace(pool);
        }
        this->small_ntt_tables_.to_device_inplace(pool);

        if (this->plain_ntt_tables_.has_value()) {
            this->plain_ntt_tables_.value()->to_device_inplace(pool);
            this->plain_ntt_tables_.value().to_device_inplace(pool);
        }

        if (this->galois_tool_.has_value()) {
            this->galois_tool_.value().to_device_inplace(pool);
        }

        this->total_coeff_modulus_.to_device_inplace(pool);
        this->coeff_div_plain_modulus_.to_device_inplace(pool);

        this->plain_upper_half_increment_.to_device_inplace(pool);
        this->upper_half_threshold_.to_device_inplace(pool);
        this->upper_half_increment_.to_device_inplace(pool);

        this->device = true;
        
    }
    
    void ContextData::validate(SecurityLevel sec_level) {
        // std::cout << "validating context data " << this->parms_id() << std::endl;
        EncryptionParameterQualifiers& qualifiers = this->qualifiers_;
        EncryptionParameters& parms = this->parms_;
        qualifiers.parameter_error = EncryptionParameterErrorType::Success;

        if (parms.on_device()) {
            qualifiers.parameter_error = EncryptionParameterErrorType::CreatedFromDeviceParms;
            return;
        }
        
        if (parms.scheme() == SchemeType::Nil) {
            qualifiers.parameter_error = EncryptionParameterErrorType::InvalidScheme;
            return;
        }
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        ConstPointer<Modulus> plain_modulus = parms.plain_modulus();

        // The number of coeff moduli is restricted to 64 to prevent unexpected behaviors
        if (coeff_modulus.size() > utils::HE_COEFF_MOD_COUNT_MAX
            || coeff_modulus.size() < utils::HE_COEFF_MOD_COUNT_MIN)
        {
            qualifiers.parameter_error = EncryptionParameterErrorType::InvalidCoeffModulusSize;
            return;
        }

        size_t coeff_modulus_size = coeff_modulus.size();
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            // Check coefficient moduli bounds
            if ((coeff_modulus[i].value() >> utils::HE_USER_MOD_BIT_COUNT_MAX) > 0 || 
                (coeff_modulus[i].value() >> (utils::HE_USER_MOD_BIT_COUNT_MIN - 1)) == 0)
            {
                qualifiers.parameter_error = EncryptionParameterErrorType::InvalidCoeffModulusBitCount;
                return;
            }
        }

        // Compute the product of all coeff moduli
        this->total_coeff_modulus_ = Array<uint64_t>(coeff_modulus_size, false, nullptr);
        Array<uint64_t> coeff_modulus_values(coeff_modulus_size, false, nullptr);
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            coeff_modulus_values[i] = coeff_modulus[i].value();
        }
        utils::multiply_many_uint64(coeff_modulus_values.const_reference(), this->total_coeff_modulus_.reference(), nullptr);
        this->total_coeff_modulus_bit_count_ = utils::get_significant_bit_count_uint(
            this->total_coeff_modulus_.const_reference());

        // Check polynomial modulus degree and create poly_modulus
        size_t poly_modulus_degree = parms.poly_modulus_degree();
        if (poly_modulus_degree < utils::HE_POLY_MOD_DEGREE_MIN || poly_modulus_degree > utils::HE_POLY_MOD_DEGREE_MAX) {
            qualifiers.parameter_error = EncryptionParameterErrorType::InvalidPolyModulusDegree;
            return;
        }
        int coeff_count_power_int = utils::get_power_of_two(static_cast<uint64_t>(poly_modulus_degree));
        if (coeff_count_power_int < 0) {
            qualifiers.parameter_error = EncryptionParameterErrorType::InvalidPolyModulusDegreeNonPowerOfTwo;
            return;
        }
        size_t coeff_count_power = static_cast<size_t>(coeff_count_power_int);

        if (coeff_modulus_size * poly_modulus_degree > (1ul<<32)) {
            qualifiers.parameter_error = EncryptionParameterErrorType::InvalidParametersTooLarge;
            return;
        }

        // Polynomial modulus X^(2^k) + 1 is guaranteed at this point
        qualifiers.using_fft = true;

        // Assume parameters satisfy desired security level
        qualifiers.security_level = sec_level;
        // Check if the parameters are secure according to HomomorphicEncryption.org security standard
        if (this->total_coeff_modulus_bit_count_ > CoeffModulus::max_bit_count(poly_modulus_degree, sec_level)) {
            qualifiers.security_level = SecurityLevel::Nil;
            if (sec_level != SecurityLevel::Nil) {
                qualifiers.parameter_error = EncryptionParameterErrorType::InvalidParametersInsecure;
                return;
            } 
        }
        
        // Set up RNSBase for coeff_modulus
        // RNSBase's constructor may fail due to:
        //   (1) coeff_mod not coprime
        //   (2) cannot find inverse of punctured products (because of (1))
        RNSBase coeff_modulus_base;
        try {
            coeff_modulus_base = std::move(RNSBase(coeff_modulus));
        } catch (const std::exception& e) {
            qualifiers.parameter_error = EncryptionParameterErrorType::FailedCreatingRNSBase;
            return;
        }

        // Can we use NTT with coeff_modulus?
        qualifiers.using_ntt = true;
        Array<NTTTables> small_ntt_tables;
        try {
            small_ntt_tables = std::move(NTTTables::create_ntt_tables(coeff_count_power, coeff_modulus));
        } catch (const std::exception& e) {
            qualifiers.using_ntt = false;
            qualifiers.parameter_error = EncryptionParameterErrorType::InvalidCoeffModulusNoNTT;
            return;
        }
        this->small_ntt_tables_ = std::move(small_ntt_tables);

        switch (parms.scheme()) {
            case SchemeType::BFV: case SchemeType::BGV: {

                // Plain modulus must be at least 2 and at most 60 bits
                if ((plain_modulus->value() >> utils::HE_PLAIN_MOD_BIT_COUNT_MAX) > 0 ||
                    (plain_modulus->value() >> (utils::HE_PLAIN_MOD_BIT_COUNT_MIN - 1)) == 0)
                {
                    qualifiers.parameter_error = EncryptionParameterErrorType::InvalidPlainModulusBitCount;
                    return;
                }

                // Check that all coeff moduli are relatively prime to plain_modulus
                for (size_t i = 0; i < coeff_modulus_size; i++) {
                    if (!utils::are_coprime(coeff_modulus[i].value(), plain_modulus->value())) {
                        qualifiers.parameter_error = EncryptionParameterErrorType::InvalidPlainModulusCoprimality;
                        return;
                    }
                }

                // Check that plain_modulus is smaller than total coeff modulus
                uint64_t plain_modulus_value = plain_modulus->value();
                if (!utils::is_less_than_uint(
                    ConstSlice<uint64_t>(&plain_modulus_value, 1, false, nullptr),
                    this->total_coeff_modulus_.const_reference()
                )) {
                    qualifiers.parameter_error = EncryptionParameterErrorType::InvalidPlainModulusTooLarge;
                    return;
                }

                // Can we use batching? (NTT with plain_modulus)
                qualifiers.using_batching = true;
                try {
                    Box<NTTTables> y = utils::Box(new NTTTables(coeff_count_power, *plain_modulus), false, nullptr);
                    this->plain_ntt_tables_ = std::optional(std::move(y));
                } catch (const std::exception& e) {
                    qualifiers.using_batching = false;
                }

                // Check for plain_lift
                // If all the small coefficient moduli are larger than plain modulus, we can quickly
                // lift plain coefficients to RNS form
                qualifiers.using_fast_plain_lift = true;
                for (size_t i = 0; i < coeff_modulus_size; i++) {
                    if (coeff_modulus[i].value() <= plain_modulus->value()) {
                        qualifiers.using_fast_plain_lift = false;
                        break;
                    }
                }
                
                // Calculate coeff_div_plain_modulus (BFV-"Delta") and the remainder upper_half_increment
                Array<uint64_t> temp_coeff_div_plain_modulus(coeff_modulus_size, false, nullptr);
                this->upper_half_increment_ = Array<uint64_t>(coeff_modulus_size, false, nullptr);
                Array<uint64_t> wide_plain_modulus = Array<uint64_t>(coeff_modulus_size, false, nullptr);
                wide_plain_modulus[0] = plain_modulus->value();
                utils::divide_uint(
                    this->total_coeff_modulus_.const_reference(),
                    wide_plain_modulus.const_reference(),
                    temp_coeff_div_plain_modulus.reference(),
                    this->upper_half_increment_.reference(),
                    nullptr
                );
                
                // Store the non-RNS form of upper_half_increment for BFV encryption
                this->coeff_modulus_mod_plain_modulus_ = this->upper_half_increment_[0];

                // Decompose coeff_div_plain_modulus into RNS factors
                coeff_modulus_base.decompose_single(temp_coeff_div_plain_modulus.reference());
                this->coeff_div_plain_modulus_ = Array<MultiplyUint64Operand>(coeff_modulus_size, false, nullptr);
                for (size_t i = 0; i < coeff_modulus_size; i++) {
                    this->coeff_div_plain_modulus_[i] = MultiplyUint64Operand(temp_coeff_div_plain_modulus[i], coeff_modulus[i]);
                }
                
                // Decompose upper_half_increment into RNS factors
                coeff_modulus_base.decompose_single(this->upper_half_increment_.reference());
                
                // Calculate (plain_modulus + 1) / 2.
                this->plain_upper_half_threshold_ = (plain_modulus->value() + 1) >> 1;
                
                // Calculate coeff_modulus - plain_modulus.
                this->plain_upper_half_increment_ = Array<uint64_t>(coeff_modulus_size, false, nullptr);
                if (qualifiers.using_fast_plain_lift) {
                    // Calculate coeff_modulus[i] - plain_modulus if using_fast_plain_lift
                    for (size_t i = 0; i < coeff_modulus_size; i++) {
                        this->plain_upper_half_increment_[i] = coeff_modulus[i].value() - plain_modulus->value();
                    }
                } else {
                    utils::sub_uint(
                        this->total_coeff_modulus_.const_reference(),
                        wide_plain_modulus.const_reference(),
                        this->plain_upper_half_increment_.reference()
                    );
                }

                break;
            }
            case SchemeType::CKKS: {
                // Check that plain_modulus is set to zero
                if (plain_modulus->value() != 0) {
                    qualifiers.parameter_error = EncryptionParameterErrorType::InvalidPlainModulusNonZero;
                    return;
                }

                // When using CKKS batching (BatchEncoder) is always enabled
                qualifiers.using_batching = true;

                // Cannot use fast_plain_lift for CKKS since the plaintext coefficients
                // can easily be larger than coefficient moduli
                qualifiers.using_fast_plain_lift = false;

                // Calculate 2^64 / 2 (most negative plaintext coefficient value)
                this->plain_upper_half_threshold_ = 1ull << 63;
                
                // Calculate plain_upper_half_increment = 2^64 mod coeff_modulus for CKKS plaintexts
                this->plain_upper_half_increment_ = Array<uint64_t>(coeff_modulus_size, false, nullptr);
                for (size_t i = 0; i < coeff_modulus_size; i++) {
                    uint64_t tmp = coeff_modulus[i].reduce(1ull<<63);
                    this->plain_upper_half_increment_[i] = utils::multiply_uint64_mod(
                        tmp, coeff_modulus[i].value() - 2, coeff_modulus[i]
                    );
                }

                // Compute the upper_half_threshold for this modulus.
                this->upper_half_threshold_ = Array<uint64_t>(coeff_modulus_size, false, nullptr);
                utils::increment_uint(
                    this->total_coeff_modulus_.const_reference(),
                    this->upper_half_threshold_.reference()
                );
                utils::right_shift_uint_inplace(
                    this->upper_half_threshold_.reference(),
                    1,
                    coeff_modulus_size
                );

                break;
            }
            default: {
                // This should never be executed
                // because scheme check has been done previously.
                throw std::runtime_error("[ContextData::validate] Unreachable.");
            }
        }
        
        // Create RNSTool
        // RNSTool's constructor may fail due to:
        //   (1) auxiliary base being too large
        //   (2) cannot find inverse of punctured products in auxiliary base
        try {
            this->rns_tool_ = std::optional<RNSTool>(
                std::move(RNSTool(poly_modulus_degree, coeff_modulus_base, *plain_modulus))
            );
        } catch (const std::exception& e) {
            qualifiers.parameter_error = EncryptionParameterErrorType::FailedCreatingRNSTool;
            return;
        }

        // Check whether the coefficient modulus consists of a set of primes that are in decreasing order
        qualifiers.using_descending_modulus_chain = true;
        for (size_t i = 1; i < coeff_modulus_size; i++) {
            if (coeff_modulus[i - 1].value() <= coeff_modulus[i].value()) {
                qualifiers.using_descending_modulus_chain = false;
                break;
            }
        }

        // Create GaloisTool
        try {
            this->galois_tool_ = std::optional<GaloisTool>(GaloisTool(coeff_count_power));
        } catch (const std::exception& e) {
            qualifiers.parameter_error = EncryptionParameterErrorType::FailedCreatingGaloisTool;
            return;
        }

        // Ok
        // std::cerr << "validating context data " << this->parms_id() << " done" << std::endl;

    }

}