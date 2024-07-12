#include "encryption_parameters.h"
#include "utils/serialize.h"

namespace troy {

    const ParmsID parms_id_zero = utils::HashFunction::hash_zero_block;

    void EncryptionParameters::compute_parms_id() {
        if (this->on_device()) {
            throw std::logic_error("[EncryptionParameters::compute_parms_id] Cannot compute parms_id on device");
        }
        size_t coeff_modulus_size = this->coeff_modulus().size();
        size_t total_count = 
            1 + // scheme
            1 + // poly degree
            coeff_modulus_size +
            1 // plain_modulus
            ;
        utils::Array<uint64_t> data(total_count, false, nullptr);
        data[0] = static_cast<uint64_t>(this->scheme());
        data[1] = static_cast<uint64_t>(this->poly_modulus_degree());
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            data[2 + i] = this->coeff_modulus()[i].value();
        }
        data[2 + coeff_modulus_size] = this->plain_modulus()->value();
        utils::HashFunction::hash(data.raw_pointer(), total_count, this->parms_id_);
        // Did we somehow manage to get a zero block as result? This is reserved for
        // plaintexts to indicate non-NTT-transformed form.
        if (this->parms_id() == parms_id_zero) {
            throw std::logic_error("[EncryptionParameters::compute_parms_id] Computed parms_id is zero");
        }
    }

    std::ostream& operator << (std::ostream& os, const EncryptionParameters& parms) {
        if (parms.on_device()) {
            os << parms.to_host();
            return os;
        }
        os << "EncryptionParameters { Scheme = " << parms.scheme() << ", "
            << "PolyModulusDegree = " << parms.poly_modulus_degree() << ", "
            << "CoeffModulus = {";
        size_t count = parms.coeff_modulus().size();
        for (size_t i = 0; i < count; i++) {
            os << parms.coeff_modulus()[i].value();
            if (i < count - 1) {
                os << ", ";
            }
        }
        os << "}, PlainModulus = " << parms.plain_modulus()->value() << " }";
        return os;
    }

    size_t EncryptionParameters::save(std::ostream& stream) const {
        if (this->on_device()) {
            throw std::logic_error("[EncryptionParameters::save] Cannot save a device EncryptionParameters.");
        }
        serialize::save_object(stream, this->scheme());
        serialize::save_object(stream, this->poly_modulus_degree());
        serialize::save_object(stream, this->coeff_modulus().size());
        for (size_t i = 0; i < coeff_modulus().size(); i++) {
            serialize::save_object(stream, this->coeff_modulus()[i].value());
        }
        if (scheme() == SchemeType::BFV || scheme() == SchemeType::BGV) {
            serialize::save_object(stream, this->plain_modulus()->value());
        }
        serialize::save_object(stream, this->use_special_prime_for_encryption());
        return serialized_size_upperbound();
    }

    size_t EncryptionParameters::serialized_size_upperbound() const {
        size_t size = 0;
        size += sizeof(SchemeType);
        size += sizeof(size_t);
        size += sizeof(size_t);
        size += coeff_modulus().size() * sizeof(uint64_t);
        if (scheme() == SchemeType::BFV || scheme() == SchemeType::BGV) {
            size += sizeof(uint64_t);
        }
        size += sizeof(bool);
        return size;
    }

    void EncryptionParameters::load(std::istream& stream) {
        if (this->on_device()) {
            throw std::logic_error("[EncryptionParameters::load] Cannot load into a device EncryptionParameters.");
        }
        SchemeType scheme;
        size_t poly_modulus_degree;
        size_t coeff_modulus_size;
        serialize::load_object(stream, scheme);
        this->scheme_ = scheme;
        serialize::load_object(stream, poly_modulus_degree);
        this->set_poly_modulus_degree(poly_modulus_degree);
        serialize::load_object(stream, coeff_modulus_size);
        std::vector<Modulus> coeff_modulus;
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            uint64_t value;
            serialize::load_object(stream, value);
            coeff_modulus.push_back(Modulus(value));
        }
        this->set_coeff_modulus(coeff_modulus);
        if (scheme == SchemeType::BFV || scheme == SchemeType::BGV) {
            Modulus plain_modulus;
            uint64_t value;
            serialize::load_object(stream, value);
            plain_modulus = Modulus(value);
            this->set_plain_modulus(plain_modulus);
        }
        bool use_special_prime_for_encryption;
        serialize::load_object(stream, use_special_prime_for_encryption);
        this->set_use_special_prime_for_encryption(use_special_prime_for_encryption);
        this->compute_parms_id();
    }

}