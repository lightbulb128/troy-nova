#include "encryption_parameters.h"

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
}