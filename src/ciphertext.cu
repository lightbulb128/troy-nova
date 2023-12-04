#include "ciphertext.cuh"

namespace troy {

    void Ciphertext::resize_internal(size_t polynomial_count, size_t coeff_modulus_size, size_t poly_modulus_degree) {
        if (polynomial_count < utils::HE_CIPHERTEXT_SIZE_MIN || polynomial_count > utils::HE_CIPHERTEXT_SIZE_MAX) {
            throw std::invalid_argument("[Ciphertext::resize_internal] Polynomial count is invalid.");
        }

        size_t data_size = polynomial_count * coeff_modulus_size * poly_modulus_degree;
        this->data_.resize(data_size);

        this->polynomial_count_ = polynomial_count;
        this->coeff_modulus_size_ = coeff_modulus_size;
        this->poly_modulus_degree_ = poly_modulus_degree;
    }
    
    void Ciphertext::resize(HeContextPointer context, const ParmsID& parms_id, size_t polynomial_count) {
        if (!context->parameters_set()) {
            throw std::invalid_argument("[Ciphertext::resize] Context is not set correctly.");
        }
        std::optional<ContextDataPointer> context_data_opt = context->get_context_data(parms_id);
        if (!context_data_opt.has_value()) {
            throw std::invalid_argument("[Ciphertext::resize] ParmsID is not valid.");
        }
        ContextDataPointer context_data = context_data_opt.value();
        const EncryptionParameters &parms = context_data->parms();
        this->parms_id_ = parms_id;
        this->resize_internal(polynomial_count, parms.coeff_modulus().size(), parms.poly_modulus_degree());
    }
    
    bool Ciphertext::is_transparent() const {
        if (this->data().size() == 0) return true;
        if (this->polynomial_count() < utils::HE_CIPHERTEXT_SIZE_MIN) return true;
        return reduction::nonzero_count(this->data().const_reference()) == 0;
    }

}