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
    
    void Ciphertext::expand_seed(HeContextPointer context) {
        if (!this->contains_seed()) {
            throw std::invalid_argument("[Ciphertext::expand_seed] Ciphertext does not contain seed.");
        }
        std::optional<ContextDataPointer> context_data_optional = context->get_context_data(this->parms_id());
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[Ciphertext::expand_seed] ParmsID is not valid.");
        }
        utils::RandomGenerator c1_prng(this->seed());
        if (this->on_device()) {c1_prng.init_curand_states(this->poly_modulus_degree());}
        utils::ConstSlice<Modulus> coeff_modulus = context_data_optional.value()->parms().coeff_modulus();
        c1_prng.sample_poly_uniform(this->poly(1), this->poly_modulus_degree(), coeff_modulus);
        this->seed() = 0;
    }

    void Ciphertext::save(std::ostream& stream, HeContextPointer context) const {
        serialize::save_object(stream, this->parms_id());
        serialize::save_object(stream, this->polynomial_count());
        serialize::save_object(stream, this->coeff_modulus_size());
        serialize::save_object(stream, this->poly_modulus_degree());
        unsigned char flags = 0;
        flags |= static_cast<unsigned char>(this->is_ntt_form());
        flags |= static_cast<unsigned char>(this->contains_seed()) << 1;
        flags |= static_cast<unsigned char>(this->on_device()) << 2;
        serialize::save_object(stream, flags);

        SchemeType scheme = context->key_context_data().value()->parms().scheme();
        if (scheme == SchemeType::CKKS) {
            serialize::save_object(stream, this->scale());
        }
        if (scheme == SchemeType::BGV) {
            serialize::save_object(stream, this->correction_factor());
        }

        // save data
        if (this->contains_seed()) {
            if (this->polynomial_count() != 2) {
                throw std::logic_error("[Ciphertext::save] Ciphertext contains seed but polynomial count is not 2.");
            }
            // save seed
            serialize::save_object(stream, this->seed());
            size_t poly_size = this->poly_modulus_degree() * this->coeff_modulus_size();
            // only save c0
            if (this->on_device()) {
                utils::Array<uint64_t> c0(poly_size, false);
                c0.copy_from_slice(this->poly(0));
                serialize::save_array(stream, c0.raw_pointer(), poly_size);
            } else {
                serialize::save_array(stream, this->poly(0).raw_pointer(), poly_size);
            }
        } else {
            // save all data
            if (this->on_device()) {
                utils::Array<uint64_t> data(this->data().size(), false);
                data.copy_from_slice(this->data().const_reference());
                serialize::save_array(stream, data.raw_pointer(), data.size());
            } else {
                serialize::save_array(stream, this->data().raw_pointer(), this->data().size());
            }
        }
    }

    void Ciphertext::load(std::istream& stream, HeContextPointer context) {
        serialize::load_object(stream, this->parms_id());
        serialize::load_object(stream, this->polynomial_count_);
        serialize::load_object(stream, this->coeff_modulus_size_);
        serialize::load_object(stream, this->poly_modulus_degree_);

        unsigned char flags;
        serialize::load_object(stream, flags);
        this->is_ntt_form_ = flags & 1;
        bool contains_seed = flags & 2;
        bool device = flags & 4;
        this->data().resize(0);
        this->data().to_host_inplace();

        SchemeType scheme = context->key_context_data().value()->parms().scheme();
        if (scheme == SchemeType::CKKS) {
            serialize::load_object(stream, this->scale_);
        } else {
            this->scale() = 1.0;
        }
        if (scheme == SchemeType::BGV) {
            serialize::load_object(stream, this->correction_factor_);
        } else {
            this->correction_factor() = 1.0;
        }

        if (contains_seed) {
            this->data().resize(this->poly_modulus_degree_ * this->coeff_modulus_size_ * 2);
            serialize::load_object(stream, this->seed_);
            size_t poly_size = this->poly_modulus_degree_ * this->coeff_modulus_size_;
            // load c0
            serialize::load_array(stream, this->poly(0).raw_pointer(), poly_size);
            if (device) this->data().to_device_inplace();
            // expand seed
            this->expand_seed(context);
        } else {
            this->seed() = 0;
            this->data().resize(this->poly_modulus_degree_ * this->coeff_modulus_size_ * this->polynomial_count_);
            // load all data
            serialize::load_array(stream, this->data().raw_pointer(), this->data().size());
            if (device) this->data().to_device_inplace();
        }

    }
    
    size_t Ciphertext::serialized_size(HeContextPointer context) const {
        size_t size = 0;
        size += sizeof(ParmsID); // parms_id
        size += sizeof(size_t); // polynomial_count
        size += sizeof(size_t); // coeff_modulus_size
        size += sizeof(size_t); // poly_modulus_degree
        size += sizeof(unsigned char); // flags
        SchemeType scheme = context->key_context_data().value()->parms().scheme();
        if (scheme == SchemeType::CKKS) {
            size += sizeof(double); // scale
        }
        if (scheme == SchemeType::BGV) {
            size += sizeof(double); // correction_factor
        }
        if (this->contains_seed()) {
            size += sizeof(uint64_t); // seed
            size += this->poly_modulus_degree() * this->coeff_modulus_size() * sizeof(uint64_t); // c0
        } else {
            size += this->poly_modulus_degree() * this->coeff_modulus_size() * this->polynomial_count() * sizeof(uint64_t); // data
        }
        return size;
    }
}