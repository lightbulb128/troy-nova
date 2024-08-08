#include "ciphertext.h"

namespace troy {

    Ciphertext Ciphertext::like(const Ciphertext& other, size_t polynomial_count, size_t coeff_modulus_size, bool fill_zeros, MemoryPoolHandle pool) {
        if (polynomial_count < utils::HE_CIPHERTEXT_SIZE_MIN || polynomial_count > utils::HE_CIPHERTEXT_SIZE_MAX) {
            throw std::invalid_argument("[Ciphertext::like] Polynomial count is invalid.");
        }
        Ciphertext ret;
        ret.polynomial_count_ = polynomial_count;
        ret.coeff_modulus_size_ = coeff_modulus_size;
        ret.poly_modulus_degree_ = other.poly_modulus_degree();
        ret.parms_id_ = other.parms_id();
        ret.scale_ = other.scale();
        ret.is_ntt_form_ = other.is_ntt_form();
        ret.correction_factor_ = other.correction_factor();
        size_t data_size = polynomial_count * coeff_modulus_size * other.poly_modulus_degree();
        ret.data_ = utils::DynamicArray<uint64_t>::create_uninitialized(data_size, other.on_device(), pool);
        if (fill_zeros) {
            ret.data().reference().set_zero();
        }
        return ret;
    }

    void Ciphertext::resize_internal(size_t polynomial_count, size_t coeff_modulus_size, size_t poly_modulus_degree, bool fill_extra_with_zeros, bool copy_data) {
        if (polynomial_count < utils::HE_CIPHERTEXT_SIZE_MIN || polynomial_count > utils::HE_CIPHERTEXT_SIZE_MAX) {
            throw std::invalid_argument("[Ciphertext::resize_internal] Polynomial count is invalid.");
        }

        size_t data_size = polynomial_count * coeff_modulus_size * poly_modulus_degree;
        if (fill_extra_with_zeros) {
            this->data_.resize(data_size, copy_data);
        } else {
            this->data_.resize_uninitialized(data_size, copy_data);
        }

        this->polynomial_count_ = polynomial_count;
        this->coeff_modulus_size_ = coeff_modulus_size;
        this->poly_modulus_degree_ = poly_modulus_degree;
    }
    
    void Ciphertext::resize(HeContextPointer context, const ParmsID& parms_id, size_t polynomial_count, bool fill_extra_with_zeros, bool copy_data) {
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
        this->resize_internal(polynomial_count, parms.coeff_modulus().size(), parms.poly_modulus_degree(), fill_extra_with_zeros, copy_data);
    }

    void Ciphertext::reconfigure_like(HeContextPointer context, const Ciphertext& other, size_t polynomial_count, bool fill_extra_with_zeros) {
        if (!context->parameters_set()) {
            throw std::invalid_argument("[Ciphertext::resize] Context is not set correctly.");
        }
        std::optional<ContextDataPointer> context_data_opt = context->get_context_data(other.parms_id());
        if (!context_data_opt.has_value()) {
            throw std::invalid_argument("[Ciphertext::resize] ParmsID is not valid.");
        }
        ContextDataPointer context_data = context_data_opt.value();
        const EncryptionParameters &parms = context_data->parms();
        this->parms_id_ = other.parms_id();
        this->resize_internal(polynomial_count, parms.coeff_modulus().size(), parms.poly_modulus_degree(), fill_extra_with_zeros, true);
        this->correction_factor_ = other.correction_factor();
        this->scale_ = other.scale();
        this->is_ntt_form_ = other.is_ntt_form();
    }
    
    bool Ciphertext::is_transparent() const {
        if (this->data().size() == 0) return true;
        if (this->polynomial_count() < utils::HE_CIPHERTEXT_SIZE_MIN) return true;
        return reduction::nonzero_count(this->data().const_reference(), pool()) == 0;
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
        utils::ConstSlice<Modulus> coeff_modulus = context_data_optional.value()->parms().coeff_modulus();
        c1_prng.sample_poly_uniform(this->poly(1), this->poly_modulus_degree(), coeff_modulus);
        this->seed() = 0;
    }

    size_t Ciphertext::save_raw(std::ostream& stream, HeContextPointer context) const {
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
                utils::Array<uint64_t> c0(poly_size, false, nullptr);
                c0.copy_from_slice(this->poly(0));
                serialize::save_array(stream, c0.raw_pointer(), poly_size);
            } else {
                serialize::save_array(stream, this->poly(0).raw_pointer(), poly_size);
            }
        } else {
            // save all data
            if (this->on_device()) {
                utils::Array<uint64_t> data(this->data().size(), false, nullptr);
                data.copy_from_slice(this->data().const_reference());
                serialize::save_array(stream, data.raw_pointer(), data.size());
            } else {
                serialize::save_array(stream, this->data().raw_pointer(), this->data().size());
            }
        }
        return this->serialized_raw_size(context);
    }

    void Ciphertext::load_raw(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool) {
        serialize::load_object(stream, this->parms_id());
        serialize::load_object(stream, this->polynomial_count_);
        serialize::load_object(stream, this->coeff_modulus_size_);
        serialize::load_object(stream, this->poly_modulus_degree_);

        unsigned char flags;
        serialize::load_object(stream, flags);
        this->is_ntt_form_ = flags & 1;
        bool contains_seed = flags & 2;
        bool device = flags & 4;
        bool terms = flags & 8;
        if (terms) {
            throw std::logic_error("[Ciphertext::load] Trying to call load with ciphertext with only terms saved.");
        }
        this->data().resize(0, false);
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
            this->data().resize(this->poly_modulus_degree_ * this->coeff_modulus_size_ * 2, true);
            serialize::load_object(stream, this->seed_);
            size_t poly_size = this->poly_modulus_degree_ * this->coeff_modulus_size_;
            // load c0
            serialize::load_array(stream, this->poly(0).raw_pointer(), poly_size);
            if (device) this->data().to_device_inplace(pool);
            // expand seed
            this->expand_seed(context);
        } else {
            this->seed() = 0;
            this->data().resize(this->poly_modulus_degree_ * this->coeff_modulus_size_ * this->polynomial_count_, true);
            // load all data
            serialize::load_array(stream, this->data().raw_pointer(), this->data().size());
            if (device) this->data().to_device_inplace(pool);
        }

    }
    
    size_t Ciphertext::serialized_raw_size(HeContextPointer context) const {
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

    size_t Ciphertext::save_terms_raw(std::ostream& stream, HeContextPointer context, const std::vector<size_t>& terms, MemoryPoolHandle pool) const {
        serialize::save_object(stream, this->parms_id());
        serialize::save_object(stream, this->polynomial_count());
        serialize::save_object(stream, this->coeff_modulus_size());
        serialize::save_object(stream, this->poly_modulus_degree());
        unsigned char flags = 0;
        flags |= static_cast<unsigned char>(this->is_ntt_form());
        flags |= static_cast<unsigned char>(this->contains_seed()) << 1;
        flags |= static_cast<unsigned char>(this->on_device()) << 2;
        flags |= 1 << 3; // terms
        serialize::save_object(stream, flags);

        SchemeType scheme = context->key_context_data().value()->parms().scheme();
        if (scheme == SchemeType::CKKS) {
            serialize::save_object(stream, this->scale());
        }
        if (scheme == SchemeType::BGV) {
            serialize::save_object(stream, this->correction_factor());
        }

        // save data
        size_t poly_size = this->poly_modulus_degree() * this->coeff_modulus_size();
        if (this->contains_seed()) {
            if (this->polynomial_count() != 2) {
                throw std::logic_error("[Ciphertext::save] Ciphertext contains seed but polynomial count is not 2.");
            }
            // save seed
            serialize::save_object(stream, this->seed());
        }

        utils::ConstSlice<uint64_t> c0(nullptr, 0, false, nullptr);
        utils::Array<uint64_t> temp_buffer(0, false, nullptr);
        bool device = this->on_device();
        if (is_ntt_form_) {
            utils::ConstSlice<utils::NTTTables> ntt_tables = context->get_context_data(this->parms_id()).value()->small_ntt_tables();
            temp_buffer = utils::Array<uint64_t>::create_and_copy_from_slice(this->poly(0), pool);
            utils::intt_inplace_p(temp_buffer.reference(), this->poly_modulus_degree(), ntt_tables);
            temp_buffer.to_host_inplace();
            c0 = temp_buffer.const_reference();
        } else {
            if (device) {
                temp_buffer = utils::Array<uint64_t>::create_and_copy_from_slice(this->poly(0), pool);
                temp_buffer.to_host_inplace();
                c0 = temp_buffer.const_reference();
            } else {
                c0 = this->poly(0);
            }
        }
        // save c0
        for (size_t j = 0; j < this->coeff_modulus_size(); j++) {
            for (size_t i = 0; i < terms.size(); i++) {
                serialize::save_object(stream, c0[j * this->poly_modulus_degree() + terms[i]]);
            }
        }
        // save remaining polys
        size_t start_polynomial = contains_seed() ? 2 : 1;
        if (this->on_device()) {
            utils::Array<uint64_t> data(this->data().size() - poly_size * start_polynomial, false, nullptr);
            data.copy_from_slice(this->polys(start_polynomial, this->polynomial_count_));
            serialize::save_array(stream, data.raw_pointer(), data.size());
        } else {
            utils::ConstSlice<uint64_t> data = this->polys(start_polynomial, this->polynomial_count_);
            serialize::save_array(stream, data.raw_pointer(), data.size());
        }

        return this->serialized_terms_raw_size(context, terms);
    }

    
    void Ciphertext::load_terms_raw(std::istream& stream, HeContextPointer context, const std::vector<size_t>& terms, MemoryPoolHandle pool) {
        serialize::load_object(stream, this->parms_id());
        serialize::load_object(stream, this->polynomial_count_);
        serialize::load_object(stream, this->coeff_modulus_size_);
        serialize::load_object(stream, this->poly_modulus_degree_);

        unsigned char flags;
        serialize::load_object(stream, flags);
        this->is_ntt_form_ = flags & 1;
        bool contains_seed = flags & 2;
        bool device = flags & 4;
        bool terms_flag = flags & 8;
        if (!terms_flag) {
            throw std::logic_error("[Ciphertext::load_terms] Trying to call load_terms with ciphertext with all terms saved.");
        }
        this->data().resize(0, false);
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

        size_t poly_size = this->poly_modulus_degree_ * this->coeff_modulus_size_;
        if (contains_seed) {
            this->data().resize(poly_size * 2, true);
            serialize::load_object(stream, this->seed_);
        } else {
            this->data().resize(poly_size * polynomial_count_, true);
            this->seed() = 0;
        }

        utils::Slice<uint64_t> c0 = this->poly(0);
        for (size_t j = 0; j < this->coeff_modulus_size(); j++) {
            for (size_t i = 0; i < terms.size(); i++) {
                serialize::load_object(stream, c0[j * this->poly_modulus_degree() + terms[i]]);
            }
        }

        // load remaining polys
        size_t start_polynomial = contains_seed ? 2 : 1;
        utils::Slice<uint64_t> polys = this->polys(start_polynomial, this->polynomial_count_);
        serialize::load_array(stream, polys.raw_pointer(), polys.size());
        if (device) this->data().to_device_inplace(pool);
        if (is_ntt_form_) {
            utils::ConstSlice<utils::NTTTables> ntt_tables = context->get_context_data(this->parms_id()).value()->small_ntt_tables();
            utils::ntt_inplace_p(this->poly(0), this->poly_modulus_degree(), ntt_tables);
        }
        // expand seed
        if (contains_seed) this->expand_seed(context);
    }

    size_t Ciphertext::serialized_terms_raw_size(HeContextPointer context, const std::vector<size_t>& terms) const {
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
        }
        size += terms.size() * sizeof(uint64_t) * this->coeff_modulus_size(); // c0
        size_t start_polynomial = this->contains_seed() ? 2 : 1;
        size += this->poly_modulus_degree() * this->coeff_modulus_size() * (this->polynomial_count() - start_polynomial) * sizeof(uint64_t); // all polynomials after c0
        return size;
    }
}