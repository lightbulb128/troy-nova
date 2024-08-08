#include "plaintext.h"

namespace troy {
    
    Plaintext Plaintext::like(const Plaintext& other, bool fill_zeros, MemoryPoolHandle pool) {
        Plaintext result;
        result.coeff_count_ = other.coeff_count_;
        result.parms_id_ = other.parms_id_;
        result.scale_ = other.scale_;
        result.is_ntt_form_ = other.is_ntt_form_;
        result.coeff_modulus_size_ = other.coeff_modulus_size_;
        result.poly_modulus_degree_ = other.poly_modulus_degree_;
        result.data_ = utils::DynamicArray<uint64_t>::create_uninitialized(other.data().size(), other.on_device(), pool);
        if (fill_zeros) {
            result.reference().set_zero();
        }
        return result;
    }

    size_t Plaintext::save_raw(std::ostream& stream) const {
        serialize::save_object(stream, this->parms_id());
        serialize::save_object(stream, this->scale());
        serialize::save_object(stream, this->coeff_count_);
        serialize::save_bool(stream, this->on_device());
        serialize::save_object(stream, this->data().size());
        if (this->on_device()) {
            utils::DynamicArray<uint64_t> data_host = this->data().to_host();
            serialize::save_array(stream, data_host.raw_pointer(), data_host.size());
        } else {
            serialize::save_array(stream, this->data().raw_pointer(), this->data().size());
        }
        serialize::save_bool(stream, this->is_ntt_form());
        serialize::save_object(stream, this->poly_modulus_degree());
        serialize::save_object(stream, this->coeff_modulus_size());
        return this->serialized_raw_size();
    }

    void Plaintext::load_raw(std::istream& stream, MemoryPoolHandle pool) {
        serialize::load_object(stream, this->parms_id());
        serialize::load_object(stream, this->scale());
        serialize::load_object(stream, this->coeff_count_);
        bool device;
        serialize::load_bool(stream, device);
        size_t size;
        serialize::load_object(stream, size);
        this->data().resize(0, false); this->data().to_host_inplace();
        this->data().resize(size, true);
        serialize::load_array(stream, this->data().raw_pointer(), size);
        if (device) {
            this->data().to_device_inplace(pool);
        }
        serialize::load_bool(stream, this->is_ntt_form());
        serialize::load_object(stream, this->poly_modulus_degree());
        serialize::load_object(stream, this->coeff_modulus_size());
    }
    
    size_t Plaintext::serialized_raw_size() const {
        size_t size = 0;
        size += sizeof(ParmsID); // parms_id
        size += sizeof(double); // scale
        size += sizeof(size_t); // coeff_count
        size += sizeof(bool); // on_device
        size += sizeof(size_t); // data.size()
        size += this->data().size() * sizeof(uint64_t); // data
        size += sizeof(bool); // is_ntt_form
        size += sizeof(size_t); // poly_modulus_degree
        size += sizeof(size_t); // coeff_modulus_size
        return size;
    }

    void Plaintext::resize_rns_internal(size_t poly_modulus_degree, size_t coeff_modulus_size, size_t coeff_count, bool fill_extra_with_zeros, bool copy_data) {
        size_t data_size = coeff_count * coeff_modulus_size;
        if (fill_extra_with_zeros) {
            this->data().resize(data_size, copy_data);
        } else {
            this->data().resize_uninitialized(data_size, copy_data);
        }
        this->poly_modulus_degree_ = poly_modulus_degree;
        this->coeff_modulus_size_ = coeff_modulus_size;
        this->coeff_count_ = coeff_count;
    }

    void Plaintext::resize_rns(const HeContext& context, const ParmsID& parms_id, bool fill_extra_with_zeros, bool copy_data) {
        if (!context.parameters_set()) {
            throw std::invalid_argument("[Plaintext::resize_rns] context is not set");
        }
        std::optional<ContextDataPointer> context_data_optional = context.get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[Plaintext::resize_rns] parms_id is not valid");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        this->parms_id_ = parms_id;
        this->resize_rns_internal(parms.poly_modulus_degree(), parms.coeff_modulus().size(), parms.poly_modulus_degree(), fill_extra_with_zeros, copy_data);
    }

    void Plaintext::resize_rns_partial(const HeContext& context, const ParmsID& parms_id, size_t coeff_count, bool fill_extra_with_zeros, bool copy_data) {
        if (!context.parameters_set()) {
            throw std::invalid_argument("[Plaintext::resize_rns] context is not set");
        }
        std::optional<ContextDataPointer> context_data_optional = context.get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[Plaintext::resize_rns] parms_id is not valid");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        this->parms_id_ = parms_id;
        this->resize_rns_internal(parms.poly_modulus_degree(), parms.coeff_modulus().size(), coeff_count, fill_extra_with_zeros, copy_data);
    }


}