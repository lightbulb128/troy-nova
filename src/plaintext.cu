#include "plaintext.h"

namespace troy {

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
        this->data().resize(0); this->data().to_host_inplace();
        this->data().resize(size);
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

    void Plaintext::resize_rns_internal(size_t poly_modulus_degree, size_t coeff_modulus_size) {
        size_t data_size = poly_modulus_degree * coeff_modulus_size;
        this->data().resize(data_size);
        this->poly_modulus_degree_ = poly_modulus_degree;
        this->coeff_modulus_size_ = coeff_modulus_size;
    }

    void Plaintext::resize_rns(const HeContext& context, const ParmsID& parms_id) {
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
        this->resize_rns_internal(parms.poly_modulus_degree(), parms.coeff_modulus().size());
    }


}