#include "plaintext.cuh"

namespace troy {

    void Plaintext::save(std::ostream& stream) const {
        serialize::save_object(stream, this->parms_id());
        serialize::save_object(stream, this->scale());
        serialize::save_object(stream, this->coeff_count());
        serialize::save_bool(stream, this->on_device());
        serialize::save_object(stream, this->data().size());
        if (this->on_device()) {
            utils::DynamicArray<uint64_t> data_host = this->data().to_host();
            serialize::save_array(stream, data_host.raw_pointer(), data_host.size());
        } else {
            serialize::save_array(stream, this->data().raw_pointer(), this->data().size());
        }
    }

    void Plaintext::load(std::istream& stream) {
        serialize::load_object(stream, this->parms_id());
        serialize::load_object(stream, this->scale());
        serialize::load_object(stream, this->coeff_count());
        bool device;
        serialize::load_bool(stream, device);
        size_t size;
        serialize::load_object(stream, size);
        this->data().resize(0); this->data().to_host_inplace();
        this->data().resize(size);
        serialize::load_array(stream, this->data().raw_pointer(), size);
        if (device) {
            this->data().to_device_inplace();
        }
    }
    
    size_t Plaintext::serialized_size() const {
        size_t size = 0;
        size += sizeof(ParmsID); // parms_id
        size += sizeof(double); // scale
        size += sizeof(size_t); // coeff_count
        size += sizeof(bool); // on_device
        size += sizeof(size_t); // data.size()
        size += this->data().size() * sizeof(uint64_t); // data
        return size;
    }

}