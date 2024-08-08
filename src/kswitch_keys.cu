#include "kswitch_keys.h"

namespace troy {

    size_t KSwitchKeys::save(std::ostream& stream, HeContextPointer context, CompressionMode mode) const {
        size_t total = 0;
        serialize::save_object(stream, this->parms_id());
        total += sizeof(this->parms_id());
        size_t size1d = this->data().size();
        serialize::save_object(stream, size1d);
        total += sizeof(size1d);
        size_t valid_count = 0;
        for (size_t i = 0; i < size1d; i++) {
            if (this->data()[i].size() > 0) {
                valid_count++;
            }
        }
        serialize::save_object(stream, valid_count);
        total += sizeof(valid_count);
        for (size_t i = 0; i < size1d; i++) {
            if (this->data()[i].size() == 0) continue;
            size_t size2d = this->data()[i].size();
            // save id
            serialize::save_object(stream, i);
            total += sizeof(i);
            serialize::save_object(stream, size2d);
            total += sizeof(size2d);
            for (size_t j = 0; j < size2d; j++) {
                total += this->data()[i][j].save(stream, context, mode);
            }
        }
        return total;
    }

    void KSwitchKeys::load(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool) {
        serialize::load_object(stream, this->parms_id());
        size_t size1d;
        serialize::load_object(stream, size1d);
        size_t valid_count;
        serialize::load_object(stream, valid_count);
        this->data().resize(0);
        this->data().resize(size1d);
        for (size_t i = 0; i < valid_count; i++) {
            size_t id;
            serialize::load_object(stream, id);
            size_t size2d;
            serialize::load_object(stream, size2d);
            this->data()[id].resize(size2d);
            for (size_t j = 0; j < size2d; j++) {
                this->data()[id][j].load(stream, context, pool);
            }
        }
        this->build_key_data_ptrs(pool);
    }

    size_t KSwitchKeys::serialized_size_upperbound(HeContextPointer context, CompressionMode mode) const {
        size_t bytes = 0;
        bytes += sizeof(this->parms_id());
        size_t size1d = this->data().size();
        bytes += sizeof(size1d);
        size_t valid_count = 0;
        for (size_t i = 0; i < size1d; i++) {
            if (this->data()[i].size() > 0) {
                valid_count++;
            }
        }
        bytes += sizeof(valid_count);
        for (size_t i = 0; i < size1d; i++) {
            if (this->data()[i].size() == 0) continue;
            size_t size2d = this->data()[i].size();
            // save id
            bytes += sizeof(i);
            bytes += sizeof(size2d);
            for (size_t j = 0; j < size2d; j++) {
                bytes += this->data()[i][j].serialized_size_upperbound(context, mode);
            }
        }
        return bytes;
    }

}