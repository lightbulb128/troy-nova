#pragma once
#include <iostream>
#include <sstream>
#include "box.h"
#include "compression.h"

namespace troy {
    class HeContext;
}

namespace troy {namespace serialize {

    template<typename T>
    inline void save_object(std::ostream& os, const T& obj) {
        os.write(reinterpret_cast<const char*>(&obj), sizeof(T));
    }

    template<typename T>
    inline void load_object(std::istream& is, T& obj) {
        is.read(reinterpret_cast<char*>(&obj), sizeof(T));
    }

    template<typename T>
    inline void save_array(std::ostream& os, const T* arr, size_t size) {
        os.write(reinterpret_cast<const char*>(arr), sizeof(T) * size);
    }

    template<typename T>
    inline void load_array(std::istream& is, T* arr, size_t size) {
        is.read(reinterpret_cast<char*>(arr), sizeof(T) * size);
    }

    inline void save_bool(std::ostream& os, bool b) {
        os.write(reinterpret_cast<const char*>(&b), sizeof(bool));
    }

    inline void load_bool(std::istream& is, bool& b) {
        unsigned char c;
        is.read(reinterpret_cast<char*>(&c), sizeof(unsigned char));
        if (c > 1) {
            throw std::runtime_error("Invalid bool value");
        }
        b = c;
    }

    inline size_t serialized_size_upperbound(size_t raw_size, CompressionMode mode) {
        size_t compressed_size = utils::compression::compressed_size_upperbound(raw_size, mode);
        return compressed_size + sizeof(CompressionMode) + sizeof(size_t);
    }

    // This struct is for obtaining the inner buffer of a std::stringbuf
    struct StringBufWithRawPointer: public std::stringbuf {
        inline char* get_buffer() {return this->pbase();}
    };

    template <typename R>
    inline size_t compress(std::ostream& os, R save_raw, CompressionMode mode) {
        if (mode == CompressionMode::Nil) {
            StringBufWithRawPointer buf;
            std::ostream ss(&buf);
            size_t actual_size = save_raw(ss);
            save_object(os, mode);
            save_object(os, actual_size);
            os.write(buf.get_buffer(), actual_size);
            return actual_size + sizeof(CompressionMode) + sizeof(size_t);
        } else {
            // first serialize without compression
            StringBufWithRawPointer buf;
            std::ostream ss(&buf);
            size_t actual_size = save_raw(ss);
            // then compress
            void* raw_pointer = reinterpret_cast<void*>(buf.get_buffer());
            StringBufWithRawPointer buf_compressed;
            std::ostream os_compressed(&buf_compressed);
            size_t compressed_size = utils::compression::compress(raw_pointer, actual_size, os_compressed, mode);
            if (compressed_size < actual_size) {
                // finally, write mode plus compressed size plus data to os
                save_object(os, mode);
                save_object(os, compressed_size);
                os.write(buf_compressed.get_buffer(), compressed_size);
                // total size is compressed size plus mode and a size_t's size
                return compressed_size + sizeof(CompressionMode) + sizeof(size_t);
            } else {
                // if compression is not effective, write mode plus actual size plus data to os
                save_object(os, CompressionMode::Nil);
                save_object(os, actual_size);
                os.write(buf.get_buffer(), actual_size);
                // total size is actual size plus mode and a size_t's size
                return actual_size + sizeof(CompressionMode) + sizeof(size_t);
            
            }
        }
    }

    template <typename R>
    inline void decompress(std::istream& is, R load_raw) {
        CompressionMode mode;
        load_object(is, mode);
        if (mode == CompressionMode::Nil) {
            size_t actual_size;
            load_object(is, actual_size);
            load_raw(is);
        } else {
            size_t compressed_size;
            load_object(is, compressed_size);
            std::vector<char> buffer(compressed_size);
            is.read(buffer.data(), compressed_size);
            StringBufWithRawPointer buf;
            std::ostream os(&buf);
            utils::compression::decompress(buffer.data(), compressed_size, os, mode);
            std::istream is(&buf);
            load_raw(is);
        }
    }

}}