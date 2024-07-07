#pragma once
#include <iostream>
#include "box.h"

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
    
}}