#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <iostream>
#include "../../src/troy.cuh"
#include "pybind11/cast.h"

PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<std::complex<double>>);

namespace py = pybind11;
using namespace troy;
using namespace troy::linear;
using std::vector;
using std::complex;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::string;
using troy::utils::ConstSlice;
using troy::utils::Slice;
using troy::utils::Array;
using uint128_t = __uint128_t;

template <typename T>
vector<T> get_vector_from_buffer(const py::array_t<T>& values) {
    py::buffer_info buf = values.request();
    T* ptr = reinterpret_cast<T*>(buf.ptr);
    vector<T> vec(buf.shape[0]);
    for (int i = 0; i < buf.shape[0]; i++) {
        vec[i] = ptr[i];
    }
    return vec;
}

template<typename T> 
T* get_pointer_from_buffer(const py::array_t<T>& values) {
    py::buffer_info buf = values.request();
    T* ptr = reinterpret_cast<T*>(buf.ptr);
    return ptr;
}

template <typename T>
py::array_t<T> get_buffer_from_vector(const vector<T>& vec) {
    py::array_t<T> buffer(vec.size());
    py::buffer_info buf = buffer.request();
    T* ptr = reinterpret_cast<T*>(buf.ptr);
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = vec[i];
    }
    return buffer;
}

template <typename T>
py::array_t<T> get_buffer_from_slice(ConstSlice<T> slice) {
    if (slice.on_device()) {
        Array<T> host(slice.size(), false);
        host.copy_from_slice(slice);
        return get_buffer_from_slice(host.const_reference());
    }
    py::array_t<T> buffer(slice.size());
    py::buffer_info buf = buffer.request();
    T* ptr = reinterpret_cast<T*>(buf.ptr);
    for (int i = 0; i < slice.size(); i++) {
        ptr[i] = slice[i];
    }
    return buffer;
}

template <typename T>
py::array_t<T> get_buffer_from_slice(Slice<T> slice) {
    return get_buffer_from_slice(slice.as_const());
}

template <typename T>
vector<T> get_vector_from_slice(ConstSlice<T> slice) {
    if (slice.on_device()) {
        Array<T> host(slice.size(), false);
        host.copy_from_slice(slice);
        return get_vector_from_slice(host.const_reference());
    }
    vector<T> buffer(slice.size());
    for (int i = 0; i < slice.size(); i++) {
        buffer[i] = slice[i];
    }
    return buffer;
}

template <typename T>
vector<T> get_buffer_from_slice(Slice<T> slice) {
    return get_vector_from_slice(slice.as_const());
}

template <typename T>
py::array_t<T> get_buffer_from_array(const Array<T>& array) {
    return get_buffer_from_slice(array.const_reference());
}

template <typename T>
std::string to_string(const T& object) {
    std::ostringstream ss;
    ss << object;
    return ss.str();
}

template <typename T>
py::bytes save(const T& object) {
    std::ostringstream ss;
    object.save(ss);
    return py::bytes(ss.str());
}

template <typename T>
void load(T& object, const py::bytes& str, MemoryPoolHandle pool) {
    std::istringstream ss(str);
    object.load(ss, pool);
}

template <typename T>
T load_new(const py::bytes& str, MemoryPoolHandle pool) {
    std::istringstream ss(str);
    T object;
    object.load(ss, pool);
    return object;
}


template <typename T>
size_t serialized_size(const T& object) {
    return object.serialized_size();
}

template <typename T>
py::bytes save_he(const T& object, HeContextPointer context) {
    std::ostringstream ss;
    object.save(ss, context);
    return py::bytes(ss.str());
}

template <typename T>
void load_he(T& object, const py::bytes& str, HeContextPointer context, MemoryPoolHandle pool) {
    std::istringstream ss(str);
    object.load(ss, context, pool);
}

template <typename T>
T load_new_he(const py::bytes& str, HeContextPointer context, MemoryPoolHandle pool) {
    std::istringstream ss(str);
    T object;
    object.load(ss, context, pool);
    return object;
}


template <typename T>
size_t serialized_size_he(const T& object, HeContextPointer context) {
    return object.serialized_size(context);
}

typedef std::optional<MemoryPoolHandle> MemoryPoolHandleArgument;

inline MemoryPoolHandle nullopt_default_pool(MemoryPoolHandleArgument pool) {
    if (!pool.has_value()) {
        return MemoryPool::GlobalPool();
    }
    return pool.value();
}

#define MEMORY_POOL_ARGUMENT py::arg_v("pool", std::nullopt, "MemoryPool.global_pool()")
#define OPTIONAL_PARMS_ID_ARGUMENT py::arg("parms_id") = std::nullopt

void register_basics(pybind11::module& m);
void register_modulus(pybind11::module& m);
void register_encryption_parameters(pybind11::module& m);
void register_he_context(pybind11::module& m);
void register_plaintext(pybind11::module& m);
void register_ciphertext(pybind11::module& m);
void register_lwe_ciphertext(pybind11::module& m);
void register_secret_key(pybind11::module& m);
void register_public_key(pybind11::module& m);
void register_kswitch_keys(pybind11::module& m);
void register_key_generator(pybind11::module& m);
void register_batch_encoder(pybind11::module& m);
void register_ckks_encoder(pybind11::module& m);
void register_polynomial_encoder_ring2k(pybind11::module& m);
void register_random_generator(pybind11::module& m);
void register_encryptor(pybind11::module& m);
void register_decryptor(pybind11::module& m);
void register_evaluator(pybind11::module& m);
void register_matmul_helper(pybind11::module& m);
void register_conv2d_helper(pybind11::module& m);