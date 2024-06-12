#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <iostream>
#include "../src/troy.cuh"

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
void load(T& object, const py::bytes& str) {
    std::istringstream ss(str);
    object.load(ss);
}

template <typename T>
T load_new(const py::bytes& str) {
    std::istringstream ss(str);
    T object;
    object.load(ss);
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
void load_he(T& object, const py::bytes& str, HeContextPointer context) {
    std::istringstream ss(str);
    object.load(ss, context);
}

template <typename T>
T load_new_he(const py::bytes& str, HeContextPointer context) {
    std::istringstream ss(str);
    T object;
    object.load(ss, context);
    return object;
}


template <typename T>
size_t serialized_size_he(const T& object, HeContextPointer context) {
    return object.serialized_size(context);
}

PYBIND11_MODULE(pytroy_raw, m) {

    m
        .def("it_works", []() {
            return 42;
        })
        .def("initialize_kernel", [](int device){
            troy::kernel_provider::initialize(device);
        })
        .def("destroy_memory_pool", []{
            troy::utils::MemoryPool::Destroy();
        })
    ;

    py::enum_<SchemeType>(m, "SchemeType")
        .value("Nil", SchemeType::Nil)
        .value("BFV", SchemeType::BFV)
        .value("CKKS", SchemeType::CKKS)
        .value("BGV", SchemeType::BGV)
    ;

    py::enum_<SecurityLevel>(m, "SecurityLevel")
        .value("Nil", SecurityLevel::Nil)
        .value("Classical128", SecurityLevel::Classical128)
        .value("Classical192", SecurityLevel::Classical192)
        .value("Classical256", SecurityLevel::Classical256)
    ;

    py::class_<Modulus>(m, "Modulus")
        .def(py::init<uint64_t>(), py::arg("value") = 0)
        .def("is_prime", &Modulus::is_prime)
        .def("is_zero", &Modulus::is_zero)
        .def("value", &Modulus::value)
        .def("bit_count", &Modulus::bit_count)
        .def("__str__", [](const Modulus& m){ return to_string(m); })
        .def("__repr__", [](const Modulus& m){ return to_string(m); })
        .def("reduce", &Modulus::reduce)
        .def("reduce_mul", [](const Modulus& m, uint64_t a, uint64_t b){
            return m.reduce_mul_uint64(a, b);
        })
    ;

    py::class_<CoeffModulus>(m, "CoeffModulus")
        .def_static("max_bit_count", &CoeffModulus::max_bit_count,
            py::arg("poly_modulus_degree"), py::arg_v("sec_level", SecurityLevel::Classical128, "SecurityLevel.Classical128"))
        .def_static("bfv_default", [](size_t poly_modulus_degree, SecurityLevel sec){
            return get_buffer_from_array(CoeffModulus::bfv_default(poly_modulus_degree, sec));
        }, py::arg("poly_modulus_degree"), py::arg_v("sec_level", SecurityLevel::Classical128, "SecurityLevel.Classical128"))
        .def_static("create", [](size_t poly_modulus_degree, const std::vector<int>& bit_sizes){
            vector<size_t> bit_sizes_copy(bit_sizes.size(), 0);
            for (int i = 0; i < bit_sizes.size(); i++) {
                bit_sizes_copy[i] = bit_sizes[i];
            }
            return CoeffModulus::create(poly_modulus_degree, bit_sizes_copy).to_vector();
        })
    ;

    py::class_<PlainModulus>(m, "PlainModulus")
        .def_static("batching", &PlainModulus::batching,
            py::arg("poly_modulus_degree"), py::arg_v("sec_level", SecurityLevel::Classical128, "SecurityLevel.Classical128"))
        .def_static("batching_multiple", [](size_t poly_modulus_degree, const std::vector<int>& bit_sizes){
            vector<size_t> bit_sizes_copy(bit_sizes.size(), 0);
            for (int i = 0; i < bit_sizes.size(); i++) {
                bit_sizes_copy[i] = bit_sizes[i];
            }
            return PlainModulus::batching_multiple(poly_modulus_degree, bit_sizes_copy).to_vector();
        })
    ;

    py::class_<ParmsID>(m, "ParmsID")
        .def_static("zero", [](){
            return parms_id_zero;
        })
        .def("to_vector", [](const ParmsID& p){
            vector<uint64_t> ret; ret.reserve(utils::HashFunction::hash_block_uint64_count);
            for (int i = 0; i < utils::HashFunction::hash_block_uint64_count; i++) {
                ret.push_back(p[i]);
            }
            return get_buffer_from_vector(ret);
        })
    ;

    py::class_<EncryptionParameters>(m, "EncryptionParameters")
        .def(py::init<SchemeType>())
        .def("set_poly_modulus_degree", &EncryptionParameters::set_poly_modulus_degree)
        .def("set_plain_modulus", [](EncryptionParameters& self, const Modulus& plain_modulus){
            self.set_plain_modulus(plain_modulus);
        })
        .def("set_plain_modulus", [](EncryptionParameters& self, uint64_t plain_modulus){
            self.set_plain_modulus(Modulus(plain_modulus));
        })
        .def("set_coeff_modulus", [](EncryptionParameters& self, const vector<Modulus>& coeff_modulus){
            self.set_coeff_modulus(coeff_modulus);
        })
        .def("scheme", &EncryptionParameters::scheme)
        .def("parms_id", &EncryptionParameters::parms_id)
        .def("poly_modulus_degree", &EncryptionParameters::poly_modulus_degree)
        .def("plain_modulus", [](const EncryptionParameters& self){
            return self.plain_modulus_host();
        })
        .def("coeff_modulus", [](const EncryptionParameters& self){
            return get_vector_from_slice(self.coeff_modulus());
        })
        .def("__str__", [](const EncryptionParameters& self){ return to_string(self); })
        .def("__repr__", [](const EncryptionParameters& self){ return to_string(self); })
    ;

    py::class_<ContextData, std::shared_ptr<ContextData>>(m, "ContextData")
        .def("parms", [](const ContextData& self){
            return self.parms();
        })
        .def("parms_id", [](const ContextData& self){
            return self.parms_id();
        })
        .def("chain_index", [](const ContextData& self){
            return self.chain_index();
        })
        .def("prev_context_data", [](const ContextData& self){
            if (self.prev_context_data()) {
                return std::optional(self.prev_context_data().value().lock());
            } else {
                return std::optional<ContextDataPointer>(std::nullopt);
            }
        })
        .def("next_context_data", [](const ContextData& self){
            return self.next_context_data();
        })
        .def("on_device", [](const ContextData& self){
            return self.on_device();
        })
    ;

    py::class_<HeContext, std::shared_ptr<HeContext>>(m, "HeContext")
        .def(py::init([](const EncryptionParameters& parms, bool expand_mod_chain, SecurityLevel sec_level, uint64_t random_seed){
            return HeContext::create(parms, expand_mod_chain, sec_level, random_seed);
        }), py::arg("parms"), py::arg("expand_mod_chain") = true, py::arg_v("sec_level", SecurityLevel::Classical128, "SecurityLevel.Classical128"), py::arg("random_seed") = 0)
        .def("to_device_inplace", [](const HeContextPointer& self){
            return self->to_device_inplace();
        })
        .def("get_context_data", [](const HeContextPointer& self, const ParmsID& parms_id){
            return self->get_context_data(parms_id);
        })
        .def("first_context_data", [](const HeContextPointer& self){
            return self->first_context_data();
        })
        .def("last_context_data", [](const HeContextPointer& self){
            return self->last_context_data();
        })
        .def("key_context_data", [](const HeContextPointer& self){
            return self->key_context_data();
        })
        .def("first_parms_id", [](const HeContextPointer& self){
            return self->first_parms_id();
        })
        .def("last_parms_id", [](const HeContextPointer& self){
            return self->last_parms_id();
        })
        .def("key_parms_id", [](const HeContextPointer& self){
            return self->key_parms_id();
        })
        .def("using_keyswitching", [](const HeContextPointer& self){
            return self->using_keyswitching();
        })
        .def("on_device", [](const HeContextPointer& self){
            return self->on_device();
        })
    ;

    py::class_<Plaintext>(m, "Plaintext")
        .def(py::init<>())
        .def("clone", &Plaintext::clone)
        .def("to_device_inplace", &Plaintext::to_device_inplace)
        .def("to_host_inplace", &Plaintext::to_host_inplace)
        .def("to_device", &Plaintext::to_device)
        .def("to_host", &Plaintext::to_host)
        .def("on_device", &Plaintext::on_device)
        .def("parms_id", py::overload_cast<>(&Plaintext::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](Plaintext& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("scale", py::overload_cast<>(&Plaintext::scale, py::const_))
        .def("set_scale", [](Plaintext& self, double scale){ self.scale() = scale; })
        .def("coeff_count", py::overload_cast<>(&Plaintext::coeff_count, py::const_))
        .def("set_coeff_count", [](Plaintext& self, size_t coeff_count){ self.coeff_count() = coeff_count; })
        .def("resize", &Plaintext::resize)
        .def("is_ntt_form", &Plaintext::is_ntt_form)
        .def("save", [](const Plaintext& self) {return save(self); })
        .def("load", [](Plaintext& self, const py::bytes& str) {return load<Plaintext>(self, str); })
        .def_static("load_new", [](const py::bytes& str) {return load_new<Plaintext>(str); })
        .def("serialized_size", [](const Plaintext& self) {return serialized_size(self); })
    ;

    py::class_<Ciphertext>(m, "Ciphertext")
        .def(py::init<>())
        .def("clone", &Ciphertext::clone)
        .def("to_device_inplace", &Ciphertext::to_device_inplace)
        .def("to_host_inplace", &Ciphertext::to_host_inplace)
        .def("to_device", &Ciphertext::to_device)
        .def("to_host", &Ciphertext::to_host)
        .def("on_device", &Ciphertext::on_device)
        .def("parms_id", py::overload_cast<>(&Ciphertext::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](Ciphertext& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("scale", py::overload_cast<>(&Ciphertext::scale, py::const_))
        .def("set_scale", [](Ciphertext& self, double scale){ self.scale() = scale; })
        .def("polynomial_count", py::overload_cast<>(&Ciphertext::polynomial_count, py::const_))
        .def("coeff_modulus_size", py::overload_cast<>(&Ciphertext::coeff_modulus_size, py::const_))
        .def("poly_modulus_degree", py::overload_cast<>(&Ciphertext::poly_modulus_degree, py::const_))
        .def("is_ntt_form", py::overload_cast<>(&Ciphertext::is_ntt_form, py::const_))
        .def("set_is_ntt_form", [](Ciphertext& self, bool is_ntt_form){ self.is_ntt_form() = is_ntt_form; })
        .def("correction_factor", py::overload_cast<>(&Ciphertext::correction_factor, py::const_))
        .def("set_correction_factor", [](Ciphertext& self, uint64_t correction_factor){ self.correction_factor() = correction_factor; })
        .def("seed", py::overload_cast<>(&Ciphertext::seed, py::const_))
        .def("set_seed", [](Ciphertext& self, uint64_t seed){ self.seed() = seed; })
        .def("contains_seed", &Ciphertext::contains_seed)
        .def("expand_seed", &Ciphertext::expand_seed)
        .def("is_transparent", &Ciphertext::is_transparent)
        .def("save", [](const Ciphertext& self, HeContextPointer context) {return save_he(self, context); })
        .def("load", [](Ciphertext& self, const py::bytes& str, HeContextPointer context) {return load_he<Ciphertext>(self, str, context); })
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context) {return load_new_he<Ciphertext>(str, context); })
        .def("serialized_size", [](const Ciphertext& self, HeContextPointer context) {return serialized_size_he(self, context); })
        .def("save_terms", [](const Ciphertext& self, HeContextPointer context, const py::array_t<size_t>& terms) {
            ostringstream ss; self.save_terms(ss, context, get_vector_from_buffer(terms)); 
            return py::bytes(ss.str());
        })
        .def("load_terms", [](Ciphertext& self, const py::bytes& str, HeContextPointer context, const py::array_t<size_t>& terms) {
            istringstream ss(str); self.load_terms(ss, context, get_vector_from_buffer(terms)); 
        })
        .def("load_terms_new", [](const py::bytes& str, HeContextPointer context, const py::array_t<size_t>& terms) {
            istringstream ss(str); Ciphertext c; c.load_terms(ss, context, get_vector_from_buffer(terms)); 
            return c;
        })
        .def("serialized_terms_size", [](const Ciphertext& self, HeContextPointer context, const py::array_t<size_t>& terms) {
            return self.serialized_terms_size(context, get_vector_from_buffer(terms));
        })
    ;

    py::class_<LWECiphertext>(m, "LWECiphertext")
        .def("clone", &LWECiphertext::clone)
        .def("on_device", &LWECiphertext::on_device)
        .def("assemble_lwe", &LWECiphertext::assemble_lwe)
    ;

    py::class_<SecretKey>(m, "SecretKey")
        .def(py::init<>())
        .def(py::init<const Plaintext&>())
        .def("clone", &SecretKey::clone)
        .def("to_device_inplace", &SecretKey::to_device_inplace)
        .def("to_host_inplace", &SecretKey::to_host_inplace)
        .def("to_device", &SecretKey::to_device)
        .def("to_host", &SecretKey::to_host)
        .def("on_device", &SecretKey::on_device)
        .def("parms_id", py::overload_cast<>(&SecretKey::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](SecretKey& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("save", [](const SecretKey& self) {return save(self); })
        .def("load", [](SecretKey& self, const py::bytes& str) {return load<SecretKey>(self, str); })
        .def_static("load_new", [](const py::bytes& str) {return load_new<SecretKey>(str); })
        .def("serialized_size", [](const SecretKey& self) {return serialized_size(self); })
        .def("as_plaintext", [](const SecretKey& self) {return self.as_plaintext(); }, py::return_value_policy::reference)
        .def("get_plaintext", [](const SecretKey& self) {return self.as_plaintext().clone(); })
    ;

    py::class_<PublicKey>(m, "PublicKey")
        .def(py::init<>())
        .def(py::init<const Ciphertext&>())
        .def("clone", &PublicKey::clone)
        .def("to_device_inplace", &PublicKey::to_device_inplace)
        .def("to_host_inplace", &PublicKey::to_host_inplace)
        .def("to_device", &PublicKey::to_device)
        .def("to_host", &PublicKey::to_host)
        .def("on_device", &PublicKey::on_device)
        .def("parms_id", py::overload_cast<>(&PublicKey::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](PublicKey& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("as_ciphertext", [](const PublicKey& self) {return self.as_ciphertext(); }, py::return_value_policy::reference)
        .def("get_ciphertext", [](const PublicKey& self) {return self.as_ciphertext().clone(); })
        .def("save", [](const PublicKey& self, HeContextPointer context) {return save_he(self, context); })
        .def("load", [](PublicKey& self, const py::bytes& str, HeContextPointer context) {return load_he<PublicKey>(self, str, context); })
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context) {return load_new_he<PublicKey>(str, context); })
        .def("serialized_size", [](const PublicKey& self, HeContextPointer context) {return serialized_size_he(self, context); })
        .def("contains_seed", &PublicKey::contains_seed)
        .def("expand_seed", &PublicKey::expand_seed)
    ;

    py::class_<KSwitchKeys>(m, "KSwitchKeys")
        .def(py::init<>())
        .def("clone", &KSwitchKeys::clone)
        .def("to_device_inplace", &KSwitchKeys::to_device_inplace)
        .def("to_host_inplace", &KSwitchKeys::to_host_inplace)
        .def("to_device", &KSwitchKeys::to_device)
        .def("to_host", &KSwitchKeys::to_host)
        .def("on_device", &KSwitchKeys::on_device)
        .def("parms_id", py::overload_cast<>(&KSwitchKeys::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](KSwitchKeys& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("save", [](const KSwitchKeys& self, HeContextPointer context) {return save_he(self, context); })
        .def("load", [](KSwitchKeys& self, const py::bytes& str, HeContextPointer context) {return load_he<KSwitchKeys>(self, str, context); })
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context) {return load_new_he<KSwitchKeys>(str, context); })
        .def("serialized_size", [](const KSwitchKeys& self, HeContextPointer context) {return serialized_size_he(self, context); })
    ;

    py::class_<RelinKeys>(m, "RelinKeys")
        .def(py::init<>())
        .def("clone", &RelinKeys::clone)
        .def("to_device_inplace", &RelinKeys::to_device_inplace)
        .def("to_host_inplace", &RelinKeys::to_host_inplace)
        .def("to_device", &RelinKeys::to_device)
        .def("to_host", &RelinKeys::to_host)
        .def("on_device", &RelinKeys::on_device)
        .def("parms_id", py::overload_cast<>(&RelinKeys::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](RelinKeys& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("save", [](const RelinKeys& self, HeContextPointer context) {return save_he(self, context); })
        .def("load", [](RelinKeys& self, const py::bytes& str, HeContextPointer context) {return load_he<RelinKeys>(self, str, context); })
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context) {return load_new_he<RelinKeys>(str, context); })
        .def("serialized_size", [](const RelinKeys& self, HeContextPointer context) {return serialized_size_he(self, context); })
        .def("as_kswitch_keys", [](const RelinKeys& self) {return self.as_kswitch_keys(); }, py::return_value_policy::reference)
        .def("get_kswitch_keys", [](const RelinKeys& self) {return self.as_kswitch_keys().clone(); })
    ;

    py::class_<GaloisKeys>(m, "GaloisKeys")
        .def(py::init<>())
        .def("clone", &GaloisKeys::clone)
        .def("to_device_inplace", &GaloisKeys::to_device_inplace)
        .def("to_host_inplace", &GaloisKeys::to_host_inplace)
        .def("to_device", &GaloisKeys::to_device)
        .def("to_host", &GaloisKeys::to_host)
        .def("on_device", &GaloisKeys::on_device)
        .def("parms_id", py::overload_cast<>(&GaloisKeys::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](GaloisKeys& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("save", [](const GaloisKeys& self, HeContextPointer context) {return save_he(self, context); })
        .def("load", [](GaloisKeys& self, const py::bytes& str, HeContextPointer context) {return load_he<GaloisKeys>(self, str, context); })
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context) {return load_new_he<GaloisKeys>(str, context); })
        .def("serialized_size", [](const GaloisKeys& self, HeContextPointer context) {return serialized_size_he(self, context); })
        .def("as_kswitch_keys", [](const GaloisKeys& self) {return self.as_kswitch_keys(); }, py::return_value_policy::reference)
        .def("get_kswitch_keys", [](const GaloisKeys& self) {return self.as_kswitch_keys().clone(); })
    ;

    py::class_<KeyGenerator>(m, "KeyGenerator")
        .def(py::init<HeContextPointer>())
        .def(py::init<HeContextPointer, const SecretKey&>())
        .def("context", &KeyGenerator::context)
        .def("on_device", &KeyGenerator::on_device)
        .def("to_device_inplace", &KeyGenerator::to_device_inplace)
        .def("secret_key", &KeyGenerator::secret_key, py::return_value_policy::reference)
        .def("create_public_key", &KeyGenerator::create_public_key)
        .def("create_keyswitching_key", &KeyGenerator::create_keyswitching_key)
        .def("create_relin_keys", &KeyGenerator::create_relin_keys, 
            py::arg("save_seed"), py::arg("max_power") = 2
        )
        .def("create_galois_keys_from_elements", [](const KeyGenerator& self, const py::array_t<uint64_t>& galois_elts, bool save_seed) {
            return self.create_galois_keys_from_elements(get_vector_from_buffer(galois_elts), save_seed);
        })
        .def("create_galois_keys_from_steps", [](const KeyGenerator& self, const py::array_t<int>& galois_steps, bool save_seed) {
            return self.create_galois_keys_from_steps(get_vector_from_buffer(galois_steps), save_seed);
        })
        .def("create_galois_keys", &KeyGenerator::create_galois_keys)
        .def("create_automorphism_keys", &KeyGenerator::create_automorphism_keys)
    ;

    py::class_<BatchEncoder>(m, "BatchEncoder")
        .def(py::init<HeContextPointer>())
        .def("context", &BatchEncoder::context)
        .def("on_device", &BatchEncoder::on_device)
        .def("to_device_inplace", &BatchEncoder::to_device_inplace)
        .def("slot_count", &BatchEncoder::slot_count)
        .def("row_count", &BatchEncoder::row_count)
        .def("column_count", &BatchEncoder::column_count)
        .def("simd_encoding_supported", &BatchEncoder::simd_encoding_supported)
        .def("encode_simd", [](const BatchEncoder& self, const py::array_t<uint64_t>& values, Plaintext& p) {
            self.encode(get_vector_from_buffer(values), p);
        })
        .def("encode_simd_new", [](const BatchEncoder& self, const py::array_t<uint64_t>& values) {
            return self.encode_new(get_vector_from_buffer(values));
        })
        .def("encode_polynomial", [](const BatchEncoder& self, const py::array_t<uint64_t>& values, Plaintext& p) {
            self.encode_polynomial(get_vector_from_buffer(values), p);
        })
        .def("encode_polynomial_new", [](const BatchEncoder& self, const py::array_t<uint64_t>& values) {
            return self.encode_polynomial_new(get_vector_from_buffer(values));
        })
        .def("decode_simd_new", [](const BatchEncoder& self, const Plaintext& p) {
            return get_buffer_from_vector(self.decode_new(p));
        })
        .def("decode_polynomial_new", [](const BatchEncoder& self, const Plaintext& p) {
            return get_buffer_from_vector(self.decode_polynomial_new(p));
        })
    ;

    py::class_<CKKSEncoder>(m, "CKKSEncoder")
        .def(py::init<HeContextPointer>())
        .def("context", &CKKSEncoder::context)
        .def("on_device", &CKKSEncoder::on_device)
        .def("to_device_inplace", &CKKSEncoder::to_device_inplace)
        .def("slot_count", &CKKSEncoder::slot_count)
        .def("polynomial_modulus_degree", &CKKSEncoder::polynomial_modulus_degree)
        .def("poly_modulus_degree", &CKKSEncoder::polynomial_modulus_degree)
        .def("encode_complex64_simd", [](
            const CKKSEncoder& self, const py::array_t<std::complex<double>>& values, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p) {
            self.encode_complex64_simd(get_vector_from_buffer(values), parms_id, scale, p);
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"), py::arg("p"))
        .def("encode_complex64_simd_new", [](
            const CKKSEncoder& self, const py::array_t<std::complex<double>>& values, 
            std::optional<ParmsID> parms_id, double scale) {
            return self.encode_complex64_simd_new(get_vector_from_buffer(values), parms_id, scale);
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"))
        .def("encode_float64_single", [](
            const CKKSEncoder& self, double value, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p) {
            self.encode_float64_single(value, parms_id, scale, p);
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"), py::arg("p"))
        .def("encode_float64_single_new", [](
            const CKKSEncoder& self, double value, 
            std::optional<ParmsID> parms_id, double scale) {
            return self.encode_float64_single_new(value, parms_id, scale);
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"))
        .def("encode_float64_polynomial", [](
            const CKKSEncoder& self, const py::array_t<double>& values, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p) {
            self.encode_float64_polynomial(get_vector_from_buffer(values), parms_id, scale, p);
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"), py::arg("p")) 
        .def("encode_float64_polynomial_new", [](
            const CKKSEncoder& self, const py::array_t<double>& values, 
            std::optional<ParmsID> parms_id, double scale) {
            return self.encode_float64_polynomial_new(get_vector_from_buffer(values), parms_id, scale);
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"))
        .def("encode_complex64_single", [](
            const CKKSEncoder& self, std::complex<double> value, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p) {
            self.encode_complex64_single(value, parms_id, scale, p);
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"), py::arg("p"))
        .def("encode_complex64_single_new", [](
            const CKKSEncoder& self, std::complex<double> value, 
            std::optional<ParmsID> parms_id, double scale) {
            return self.encode_complex64_single_new(value, parms_id, scale);
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"))
        .def("encode_integer64_single", [](
            const CKKSEncoder& self, int64_t value, 
            std::optional<ParmsID> parms_id, Plaintext& p) {
            self.encode_integer64_single(value, parms_id, p);
        }, py::arg("value"), py::arg("parms_id"), py::arg("p"))
        .def("encode_integer64_single_new", [](
            const CKKSEncoder& self, int64_t value, 
            std::optional<ParmsID> parms_id) {
            return self.encode_integer64_single_new(value, parms_id);
        }, py::arg("value"), py::arg("parms_id"))
        .def("encode_integer64_polynomial", [](
            const CKKSEncoder& self, const py::array_t<int64_t>& values, 
            std::optional<ParmsID> parms_id, Plaintext& p) {
            self.encode_integer64_polynomial(get_vector_from_buffer(values), parms_id, p);
        }, py::arg("values"), py::arg("parms_id"), py::arg("p"))
        .def("encode_integer64_polynomial_new", [](
            const CKKSEncoder& self, const py::array_t<int64_t>& values, 
            std::optional<ParmsID> parms_id) {
            return self.encode_integer64_polynomial_new(get_vector_from_buffer(values), parms_id);
        }, py::arg("values"), py::arg("parms_id"))
        .def("decode_complex64_simd_new", [](const CKKSEncoder& self, const Plaintext& p) {
            return get_buffer_from_vector(self.decode_complex64_simd_new(p));
        })
        .def("decode_float64_polynomial_new", [](const CKKSEncoder& self, const Plaintext& p) {
            return get_buffer_from_vector(self.decode_float64_polynomial_new(p));
        })
    ;

    py::class_<utils::RandomGenerator>(m, "RandomGenerator")
        .def(py::init<>())
        .def(py::init<uint64_t>())
        .def("reset_seed", &utils::RandomGenerator::reset_seed)
        .def("sample_uint64", &utils::RandomGenerator::sample_uint64)
    ;
        
    py::class_<Encryptor>(m, "Encryptor")
        .def(py::init<HeContextPointer>())
        .def("context", &Encryptor::context)
        .def("on_device", &Encryptor::on_device)
        .def("to_device_inplace", &Encryptor::to_device_inplace)
        .def("set_public_key", &Encryptor::set_public_key)
        .def("set_secret_key", &Encryptor::set_secret_key)
        .def("public_key", &Encryptor::public_key, py::return_value_policy::reference)
        .def("secret_key", &Encryptor::secret_key, py::return_value_policy::reference)

        .def("encrypt_asymmetric", [](const Encryptor& self, const Plaintext& plain, Ciphertext& destination, utils::RandomGenerator& rng) {
            self.encrypt_asymmetric(plain, destination, &rng);
        })
        .def("encrypt_asymmetric", [](const Encryptor& self, const Plaintext& plain, Ciphertext& destination) {
            self.encrypt_asymmetric(plain, destination, nullptr);
        })
        .def("encrypt_asymmetric_new", [](const Encryptor& self, const Plaintext& plain, utils::RandomGenerator& rng) {
            return self.encrypt_asymmetric_new(plain, &rng);
        })
        .def("encrypt_asymmetric_new", [](const Encryptor& self, const Plaintext& plain) {
            return self.encrypt_asymmetric_new(plain, nullptr);
        })

        .def("encrypt_zero_asymmetric", [](const Encryptor& self, Ciphertext& destination, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng) {
            self.encrypt_zero_asymmetric(destination, parms_id, &rng);
        })
        .def("encrypt_zero_asymmetric", [](const Encryptor& self, Ciphertext& destination, std::optional<ParmsID> parms_id) {
            self.encrypt_zero_asymmetric(destination, parms_id, nullptr);
        })
        .def("encrypt_zero_asymmetric_new", [](const Encryptor& self, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng) {
            return self.encrypt_zero_asymmetric_new(parms_id, &rng);
        })
        .def("encrypt_zero_asymmetric_new", [](const Encryptor& self, std::optional<ParmsID> parms_id) {
            return self.encrypt_zero_asymmetric_new(parms_id, nullptr);
        })

        .def("encrypt_symmetric", [](const Encryptor& self, const Plaintext& plain, bool save_seed, Ciphertext& destination, utils::RandomGenerator& rng) {
            self.encrypt_symmetric(plain, save_seed, destination, &rng);
        })
        .def("encrypt_symmetric", [](const Encryptor& self, const Plaintext& plain, bool save_seed, Ciphertext& destination) {
            self.encrypt_symmetric(plain, save_seed, destination, nullptr);
        })
        .def("encrypt_symmetric_new", [](const Encryptor& self, const Plaintext& plain, bool save_seed, utils::RandomGenerator& rng) {
            return self.encrypt_symmetric_new(plain, save_seed, &rng);
        })
        .def("encrypt_symmetric_new", [](const Encryptor& self, const Plaintext& plain, bool save_seed) {
            return self.encrypt_symmetric_new(plain, save_seed, nullptr);
        })

        .def("encrypt_zero_symmetric", [](const Encryptor& self, bool save_seed, Ciphertext& destination, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng) {
            self.encrypt_zero_symmetric(save_seed, destination, parms_id, &rng);
        })
        .def("encrypt_zero_symmetric", [](const Encryptor& self, bool save_seed, Ciphertext& destination, std::optional<ParmsID> parms_id) {
            self.encrypt_zero_symmetric(save_seed, destination, parms_id, nullptr);
        })
        .def("encrypt_zero_symmetric_new", [](const Encryptor& self, bool save_seed, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng) {
            return self.encrypt_zero_symmetric_new(save_seed, parms_id, &rng);
        })
        .def("encrypt_zero_symmetric_new", [](const Encryptor& self, bool save_seed, std::optional<ParmsID> parms_id) {
            return self.encrypt_zero_symmetric_new(save_seed, parms_id, nullptr);
        })
    ;

    py::class_<Decryptor>(m, "Decryptor")
        .def(py::init<HeContextPointer, const SecretKey&>())
        .def("context", &Decryptor::context)
        .def("on_device", &Decryptor::on_device)
        .def("to_device_inplace", &Decryptor::to_device_inplace)
        .def("decrypt", &Decryptor::decrypt)
        .def("decrypt_new", &Decryptor::decrypt_new)
    ;

    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<HeContextPointer>())
        .def("context", &Evaluator::context)
        .def("on_device", &Evaluator::on_device)
        .def("negate_inplace", &Evaluator::negate_inplace)
        .def("negate", &Evaluator::negate)
        .def("negate_new", &Evaluator::negate_new)
        .def("add_inplace", &Evaluator::add_inplace)
        .def("add", &Evaluator::add)
        .def("add_new", &Evaluator::add_new)
        .def("sub_inplace", &Evaluator::sub_inplace)
        .def("sub", &Evaluator::sub)
        .def("sub_new", &Evaluator::sub_new)
        .def("multiply_inplace", &Evaluator::multiply_inplace)
        .def("multiply", &Evaluator::multiply)
        .def("multiply_new", &Evaluator::multiply_new)
        .def("square_inplace", &Evaluator::square_inplace)
        .def("square", &Evaluator::square)
        .def("square_new", &Evaluator::square_new)
        .def("apply_keyswitching_inplace", &Evaluator::apply_keyswitching_inplace)
        .def("apply_keyswitching", &Evaluator::apply_keyswitching)
        .def("apply_keyswitching_new", &Evaluator::apply_keyswitching_new)
        .def("relinearize_inplace", &Evaluator::relinearize_inplace)
        .def("relinearize", &Evaluator::relinearize)
        .def("relinearize_new", &Evaluator::relinearize_new)
        .def("mod_switch_to_next", &Evaluator::mod_switch_to_next)
        .def("mod_switch_to_next_inplace", &Evaluator::mod_switch_to_next_inplace)
        .def("mod_switch_to_next_new", &Evaluator::mod_switch_to_next_new)
        .def("mod_switch_plain_to_next_inplace", &Evaluator::mod_switch_plain_to_next_inplace)
        .def("mod_switch_plain_to_next", &Evaluator::mod_switch_plain_to_next)
        .def("mod_switch_plain_to_next_new", &Evaluator::mod_switch_plain_to_next_new)
        .def("mod_switch_to_inplace", &Evaluator::mod_switch_to_inplace)
        .def("mod_switch_to", &Evaluator::mod_switch_to)
        .def("mod_switch_to_new", &Evaluator::mod_switch_to_new)
        .def("mod_switch_plain_to_inplace", &Evaluator::mod_switch_plain_to_inplace)
        .def("mod_switch_plain_to", &Evaluator::mod_switch_plain_to)
        .def("mod_switch_plain_to_new", &Evaluator::mod_switch_plain_to_new)
        .def("rescale_to_next", &Evaluator::rescale_to_next)
        .def("rescale_to_next_inplace", &Evaluator::rescale_to_next_inplace)
        .def("rescale_to_next_new", &Evaluator::rescale_to_next_new)
        .def("rescale_to", &Evaluator::rescale_to)
        .def("rescale_to_inplace", &Evaluator::rescale_to_inplace)
        .def("rescale_to_new", &Evaluator::rescale_to_new)
        .def("add_plain_inplace", &Evaluator::add_plain_inplace)
        .def("add_plain", &Evaluator::add_plain)
        .def("add_plain_new", &Evaluator::add_plain_new)
        .def("sub_plain_inplace", &Evaluator::sub_plain_inplace)
        .def("sub_plain", &Evaluator::sub_plain)
        .def("sub_plain_new", &Evaluator::sub_plain_new)
        .def("multiply_plain_inplace", &Evaluator::multiply_plain_inplace)
        .def("multiply_plain", &Evaluator::multiply_plain)
        .def("multiply_plain_new", &Evaluator::multiply_plain_new)
        .def("transform_plain_to_ntt_inplace", &Evaluator::transform_plain_to_ntt_inplace)
        .def("transform_plain_to_ntt", &Evaluator::transform_plain_to_ntt)
        .def("transform_plain_to_ntt_new", &Evaluator::transform_plain_to_ntt_new)
        .def("transform_to_ntt_inplace", &Evaluator::transform_to_ntt_inplace)
        .def("transform_to_ntt", &Evaluator::transform_to_ntt)
        .def("transform_to_ntt_new", &Evaluator::transform_to_ntt_new)
        .def("transform_from_ntt_inplace", &Evaluator::transform_from_ntt_inplace)
        .def("transform_from_ntt", &Evaluator::transform_from_ntt)
        .def("transform_from_ntt_new", &Evaluator::transform_from_ntt_new)
        .def("apply_galois_inplace", &Evaluator::apply_galois_inplace)
        .def("apply_galois", &Evaluator::apply_galois)
        .def("apply_galois_new", &Evaluator::apply_galois_new)
        .def("apply_galois_plain_inplace", &Evaluator::apply_galois_plain_inplace)
        .def("apply_galois_plain", &Evaluator::apply_galois_plain)
        .def("apply_galois_plain_new", &Evaluator::apply_galois_plain_new)
        .def("rotate_rows_inplace", &Evaluator::rotate_rows_inplace)
        .def("rotate_rows", &Evaluator::rotate_rows)
        .def("rotate_rows_new", &Evaluator::rotate_rows_new)
        .def("rotate_columns_inplace", &Evaluator::rotate_columns_inplace)
        .def("rotate_columns", &Evaluator::rotate_columns)
        .def("rotate_columns_new", &Evaluator::rotate_columns_new)
        .def("rotate_vector_inplace", &Evaluator::rotate_vector_inplace)
        .def("rotate_vector", &Evaluator::rotate_vector)
        .def("rotate_vector_new", &Evaluator::rotate_vector_new)
        .def("complex_conjugate_inplace", &Evaluator::complex_conjugate_inplace)
        .def("complex_conjugate", &Evaluator::complex_conjugate)
        .def("complex_conjugate_new", &Evaluator::complex_conjugate_new)
        .def("extract_lwe_new", &Evaluator::extract_lwe_new)
        .def("assemble_lwe_new", &Evaluator::assemble_lwe_new)
        .def("field_trace_inplace", &Evaluator::field_trace_inplace)
        .def("divide_by_poly_modulus_degree_inplace", &Evaluator::divide_by_poly_modulus_degree_inplace)
        .def("pack_lwe_ciphertexts_new", &Evaluator::pack_lwe_ciphertexts_new)
        .def("negacyclic_shift", &Evaluator::negacyclic_shift)
        .def("negacyclic_shift_new", &Evaluator::negacyclic_shift_new)
        .def("negacyclic_shift_inplace", &Evaluator::negacyclic_shift_inplace)
    ;

    // linear

    py::class_<Plain2d>(m, "Plain2d")
        .def(py::init<>())
        .def("size", &Plain2d::size)
        .def("rows", &Plain2d::rows)
        .def("columns", &Plain2d::columns)
        .def("encrypt_asymmetric", &Plain2d::encrypt_asymmetric)
        .def("encrypt_symmetric", &Plain2d::encrypt_symmetric)
    ;

    py::class_<Cipher2d>(m, "Cipher2d")
        .def(py::init<>())
        .def("size", &Cipher2d::size)
        .def("rows", &Cipher2d::rows)
        .def("columns", &Cipher2d::columns)
        .def("expand_seed", &Cipher2d::expand_seed)
        .def("save", [](const Cipher2d& self, HeContextPointer context) {return save_he(self, context); })
        .def("load", [](Cipher2d& self, const py::bytes& str, HeContextPointer context) {return load_he<Cipher2d>(self, str, context); })
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context) {return load_new_he<Cipher2d>(str, context); })
        .def("serialized_size", [](const Cipher2d& self, HeContextPointer context) {return serialized_size_he(self, context); })
        .def("mod_switch_to_next_inplace", &Cipher2d::mod_switch_to_next_inplace)
        .def("mod_switch_to_next", &Cipher2d::mod_switch_to_next)
        .def("relinearize_inplace", &Cipher2d::relinearize_inplace)
        .def("relinearize", &Cipher2d::relinearize)
        .def("add", &Cipher2d::add)
        .def("add_inplace", &Cipher2d::add_inplace)
        .def("add_plain", &Cipher2d::add_plain)
        .def("add_plain_inplace", &Cipher2d::add_plain_inplace)
        .def("sub", &Cipher2d::sub)
        .def("sub_inplace", &Cipher2d::sub_inplace)
        .def("sub_plain", &Cipher2d::sub_plain)
        .def("sub_plain_inplace", &Cipher2d::sub_plain_inplace)
    ;

    py::enum_<MatmulObjective>(m, "MatmulObjective")
        .value("EncryptLeft", MatmulObjective::EncryptLeft)
        .value("EncryptRight", MatmulObjective::EncryptRight)
        .value("Crossed", MatmulObjective::Crossed)
    ;

    py::class_<MatmulHelper>(m, "MatmulHelper")
        .def(py::init<size_t, size_t, size_t, size_t, MatmulObjective, bool>(), 
            py::arg("batch_size"), py::arg("input_dims"), py::arg("output_dims"),
            py::arg("slot_count"), py::arg_v("objective", MatmulObjective::EncryptLeft, "MatmulObjective.EncryptLeft"),
            py::arg("pack_lwe") = true
        )
        .def("batch_size", [](const MatmulHelper& self) { return self.batch_size; })
        .def("input_dims", [](const MatmulHelper& self) { return self.input_dims; })
        .def("output_dims", [](const MatmulHelper& self) { return self.output_dims; })
        .def("slot_count", [](const MatmulHelper& self) { return self.slot_count; })
        .def("objective", [](const MatmulHelper& self) { return self.objective; })
        .def("pack_lwe", [](const MatmulHelper& self) { return self.pack_lwe; })
        .def("batch_block", [](const MatmulHelper& self) { return self.batch_block; })
        .def("input_block", [](const MatmulHelper& self) { return self.input_block; })
        .def("output_block", [](const MatmulHelper& self) { return self.output_block; })
        .def("encode_weights", [](const MatmulHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& weights) {
            if (weights.ndim() != 1) 
                throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be flattened.");
            if (weights.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be contiguous.");
            if (weights.size() != self.input_dims * self.output_dims) 
                throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be of size input_dims * output_dims.");
            return self.encode_weights(encoder, get_pointer_from_buffer(weights));
        })
        .def("encode_inputs", [](const MatmulHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& inputs) {
            if (inputs.ndim() != 1) 
                throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be flattened.");
            if (inputs.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be contiguous.");
            if (inputs.size() != self.batch_size * self.input_dims)
                throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be of size batch_size * input_dims.");
            return self.encode_inputs(encoder, get_pointer_from_buffer(inputs));
        })
        .def("encode_outputs", [](const MatmulHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& outputs) {
            if (outputs.ndim() != 1) 
                throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be flattened.");
            if (outputs.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be contiguous.");
            if (outputs.size() != self.batch_size * self.output_dims)
                throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be of size batch_size * output_dims.");
            return self.encode_outputs(encoder, get_pointer_from_buffer(outputs));
        })
        .def("matmul", &MatmulHelper::matmul)
        .def("matmul_reverse", &MatmulHelper::matmul_reverse)
        .def("matmul_cipher", &MatmulHelper::matmul_cipher)
        .def("decrypt_outputs", [](const MatmulHelper& self, const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) {
            std::vector<uint64_t> result = self.decrypt_outputs(encoder, decryptor, outputs);
            return get_buffer_from_vector(result);
        })
        .def("pack_outputs", &MatmulHelper::pack_outputs)
        .def("serialize_outputs", [](const MatmulHelper& self, const Evaluator &evaluator, const Cipher2d& x) {
            ostringstream ss; self.serialize_outputs(evaluator, x, ss); return py::bytes(ss.str());
        })
        .def("deserialize_outputs", [](const MatmulHelper& self, const Evaluator &evaluator, const py::bytes& str) {
            istringstream ss(str); return self.deserialize_outputs(evaluator, ss);
        })
    ;
    

    py::class_<Conv2dHelper>(m, "Conv2dHelper")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, MatmulObjective>(), 
            py::arg("batch_size"), 
            py::arg("input_channels"), py::arg("output_channels"),
            py::arg("image_height"), py::arg("image_width"),
            py::arg("kernel_height"), py::arg("kernel_width"),
            py::arg("slot_count"),
            py::arg_v("objective", MatmulObjective::EncryptLeft, "MatmulObjective.EncryptLeft")
        )
        .def("batch_size", [](const Conv2dHelper& self) { return self.batch_size; })
        .def("input_channels", [](const Conv2dHelper& self) { return self.input_channels; })
        .def("output_channels", [](const Conv2dHelper& self) { return self.output_channels; })
        .def("image_height", [](const Conv2dHelper& self) { return self.image_height; })
        .def("image_width", [](const Conv2dHelper& self) { return self.image_width; })
        .def("kernel_height", [](const Conv2dHelper& self) { return self.kernel_height; })
        .def("kernel_width", [](const Conv2dHelper& self) { return self.kernel_width; })
        .def("slot_count", [](const Conv2dHelper& self) { return self.slot_count; })
        .def("objective", [](const Conv2dHelper& self) { return self.objective; })
        .def("batch_block", [](const Conv2dHelper& self) { return self.batch_block; })
        .def("input_channel_block", [](const Conv2dHelper& self) { return self.input_channel_block; })
        .def("output_channel_block", [](const Conv2dHelper& self) { return self.output_channel_block; })
        .def("image_height_block", [](const Conv2dHelper& self) { return self.image_height_block; })
        .def("image_width_block", [](const Conv2dHelper& self) { return self.image_width_block; })
        .def("encode_weights", [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& weights) {
            if (weights.ndim() != 1) 
                throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be flattened.");
            if (weights.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be contiguous.");
            if (weights.size() != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
                throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be of size oc * ic * kh * kw.");
            return self.encode_weights(encoder, get_pointer_from_buffer(weights));
        })
        .def("encode_inputs", [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& inputs) {
            if (inputs.ndim() != 1) 
                throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be flattened.");
            if (inputs.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be contiguous.");
            if (inputs.size() != self.batch_size * self.input_channels * self.image_height * self.image_width)
                throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be of size bs * ic * ih * iw.");
            return self.encode_inputs(encoder, get_pointer_from_buffer(inputs));
        })
        .def("encode_outputs", [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& outputs) {
            if (outputs.ndim() != 1) 
                throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be flattened.");
            if (outputs.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be contiguous.");
            size_t output_height = self.image_height - self.kernel_height + 1;
            size_t output_width = self.image_width - self.kernel_width + 1;
            if (outputs.size() != self.batch_size * self.output_channels * output_height * output_width)
                throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be of size bs * oc * oh * ow.");
            return self.encode_outputs(encoder, get_pointer_from_buffer(outputs));
        })
        .def("conv2d", &Conv2dHelper::conv2d)
        .def("conv2d_reverse", &Conv2dHelper::conv2d_reverse)
        .def("conv2d_cipher", &Conv2dHelper::conv2d_cipher)
        .def("decrypt_outputs", [](const Conv2dHelper& self, const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) {
            std::vector<uint64_t> result = self.decrypt_outputs(encoder, decryptor, outputs);
            return get_buffer_from_vector(result);
        })
        .def("serialize_outputs", [](const Conv2dHelper& self, const Evaluator &evaluator, const Cipher2d& x) {
            ostringstream ss; self.serialize_outputs(evaluator, x, ss); return py::bytes(ss.str());
        })
        .def("deserialize_outputs", [](const Conv2dHelper& self, const Evaluator &evaluator, const py::bytes& str) {
            istringstream ss(str); return self.deserialize_outputs(evaluator, ss);
        })
    ;
    
}