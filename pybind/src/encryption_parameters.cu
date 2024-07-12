#include "header.h"

void register_encryption_parameters(pybind11::module& m) {

    py::class_<ParmsID>(m, "ParmsID")
        .def_static("zero", [](){
            return parms_id_zero;
        })
        .def("to_vector", [](const ParmsID& p){
            vector<uint64_t> ret; ret.reserve(utils::HashFunction::hash_block_uint64_count);
            for (size_t i = 0; i < utils::HashFunction::hash_block_uint64_count; i++) {
                ret.push_back(p[i]);
            }
            return get_buffer_from_vector(ret);
        })
    ;

    py::class_<EncryptionParameters>(m, "EncryptionParameters")
        .def(py::init<SchemeType>())
        .def("pool", &EncryptionParameters::pool)
        .def("device_index", &EncryptionParameters::device_index)
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
    
        .def("save", [](const EncryptionParameters& self) {
            std::ostringstream ss; self.save(ss); return py::bytes(ss.str());
        })
        .def("load", [](EncryptionParameters& self, const py::bytes& str) {
            std::istringstream ss(str); self.load(ss);
        })
        .def_static("load_new", [](const py::bytes& str) {
            std::istringstream ss(str); EncryptionParameters ret; ret.load(ss); return ret;
        })
        .def("serialized_size_upperbound", &EncryptionParameters::serialized_size_upperbound)
    ;
    
}