#include "header.h"

void register_ciphertext(pybind11::module& m) {
    
    py::class_<Ciphertext>(m, "Ciphertext")
        .def(py::init<>())
        .def("address", [](const Ciphertext& self){
            return reinterpret_cast<uintptr_t>(&self);
        })
        .def("data_address", [](const Ciphertext& self){
            return reinterpret_cast<uintptr_t>(self.data().raw_pointer());
        })
        .def("pool", &Ciphertext::pool)
        .def("device_index", &Ciphertext::device_index)
        .def("obtain_data", [](const Ciphertext& self){
            troy::utils::DynamicArray<uint64_t> data = self.data().to_host();
            return get_buffer_from_slice(data.const_reference());
        })
        .def("clone", [](const Ciphertext& self, MemoryPoolHandleArgument pool){
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_device_inplace", [](Ciphertext& self, MemoryPoolHandleArgument pool){
            return self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
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

        .def("save", [](const Ciphertext& self, HeContextPointer context, CompressionMode mode) {return save_he(self, context, mode); },
            py::arg("context"), COMPRESSION_MODE_ARGUMENT)
        .def("load", [](Ciphertext& self, const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_he<Ciphertext>(self, str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_new_he<Ciphertext>(str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def("serialized_size_upperbound", [](const Ciphertext& self, HeContextPointer context, CompressionMode mode) {return serialized_size_upperbound_he(self, context, mode); },
            py::arg("context") = nullptr, COMPRESSION_MODE_ARGUMENT)
        .def("save_terms", [](const Ciphertext& self, HeContextPointer context, const py::array_t<size_t>& terms, MemoryPoolHandleArgument pool, CompressionMode mode) {
            ostringstream ss; self.save_terms(ss, context, get_vector_from_buffer(terms), nullopt_default_pool(pool), mode); 
            return py::bytes(ss.str());
        }, py::arg("context"), py::arg("terms"), MEMORY_POOL_ARGUMENT, COMPRESSION_MODE_ARGUMENT)
        .def("load_terms", [](Ciphertext& self, const py::bytes& str, HeContextPointer context, const py::array_t<size_t>& terms, MemoryPoolHandleArgument pool) {
            istringstream ss(str); self.load_terms(ss, context, get_vector_from_buffer(terms), nullopt_default_pool(pool)); 
        }, py::arg("str"), py::arg("context"), py::arg("terms"), MEMORY_POOL_ARGUMENT)
        .def_static("load_terms_new", [](const py::bytes& str, HeContextPointer context, const py::array_t<size_t>& terms, MemoryPoolHandleArgument pool) {
            istringstream ss(str); Ciphertext c; c.load_terms(ss, context, get_vector_from_buffer(terms), nullopt_default_pool(pool)); 
            return c;
        }, py::arg("str"), py::arg("context"), py::arg("terms"), MEMORY_POOL_ARGUMENT)
        .def("serialized_terms_size_upperbound", [](const Ciphertext& self, HeContextPointer context, const py::array_t<size_t>& terms, CompressionMode mode) {
            return self.serialized_terms_size_upperbound(context, get_vector_from_buffer(terms), mode);
        }, py::arg("context"), py::arg("terms"), COMPRESSION_MODE_ARGUMENT)
    ;
    
}