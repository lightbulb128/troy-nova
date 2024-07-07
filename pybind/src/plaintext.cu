#include "header.h"

void register_plaintext(pybind11::module& m) {

    py::class_<Plaintext>(m, "Plaintext")
        .def(py::init<>())
        .def("pool", &Plaintext::pool)
        .def("device_index", &Plaintext::device_index)
        .def("clone", [](const Plaintext& self, MemoryPoolHandleArgument pool){
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_device_inplace", [](Plaintext& self, MemoryPoolHandleArgument pool){
            return self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host_inplace", &Plaintext::to_host_inplace)
        .def("to_device", [](const Plaintext& self, MemoryPoolHandleArgument pool){
            return self.to_device(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host", &Plaintext::to_host)
        .def("on_device", &Plaintext::on_device)
        .def("parms_id", py::overload_cast<>(&Plaintext::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](Plaintext& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("scale", py::overload_cast<>(&Plaintext::scale, py::const_))
        .def("set_scale", [](Plaintext& self, double scale){ self.scale() = scale; })
        .def("coeff_count", py::overload_cast<>(&Plaintext::coeff_count, py::const_))
        .def("set_coeff_count", [](Plaintext& self, size_t coeff_count){ self.coeff_count() = coeff_count; })
        .def("resize", &Plaintext::resize)
        .def("is_ntt_form", py::overload_cast<>(&Plaintext::is_ntt_form, py::const_))
        .def("set_is_ntt_form", [](Plaintext& self, bool is_ntt_form){ self.is_ntt_form() = is_ntt_form; })
        .def("save", [](const Plaintext& self) {return save(self); })
        .def("load", [](Plaintext& self, const py::bytes& str, MemoryPoolHandleArgument pool) {return load<Plaintext>(self, str, nullopt_default_pool(pool)); },
            py::arg("str"), MEMORY_POOL_ARGUMENT)
        .def_static("load_new", [](const py::bytes& str, MemoryPoolHandleArgument pool) {return load_new<Plaintext>(str, nullopt_default_pool(pool)); },
            py::arg("str"), MEMORY_POOL_ARGUMENT)
        .def("serialized_size", [](const Plaintext& self) {return serialized_size(self); })
    ;

}