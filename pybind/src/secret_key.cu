#include "header.cuh"

void register_secret_key(pybind11::module& m) {
    
    py::class_<SecretKey>(m, "SecretKey")
        .def(py::init<>())
        .def(py::init<const Plaintext&>())
        .def("pool", &SecretKey::pool)
        .def("device_index", &SecretKey::device_index)
        .def("clone", [](const SecretKey& self, MemoryPoolHandleArgument pool){
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_device_inplace", [](SecretKey& self, MemoryPoolHandleArgument pool){
            return self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host_inplace", &SecretKey::to_host_inplace)
        .def("to_device", [](const SecretKey& self, MemoryPoolHandleArgument pool){
            return self.to_device(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host", &SecretKey::to_host)
        .def("on_device", &SecretKey::on_device)
        .def("parms_id", py::overload_cast<>(&SecretKey::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](SecretKey& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("save", [](const SecretKey& self) {return save(self); })
        .def("load", [](SecretKey& self, const py::bytes& str, MemoryPoolHandleArgument pool) {
            return load<SecretKey>(self, str, nullopt_default_pool(pool)); 
        }, py::arg("str"), MEMORY_POOL_ARGUMENT)
        .def_static("load_new", [](const py::bytes& str, MemoryPoolHandleArgument pool) {
            return load_new<SecretKey>(str, nullopt_default_pool(pool)); 
        }, py::arg("str"), MEMORY_POOL_ARGUMENT)
        .def("serialized_size", [](const SecretKey& self) {return serialized_size(self); })
        .def("as_plaintext", [](const SecretKey& self) {return self.as_plaintext(); }, py::return_value_policy::reference)
        .def("get_plaintext", [](const SecretKey& self, MemoryPoolHandleArgument pool) {return self.as_plaintext().clone(nullopt_default_pool(pool)); },
            MEMORY_POOL_ARGUMENT)
    ;

}