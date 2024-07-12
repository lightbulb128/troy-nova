#include "header.h"

void register_public_key(pybind11::module& m) {
    
    py::class_<PublicKey>(m, "PublicKey")
        .def(py::init<>())
        .def(py::init<const Ciphertext&>())
        .def("pool", &PublicKey::pool)
        .def("device_index", &PublicKey::device_index)
        .def("clone", [](const PublicKey& self, MemoryPoolHandleArgument pool) {
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_device_inplace", [](PublicKey& self, MemoryPoolHandleArgument pool) {
            return self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host_inplace", &PublicKey::to_host_inplace)
        .def("to_device", [](const PublicKey& self, MemoryPoolHandleArgument pool) {
            return self.to_device(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host", &PublicKey::to_host)
        .def("on_device", &PublicKey::on_device)
        .def("parms_id", py::overload_cast<>(&PublicKey::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](PublicKey& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        .def("as_ciphertext", [](const PublicKey& self) {return self.as_ciphertext(); }, py::return_value_policy::reference)
        .def("get_ciphertext", [](const PublicKey& self, MemoryPoolHandleArgument pool) {
            return self.as_ciphertext().clone(nullopt_default_pool(pool)); 
        }, MEMORY_POOL_ARGUMENT)

        .def("save", [](const PublicKey& self, HeContextPointer context, CompressionMode mode) {return save_he(self, context, mode); },
            py::arg("context"), COMPRESSION_MODE_ARGUMENT)
        .def("load", [](PublicKey& self, const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_he<PublicKey>(self, str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_new_he<PublicKey>(str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def("serialized_size_upperbound", [](const PublicKey& self, HeContextPointer context, CompressionMode mode) {return serialized_size_upperbound_he(self, context, mode); },
            py::arg("context") = nullptr, COMPRESSION_MODE_ARGUMENT)

        .def("contains_seed", &PublicKey::contains_seed)
        .def("expand_seed", &PublicKey::expand_seed)
    ;

}