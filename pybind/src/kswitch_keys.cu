#include "header.h"

template <typename K, bool as_keyswitch_keys>
static void register_keyswitch_key_derived(pybind11::module& m, const char* name) {
    
    auto cl = py::class_<K>(m, name)
        .def(py::init<>())
        .def("pool", &K::pool)
        .def("device_index", &K::device_index)
        .def("clone", [](const K& self, MemoryPoolHandleArgument pool) {
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_device_inplace", [](K& self, MemoryPoolHandleArgument pool) {
            return self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host_inplace", &K::to_host_inplace)
        .def("to_device", [](const K& self, MemoryPoolHandleArgument pool) {
            return self.to_device(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("to_host", &K::to_host)
        .def("on_device", &K::on_device)
        .def("parms_id", py::overload_cast<>(&K::parms_id, py::const_), py::return_value_policy::reference)
        .def("set_parms_id", [](K& self, const ParmsID& parms_id){ self.parms_id() = parms_id; })
        
        .def("save", [](const K& self, HeContextPointer context, CompressionMode mode) {return save_he(self, context, mode); },
            py::arg("context"), COMPRESSION_MODE_ARGUMENT)
        .def("load", [](K& self, const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_he<K>(self, str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_new_he<K>(str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def("serialized_size_upperbound", [](const K& self, HeContextPointer context, CompressionMode mode) {return serialized_size_upperbound_he(self, context, mode); },
            py::arg("context") = nullptr, COMPRESSION_MODE_ARGUMENT)
            
    ;

    if constexpr (as_keyswitch_keys) { 
        cl
            .def("as_kswitch_keys", [](const K& self) {return self.as_kswitch_keys(); }, py::return_value_policy::reference)
            .def("get_kswitch_keys", [](const K& self, MemoryPoolHandleArgument pool) {
                return self.as_kswitch_keys().clone(nullopt_default_pool(pool));
            }, MEMORY_POOL_ARGUMENT)
        ;
    }

}

void register_kswitch_keys(pybind11::module& m) {
    register_keyswitch_key_derived<KSwitchKeys, false>(m, "KSwitchKeys");
    register_keyswitch_key_derived<RelinKeys, true>(m, "RelinKeys");
    register_keyswitch_key_derived<GaloisKeys, true>(m, "GaloisKeys");
}