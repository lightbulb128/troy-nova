#include "header.h"

void register_he_context(pybind11::module& m) {

    py::class_<ContextData, std::shared_ptr<ContextData>>(m, "ContextData")
        .def("pool", &ContextData::pool)
        .def("device_index", &ContextData::device_index)
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
        .def("pool", &HeContext::pool)
        .def("device_index", &HeContext::device_index)
        .def("to_device_inplace", [](const HeContextPointer& self, MemoryPoolHandleArgument pool){
            return self->to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
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
    
}