#include "header.h"

void register_lwe_ciphertext(pybind11::module& m) {

    py::class_<LWECiphertext>(m, "LWECiphertext")
        .def("pool", &LWECiphertext::pool)
        .def("address", [](const LWECiphertext& self){
            return reinterpret_cast<uintptr_t>(&self);
        })
        .def("device_index", &LWECiphertext::device_index)
        .def("clone", [](const LWECiphertext& self, MemoryPoolHandleArgument pool){
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("on_device", &LWECiphertext::on_device)
        .def("assemble_lwe", &LWECiphertext::assemble_lwe)
    ;
    
}