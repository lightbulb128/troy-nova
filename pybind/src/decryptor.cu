#include "header.h"

void register_decryptor(pybind11::module& m) {
    
    py::class_<Decryptor>(m, "Decryptor")
        .def(py::init([](HeContextPointer context, const SecretKey& secret_key, MemoryPoolHandleArgument pool) {
            return Decryptor(context, secret_key, nullopt_default_pool(pool));
        }), py::arg("context"), py::arg("secret_key"), MEMORY_POOL_ARGUMENT)
        .def("context", &Decryptor::context)
        .def("on_device", &Decryptor::on_device)
        .def("to_device_inplace", [](Decryptor& self, MemoryPoolHandleArgument pool) {
            self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)

        // decrypt(encrypted, destination, pool)
        .def("decrypt", [](const Decryptor& self, const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandleArgument pool) {
            self.decrypt(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)

        // decrypt_new(encrypted, pool)
        .def("decrypt_new", [](const Decryptor& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.decrypt_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)

        // bfv_decrypt_without_scaling_down(encrypted, destination, pool)
        .def("bfv_decrypt_without_scaling_down", [](const Decryptor& self, const Ciphertext& encrypted, Plaintext& destination, MemoryPoolHandleArgument pool) {
            self.bfv_decrypt_without_scaling_down(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)

        // bfv_decrypt_without_scaling_down_new(encrypted, pool)
        .def("bfv_decrypt_without_scaling_down_new", [](const Decryptor& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.bfv_decrypt_without_scaling_down_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)
    ;

}