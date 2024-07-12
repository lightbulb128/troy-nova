#include "header.h"

void register_key_generator(pybind11::module& m) {
    
    py::class_<KeyGenerator>(m, "KeyGenerator")
        .def(py::init([](HeContextPointer context, MemoryPoolHandleArgument pool){
            return KeyGenerator(context, nullopt_default_pool(pool));
        }), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def(py::init([](HeContextPointer context, const SecretKey& secret_key, MemoryPoolHandleArgument pool){
            return KeyGenerator(context, secret_key, nullopt_default_pool(pool));
        }), py::arg("context"), py::arg("secret_key"), MEMORY_POOL_ARGUMENT)
        .def("context", &KeyGenerator::context)
        .def("on_device", &KeyGenerator::on_device)
        .def("to_device_inplace", [](KeyGenerator& self, MemoryPoolHandleArgument pool){
            return self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("secret_key", &KeyGenerator::secret_key, py::return_value_policy::reference)
        .def("create_public_key", [](KeyGenerator& self, bool save_seed, MemoryPoolHandleArgument pool) {
            return self.create_public_key(save_seed, nullopt_default_pool(pool));
        }, py::arg("save_seed"), MEMORY_POOL_ARGUMENT)
        .def("create_keyswitching_key", [](KeyGenerator& self, const SecretKey& secret_key, bool save_seed, MemoryPoolHandleArgument pool) {
            return self.create_keyswitching_key(secret_key, save_seed, nullopt_default_pool(pool));
        }, py::arg("secret_key"), py::arg("save_seed"), MEMORY_POOL_ARGUMENT)
        .def("create_relin_keys", [](KeyGenerator& self, bool save_seed, size_t max_power, MemoryPoolHandleArgument pool) {
            return self.create_relin_keys(save_seed, max_power, nullopt_default_pool(pool));
        }, py::arg("save_seed"), py::arg("max_power") = 2, MEMORY_POOL_ARGUMENT)
        .def("create_galois_keys_from_elements", [](const KeyGenerator& self, const py::array_t<uint64_t>& galois_elts, bool save_seed, MemoryPoolHandleArgument pool) {
            return self.create_galois_keys_from_elements(get_vector_from_buffer(galois_elts), save_seed, nullopt_default_pool(pool));
        }, py::arg("galois_elts"), py::arg("save_seed"), MEMORY_POOL_ARGUMENT)
        .def("create_galois_keys_from_steps", [](const KeyGenerator& self, const py::array_t<int>& galois_steps, bool save_seed, MemoryPoolHandleArgument pool) {
            return self.create_galois_keys_from_steps(get_vector_from_buffer(galois_steps), save_seed, nullopt_default_pool(pool));
        }, py::arg("galois_steps"), py::arg("save_seed"), MEMORY_POOL_ARGUMENT)
        .def("create_galois_keys", [](const KeyGenerator& self, bool save_seed, MemoryPoolHandleArgument pool) {
            return self.create_galois_keys(save_seed, nullopt_default_pool(pool));
        }, py::arg("save_seed"), MEMORY_POOL_ARGUMENT)
        .def("create_automorphism_keys", [](const KeyGenerator& self, bool save_seed, MemoryPoolHandleArgument pool) {
            return self.create_automorphism_keys(save_seed, nullopt_default_pool(pool));
        }, py::arg("save_seed"), MEMORY_POOL_ARGUMENT)
    ;

}