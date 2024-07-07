#include "header.h"

void register_encryptor(pybind11::module& m) {

    py::class_<Encryptor>(m, "Encryptor")
        .def(py::init<HeContextPointer>())
        .def("context", &Encryptor::context)
        .def("on_device", &Encryptor::on_device)

        .def("to_device_inplace", [](Encryptor& self, MemoryPoolHandleArgument pool) {
            self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)

        .def("set_public_key", [](Encryptor& self, const PublicKey& public_key, MemoryPoolHandleArgument pool) {
            self.set_public_key(public_key, nullopt_default_pool(pool));
        }, py::arg("public_key"), MEMORY_POOL_ARGUMENT)
        .def("set_secret_key", [](Encryptor& self, const SecretKey& secret_key, MemoryPoolHandleArgument pool) {
            self.set_secret_key(secret_key, nullopt_default_pool(pool));
        }, py::arg("secret_key"), MEMORY_POOL_ARGUMENT)
        
        .def("public_key", &Encryptor::public_key, py::return_value_policy::reference)
        .def("secret_key", &Encryptor::secret_key, py::return_value_policy::reference)

        .def("encrypt_asymmetric", [](const Encryptor& self, const Plaintext& plain, Ciphertext& destination, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            self.encrypt_asymmetric(plain, destination, &rng, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("destination"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_asymmetric", [](const Encryptor& self, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.encrypt_asymmetric(plain, destination, nullptr, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_asymmetric_new", [](const Encryptor& self, const Plaintext& plain, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            return self.encrypt_asymmetric_new(plain, &rng, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_asymmetric_new", [](const Encryptor& self, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            return self.encrypt_asymmetric_new(plain, nullptr, nullopt_default_pool(pool));
        }, py::arg("plain"), MEMORY_POOL_ARGUMENT)

        .def("encrypt_zero_asymmetric", [](const Encryptor& self, Ciphertext& destination, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            self.encrypt_zero_asymmetric(destination, parms_id, &rng, nullopt_default_pool(pool));
        }, py::arg("destination"), py::arg("parms_id"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_zero_asymmetric", [](const Encryptor& self, Ciphertext& destination, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            self.encrypt_zero_asymmetric(destination, parms_id, nullptr, nullopt_default_pool(pool));
        }, py::arg("destination"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_zero_asymmetric_new", [](const Encryptor& self, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            return self.encrypt_zero_asymmetric_new(parms_id, &rng, nullopt_default_pool(pool));
        }, py::arg("parms_id"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_zero_asymmetric_new", [](const Encryptor& self, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            return self.encrypt_zero_asymmetric_new(parms_id, nullptr, nullopt_default_pool(pool));
        }, py::arg("parms_id"), MEMORY_POOL_ARGUMENT)

        .def("encrypt_symmetric", [](const Encryptor& self, const Plaintext& plain, bool save_seed, Ciphertext& destination, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            self.encrypt_symmetric(plain, save_seed, destination, &rng, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("save_seed"), py::arg("destination"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_symmetric", [](const Encryptor& self, const Plaintext& plain, bool save_seed, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.encrypt_symmetric(plain, save_seed, destination, nullptr, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("save_seed"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_symmetric_new", [](const Encryptor& self, const Plaintext& plain, bool save_seed, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            return self.encrypt_symmetric_new(plain, save_seed, &rng, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("save_seed"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_symmetric_new", [](const Encryptor& self, const Plaintext& plain, bool save_seed, MemoryPoolHandleArgument pool) {
            return self.encrypt_symmetric_new(plain, save_seed, nullptr, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("save_seed"), MEMORY_POOL_ARGUMENT)

        .def("encrypt_zero_symmetric", [](const Encryptor& self, bool save_seed, Ciphertext& destination, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            self.encrypt_zero_symmetric(save_seed, destination, parms_id, &rng, nullopt_default_pool(pool));
        }, py::arg("save_seed"), py::arg("destination"), py::arg("parms_id"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_zero_symmetric", [](const Encryptor& self, bool save_seed, Ciphertext& destination, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            self.encrypt_zero_symmetric(save_seed, destination, parms_id, nullptr, nullopt_default_pool(pool));
        }, py::arg("save_seed"), py::arg("destination"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_zero_symmetric_new", [](const Encryptor& self, bool save_seed, std::optional<ParmsID> parms_id, utils::RandomGenerator& rng, MemoryPoolHandleArgument pool) {
            return self.encrypt_zero_symmetric_new(save_seed, parms_id, &rng, nullopt_default_pool(pool));
        }, py::arg("save_seed"), py::arg("parms_id"), py::arg("rng"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_zero_symmetric_new", [](const Encryptor& self, bool save_seed, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            return self.encrypt_zero_symmetric_new(save_seed, parms_id, nullptr, nullopt_default_pool(pool));
        }, py::arg("save_seed"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
    ;

    
}