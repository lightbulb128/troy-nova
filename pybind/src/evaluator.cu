#include "header.h"

void register_evaluator(pybind11::module& m) {
    
    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<HeContextPointer>())
        .def("context", &Evaluator::context)
        .def("on_device", &Evaluator::on_device)

        // negate(encrypted, destination, pool)
        .def("negate", [](const Evaluator& self, const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.negate(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("negate_inplace", [](const Evaluator& self, Ciphertext& encrypted) {
            self.negate_inplace(encrypted);
        }, py::arg("encrypted"))
        .def("negate_new", [](const Evaluator& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.negate_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)

        // add(encrypted1, encrypted2, destination, pool). 
        .def("add", [](const Evaluator& self, const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.add(encrypted1, encrypted2, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("add_inplace", [](const Evaluator& self, Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandleArgument pool) {
            self.add_inplace(encrypted1, encrypted2, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), MEMORY_POOL_ARGUMENT)
        .def("add_new", [](const Evaluator& self, const Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandleArgument pool) {
            return self.add_new(encrypted1, encrypted2, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), MEMORY_POOL_ARGUMENT)

        // sub(encrypted1, encrypted2, destination, pool)
        .def("sub", [](const Evaluator& self, const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.sub(encrypted1, encrypted2, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("sub_inplace", [](const Evaluator& self, Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandleArgument pool) {
            self.sub_inplace(encrypted1, encrypted2, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), MEMORY_POOL_ARGUMENT)
        .def("sub_new", [](const Evaluator& self, const Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandleArgument pool) {
            return self.sub_new(encrypted1, encrypted2, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), MEMORY_POOL_ARGUMENT)

        // multiply(encrypted1, encrypted2, destination, pool)
        .def("multiply", [](const Evaluator& self, const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.multiply(encrypted1, encrypted2, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("multiply_inplace", [](const Evaluator& self, Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandleArgument pool) {
            self.multiply_inplace(encrypted1, encrypted2, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), MEMORY_POOL_ARGUMENT)
        .def("multiply_new", [](const Evaluator& self, const Ciphertext& encrypted1, const Ciphertext& encrypted2, MemoryPoolHandleArgument pool) {
            return self.multiply_new(encrypted1, encrypted2, nullopt_default_pool(pool));
        }, py::arg("encrypted1"), py::arg("encrypted2"), MEMORY_POOL_ARGUMENT)

        // square(encrypted, destination, pool)
        .def("square", [](const Evaluator& self, const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.square(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("square_inplace", [](const Evaluator& self, Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            self.square_inplace(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)
        .def("square_new", [](const Evaluator& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.square_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)

        // apply_keyswitching(encrypted, keyswitch_keys, destination, pool)
        .def("apply_keyswitching_inplace", [](const Evaluator& self, Ciphertext& encrypted, const KSwitchKeys& keyswitch_keys, MemoryPoolHandleArgument pool) {
            self.apply_keyswitching_inplace(encrypted, keyswitch_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("keyswitch_keys"), MEMORY_POOL_ARGUMENT)
        .def("apply_keyswitching", [](const Evaluator& self, const Ciphertext& encrypted, const KSwitchKeys& keyswitch_keys, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.apply_keyswitching(encrypted, keyswitch_keys, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("keyswitch_keys"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("apply_keyswitching_new", [](const Evaluator& self, const Ciphertext& encrypted, const KSwitchKeys& keyswitch_keys, MemoryPoolHandleArgument pool) {
            return self.apply_keyswitching_new(encrypted, keyswitch_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("keyswitch_keys"), MEMORY_POOL_ARGUMENT)

        // relinearize(encrypted, relin_keys, destination, pool)
        .def("relinearize_inplace", [](const Evaluator& self, Ciphertext& encrypted, const RelinKeys& relin_keys, MemoryPoolHandleArgument pool) {
            self.relinearize_inplace(encrypted, relin_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("relin_keys"), MEMORY_POOL_ARGUMENT)
        .def("relinearize", [](const Evaluator& self, const Ciphertext& encrypted, const RelinKeys& relin_keys, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.relinearize(encrypted, relin_keys, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("relin_keys"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("relinearize_new", [](const Evaluator& self, const Ciphertext& encrypted, const RelinKeys& relin_keys, MemoryPoolHandleArgument pool) {
            return self.relinearize_new(encrypted, relin_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("relin_keys"), MEMORY_POOL_ARGUMENT)

        // mod_switch_to_next(encrypted, destination, pool)
        .def("mod_switch_to_next_inplace", [](const Evaluator& self, Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            self.mod_switch_to_next_inplace(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_to_next", [](const Evaluator& self, const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.mod_switch_to_next(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_to_next_new", [](const Evaluator& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.mod_switch_to_next_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)

        // mod_switch_plain_to_next(plain, destination, pool)
        .def("mod_switch_plain_to_next", [](const Evaluator& self, const Plaintext& plain, Plaintext& destination, MemoryPoolHandleArgument pool) {
            self.mod_switch_plain_to_next(plain, destination, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_plain_to_next_inplace", [](const Evaluator& self, Plaintext& plain, MemoryPoolHandleArgument pool) {
            self.mod_switch_plain_to_next_inplace(plain, nullopt_default_pool(pool));
        }, py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_plain_to_next_new", [](const Evaluator& self, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            return self.mod_switch_plain_to_next_new(plain, nullopt_default_pool(pool));
        }, py::arg("plain"), MEMORY_POOL_ARGUMENT)

        // mod_switch_to(encrypted, parms_id, destination, pool)
        .def("mod_switch_to", [](const Evaluator& self, const Ciphertext& encrypted, ParmsID parms_id, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.mod_switch_to(encrypted, parms_id, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_to_inplace", [](const Evaluator& self, Ciphertext& encrypted, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            self.mod_switch_to_inplace(encrypted, parms_id, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_to_new", [](const Evaluator& self, const Ciphertext& encrypted, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            return self.mod_switch_to_new(encrypted, parms_id, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)

        // mod_switch_plain_to(plain, parms_id, destination, pool)
        .def("mod_switch_plain_to", [](const Evaluator& self, const Plaintext& plain, ParmsID parms_id, Plaintext& destination, MemoryPoolHandleArgument pool) {
            self.mod_switch_plain_to(plain, parms_id, destination, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_plain_to_inplace", [](const Evaluator& self, Plaintext& plain, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            self.mod_switch_plain_to_inplace(plain, parms_id, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_plain_to_new", [](const Evaluator& self, const Plaintext& plain, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            return self.mod_switch_plain_to_new(plain, parms_id, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)

        // rescale_to_next(encrypted, destination, pool)
        .def("rescale_to_next", [](const Evaluator& self, const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.rescale_to_next(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("rescale_to_next_inplace", [](const Evaluator& self, Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            self.rescale_to_next_inplace(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)
        .def("rescale_to_next_new", [](const Evaluator& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.rescale_to_next_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)

        // rescale_to(encrypted, parms_id, destination, pool)
        .def("rescale_to", [](const Evaluator& self, const Ciphertext& encrypted, ParmsID parms_id, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.rescale_to(encrypted, parms_id, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("rescale_to_inplace", [](const Evaluator& self, Ciphertext& encrypted, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            self.rescale_to_inplace(encrypted, parms_id, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("rescale_to_new", [](const Evaluator& self, const Ciphertext& encrypted, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            return self.rescale_to_new(encrypted, parms_id, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)

        // add_plain(encrypted, plain, destination, pool)
        .def("add_plain", [](const Evaluator& self, const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.add_plain(encrypted, plain, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("add_plain_inplace", [](const Evaluator& self, Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            self.add_plain_inplace(encrypted, plain, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("add_plain_new", [](const Evaluator& self, const Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            return self.add_plain_new(encrypted, plain, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        
        // sub_plain(encrypted, plain, destination, pool)
        .def("sub_plain", [](const Evaluator& self, const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.sub_plain(encrypted, plain, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("sub_plain_inplace", [](const Evaluator& self, Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            self.sub_plain_inplace(encrypted, plain, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("sub_plain_new", [](const Evaluator& self, const Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            return self.sub_plain_new(encrypted, plain, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), MEMORY_POOL_ARGUMENT)

        // multiply_plain(encrypted, plain, destination, pool)
        .def("multiply_plain", [](const Evaluator& self, const Ciphertext& encrypted, const Plaintext& plain, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.multiply_plain(encrypted, plain, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("multiply_plain_inplace", [](const Evaluator& self, Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            self.multiply_plain_inplace(encrypted, plain, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("multiply_plain_new", [](const Evaluator& self, const Ciphertext& encrypted, const Plaintext& plain, MemoryPoolHandleArgument pool) {
            return self.multiply_plain_new(encrypted, plain, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("multiply_plain_new_batched", [](const Evaluator& self, const py::list& encrypted, const py::list& plain, MemoryPoolHandleArgument pool) {
            return self.multiply_plain_new_batched(
                cast_list<const Ciphertext*>(encrypted), cast_list<const Plaintext*>(plain), 
                nullopt_default_pool(pool)
            );
        }, py::arg("encrypted"), py::arg("plain"), MEMORY_POOL_ARGUMENT)

        // transform_plain_to_ntt(plain, parms_id, destination, pool)
        .def("transform_plain_to_ntt", [](const Evaluator& self, const Plaintext& plain, ParmsID parms_id, Plaintext& destination, MemoryPoolHandleArgument pool) {
            self.transform_plain_to_ntt(plain, parms_id, destination, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("transform_plain_to_ntt_inplace", [](const Evaluator& self, Plaintext& plain, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            self.transform_plain_to_ntt_inplace(plain, parms_id, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("transform_plain_to_ntt_new", [](const Evaluator& self, const Plaintext& plain, ParmsID parms_id, MemoryPoolHandleArgument pool) {
            return self.transform_plain_to_ntt_new(plain, parms_id, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)

        // transform_to_ntt(encrypted, destination, pool)
        .def("transform_to_ntt", [](const Evaluator& self, const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.transform_to_ntt(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("transform_to_ntt_inplace", [](const Evaluator& self, Ciphertext& encrypted) {
            self.transform_to_ntt_inplace(encrypted);
        }, py::arg("encrypted"))
        .def("transform_to_ntt_new", [](const Evaluator& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.transform_to_ntt_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)

        // transform_from_ntt(encrypted, destination, pool)
        .def("transform_from_ntt", [](const Evaluator& self, const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.transform_from_ntt(encrypted, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("transform_from_ntt_inplace", [](const Evaluator& self, Ciphertext& encrypted) {
            self.transform_from_ntt_inplace(encrypted);
        }, py::arg("encrypted"))
        .def("transform_from_ntt_new", [](const Evaluator& self, const Ciphertext& encrypted, MemoryPoolHandleArgument pool) {
            return self.transform_from_ntt_new(encrypted, nullopt_default_pool(pool));
        }, py::arg("encrypted"), MEMORY_POOL_ARGUMENT)

        // apply_galois(encrypted, galois_element, galois_keys, destination, pool)
        .def("apply_galois", [](const Evaluator& self, const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.apply_galois(encrypted, galois_element, galois_keys, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_element"), py::arg("galois_keys"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("apply_galois_inplace", [](const Evaluator& self, Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            self.apply_galois_inplace(encrypted, galois_element, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_element"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)
        .def("apply_galois_new", [](const Evaluator& self, const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            return self.apply_galois_new(encrypted, galois_element, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_element"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)

        // apply_galois_plain(plain, galois_element, destination, pool)
        .def("apply_galois_plain", [](const Evaluator& self, const Plaintext& plain, size_t galois_element, Plaintext& destination, MemoryPoolHandleArgument pool) {
            self.apply_galois_plain(plain, galois_element, destination, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("galois_element"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("apply_galois_plain_inplace", [](const Evaluator& self, Plaintext& plain, size_t galois_element, MemoryPoolHandleArgument pool) {
            self.apply_galois_plain_inplace(plain, galois_element, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("galois_element"), MEMORY_POOL_ARGUMENT)
        .def("apply_galois_plain_new", [](const Evaluator& self, const Plaintext& plain, size_t galois_element, MemoryPoolHandleArgument pool) {
            return self.apply_galois_plain_new(plain, galois_element, nullopt_default_pool(pool));
        }, py::arg("plain"), py::arg("galois_element"), MEMORY_POOL_ARGUMENT)

        // rotate_rows(encrypted, steps, galois_keys, destination, pool)
        .def("rotate_rows", [](const Evaluator& self, const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.rotate_rows(encrypted, steps, galois_keys, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("rotate_rows_inplace", [](const Evaluator& self, Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            self.rotate_rows_inplace(encrypted, steps, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)
        .def("rotate_rows_new", [](const Evaluator& self, const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            return self.rotate_rows_new(encrypted, steps, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)

        // rotate_columns(encrypted, galois_keys, destination, pool)
        .def("rotate_columns", [](const Evaluator& self, const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.rotate_columns(encrypted, galois_keys, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_keys"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("rotate_columns_inplace", [](const Evaluator& self, Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            self.rotate_columns_inplace(encrypted, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)
        .def("rotate_columns_new", [](const Evaluator& self, const Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            return self.rotate_columns_new(encrypted, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)

        // rotate_vector(encrypted, steps, galois_keys, destination, pool)
        .def("rotate_vector", [](const Evaluator& self, const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.rotate_vector(encrypted, steps, galois_keys, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("rotate_vector_inplace", [](const Evaluator& self, Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            self.rotate_vector_inplace(encrypted, steps, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)
        .def("rotate_vector_new", [](const Evaluator& self, const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            return self.rotate_vector_new(encrypted, steps, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("steps"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)

        // complex_conjugate(encrypted, galois_keys, destination, pool)
        .def("complex_conjugate", [](const Evaluator& self, const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.complex_conjugate(encrypted, galois_keys, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_keys"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("complex_conjugate_inplace", [](const Evaluator& self, Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            self.complex_conjugate_inplace(encrypted, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)
        .def("complex_conjugate_new", [](const Evaluator& self, const Ciphertext& encrypted, const GaloisKeys& galois_keys, MemoryPoolHandleArgument pool) {
            return self.complex_conjugate_new(encrypted, galois_keys, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_keys"), MEMORY_POOL_ARGUMENT)

        // packlwes related
        .def("extract_lwe_new", [](const Evaluator& self, const Ciphertext& encrypted, size_t term, MemoryPoolHandleArgument pool) {
            return self.extract_lwe_new(encrypted, term, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("term"), MEMORY_POOL_ARGUMENT)

        .def("assemble_lwe_new", [](const Evaluator& self, const LWECiphertext& lwe_ciphertext, MemoryPoolHandleArgument pool) {
            return self.assemble_lwe_new(lwe_ciphertext, nullopt_default_pool(pool));
        }, py::arg("lwe_ciphertext"), MEMORY_POOL_ARGUMENT)

        .def("field_trace_inplace", [](const Evaluator& self, Ciphertext& encrypted, const GaloisKeys& galois_keys, size_t logn, MemoryPoolHandleArgument pool) {
            self.field_trace_inplace(encrypted, galois_keys, logn, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("galois_keys"), py::arg("logn"), MEMORY_POOL_ARGUMENT)
        
        .def("divide_by_poly_modulus_degree_inplace", [](const Evaluator& self, Ciphertext& encrypted, uint64_t mul) {
            self.divide_by_poly_modulus_degree_inplace(encrypted, mul);
        }, py::arg("encrypted"), py::arg("mul") = 1)

        .def("pack_lwe_ciphertexts_new", [](const Evaluator& self, const std::vector<LWECiphertext>& lwe_ciphertexts, const GaloisKeys& automorphism_keys, MemoryPoolHandleArgument pool) {
            return self.pack_lwe_ciphertexts_new(lwe_ciphertexts, automorphism_keys, nullopt_default_pool(pool));
        }, py::arg("lwe_ciphertexts"), py::arg("automorphism_keys"), MEMORY_POOL_ARGUMENT)
        .def("pack_lwe_ciphertexts_new", [](const Evaluator& self, const py::list& lwe_ciphertexts, const GaloisKeys& automorphism_keys, MemoryPoolHandleArgument pool) {
            return self.pack_lwe_ciphertexts_new(cast_list<const LWECiphertext*>(lwe_ciphertexts), automorphism_keys, nullopt_default_pool(pool));
        }, py::arg("lwe_ciphertexts"), py::arg("automorphism_keys"), MEMORY_POOL_ARGUMENT)
        .def("pack_rlwe_ciphertexts_new", [](
            const Evaluator& self, const py::list& rlwe_ciphertexts, 
            const GaloisKeys& automorphism_keys, 
            size_t shift, size_t input_interval, size_t output_interval, 
            MemoryPoolHandleArgument pool
        ) {
            return self.pack_rlwe_ciphertexts_new(
                cast_list<const Ciphertext*>(rlwe_ciphertexts), 
                automorphism_keys, shift, input_interval, output_interval, 
                nullopt_default_pool(pool)
            );
        }, 
            py::arg("rlwe_ciphertexts"), py::arg("automorphism_keys"), 
            py::arg("shift"), py::arg("input_interval"), py::arg("output_interval"), 
            MEMORY_POOL_ARGUMENT
        )

        .def("pack_lwe_ciphertexts_new_batched", [](const Evaluator& self, const py::list& lwe_groups, const GaloisKeys& automorphism_keys, MemoryPoolHandleArgument pool) {
            std::vector<std::vector<const LWECiphertext*>> cvv(lwe_groups.size());
            for (size_t i = 0; i < lwe_groups.size(); i++) {
                py::list pv = lwe_groups[i].cast<py::list>();
                cvv[i] = cast_list<const LWECiphertext*>(pv);
            }
            return self.pack_lwe_ciphertexts_new_batched(cvv, automorphism_keys, nullopt_default_pool(pool));
        }, py::arg("lwe_groups"), py::arg("automorphism_keys"), MEMORY_POOL_ARGUMENT)
        .def("pack_rlwe_ciphertexts_new_batched", [](
            const Evaluator& self, const py::list& rlwe_groups, 
            const GaloisKeys& automorphism_keys, 
            size_t shift, size_t input_interval, size_t output_interval, 
            MemoryPoolHandleArgument pool
        ) {
            std::vector<std::vector<const Ciphertext*>> cvv(rlwe_groups.size());
            for (size_t i = 0; i < rlwe_groups.size(); i++) {
                py::list pv = rlwe_groups[i].cast<py::list>();
                cvv[i] = cast_list<const Ciphertext*>(pv);
            }
            return self.pack_rlwe_ciphertexts_new_batched(
                cvv, automorphism_keys, shift, input_interval, output_interval, 
                nullopt_default_pool(pool)
            );
        }, 
            py::arg("rlwe_groups"), py::arg("automorphism_keys"), 
            py::arg("shift"), py::arg("input_interval"), py::arg("output_interval"), 
            MEMORY_POOL_ARGUMENT
        )

        // negacyclic_shift(encrypted, size_t shift, destination, pool)
        .def("negacyclic_shift", [](const Evaluator& self, const Ciphertext& encrypted, size_t shift, Ciphertext& destination, MemoryPoolHandleArgument pool) {
            self.negacyclic_shift(encrypted, shift, destination, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("shift"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("negacyclic_shift_inplace", [](const Evaluator& self, Ciphertext& encrypted, size_t shift, MemoryPoolHandleArgument pool) {
            self.negacyclic_shift_inplace(encrypted, shift, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("shift"), MEMORY_POOL_ARGUMENT)
        .def("negacyclic_shift_new", [](const Evaluator& self, const Ciphertext& encrypted, size_t shift, MemoryPoolHandleArgument pool) {
            return self.negacyclic_shift_new(encrypted, shift, nullopt_default_pool(pool));
        }, py::arg("encrypted"), py::arg("shift"), MEMORY_POOL_ARGUMENT)
    ;

}