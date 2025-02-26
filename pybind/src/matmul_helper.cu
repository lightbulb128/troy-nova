#include "header.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

template<typename T>
static void register_methods_matmul_polynomial_encoder_ring2k(pybind11::class_<MatmulHelper>& c, const char* bitwidth_name) {
    using Encoder = PolynomialEncoderRing2k<T>;
    auto encode_weights_ring2k = [](
        const MatmulHelper& self, const Encoder& encoder, const py::array_t<T>& weights, std::optional<ParmsID> parms_id
    ) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(T))
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.input_dims * self.output_dims) 
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encode_weights_ring2k<T>(encoder, get_pointer_from_buffer(weights), parms_id);
    };
    auto encode_inputs_ring2k = [](
        const MatmulHelper& self, const Encoder& encoder, const py::array_t<T>& inputs, std::optional<ParmsID> parms_id
    ) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(T))
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_dims)
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encode_inputs_ring2k<T>(encoder, get_pointer_from_buffer(inputs), parms_id);
    };
    
    auto encrypt_weights_ring2k = [](
        const MatmulHelper& self, const Encryptor& encryptor, const Encoder& encoder, const py::array_t<T>& weights, std::optional<ParmsID> parms_id
    ) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(T))
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.input_dims * self.output_dims) 
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encrypt_weights_ring2k<T>(encryptor, encoder, get_pointer_from_buffer(weights), parms_id);
    };
    auto encrypt_inputs_ring2k = [](
        const MatmulHelper& self, const Encryptor& encryptor, const Encoder& encoder, const py::array_t<T>& inputs, std::optional<ParmsID> parms_id
    ) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(T))
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_dims)
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encrypt_inputs_ring2k<T>(encryptor, encoder, get_pointer_from_buffer(inputs), parms_id);
    };

    auto encode_outputs_ring2k = [](
        const MatmulHelper& self, const Encoder& encoder, const py::array_t<T>& outputs, std::optional<ParmsID> parms_id
    ) {
        if (outputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be flattened.");
        if (outputs.strides(0) != sizeof(T))
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be contiguous.");
        if (static_cast<size_t>(outputs.size()) != self.batch_size * self.output_dims)
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be of size batch_size * output_dims.");
        return self.encode_outputs_ring2k<T>(encoder, get_pointer_from_buffer(outputs), parms_id);
    };
    auto decrypt_outputs_ring2k = [](
        const MatmulHelper& self, const Encoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs
    ) {
        std::vector<T> result = self.decrypt_outputs_ring2k<T>(encoder, decryptor, outputs);
        return get_buffer_from_vector(result);
    };

    c
        .def((std::string("encode_weights_ring2k") + bitwidth_name).c_str(), encode_weights_ring2k)
        .def((std::string("encode_inputs_ring2k") + bitwidth_name).c_str(), encode_inputs_ring2k)
        .def((std::string("encode_outputs_ring2k") + bitwidth_name).c_str(), encode_outputs_ring2k)
        .def((std::string("encrypt_weights_ring2k") + bitwidth_name).c_str(), encrypt_weights_ring2k)
        .def((std::string("encrypt_inputs_ring2k") + bitwidth_name).c_str(), encrypt_inputs_ring2k)
        .def((std::string("decrypt_outputs_ring2k") + bitwidth_name).c_str(), decrypt_outputs_ring2k)
    ;
}

void register_matmul_helper(pybind11::module& m) {

    py::class_<Plain2d>(m, "Plain2d")
        .def(py::init<>())
        .def("size", &Plain2d::size)
        .def("rows", &Plain2d::rows)
        .def("columns", &Plain2d::columns)
        .def("clone", [](const Plain2d& self, MemoryPoolHandleArgument pool) {
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("encrypt_asymmetric", [](const Plain2d& self, const Encryptor& encryptor, MemoryPoolHandleArgument pool) {
            return self.encrypt_asymmetric(encryptor, nullopt_default_pool(pool));
        }, py::arg("encryptor"), MEMORY_POOL_ARGUMENT)
        .def("encrypt_symmetric", [](const Plain2d& self, const Encryptor& encryptor, MemoryPoolHandleArgument pool) {
            return self.encrypt_symmetric(encryptor, nullopt_default_pool(pool));
        }, py::arg("encryptor"), MEMORY_POOL_ARGUMENT)
        .def("get", [](const Plain2d& self, size_t i, size_t j) {
            return self[i][j];
        }, py::arg("i"), py::arg("j"), py::return_value_policy::reference)
    ;

    py::class_<Cipher2d>(m, "Cipher2d")
        .def(py::init<>())
        .def("size", &Cipher2d::size)
        .def("rows", &Cipher2d::rows)
        .def("columns", &Cipher2d::columns)
        .def("clone", [](const Cipher2d& self, MemoryPoolHandleArgument pool) {
            return self.clone(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)
        .def("expand_seed", &Cipher2d::expand_seed)

        .def("save", [](const Cipher2d& self, HeContextPointer context, CompressionMode mode) {return save_he(self, context, mode); },
            py::arg("context"), COMPRESSION_MODE_ARGUMENT)
        .def("load", [](Cipher2d& self, const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_he<Cipher2d>(self, str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def_static("load_new", [](const py::bytes& str, HeContextPointer context, MemoryPoolHandleArgument pool) {return load_new_he<Cipher2d>(str, context, nullopt_default_pool(pool)); },
            py::arg("str"), py::arg("context"), MEMORY_POOL_ARGUMENT)
        .def("serialized_size_upperbound", [](const Cipher2d& self, HeContextPointer context, CompressionMode mode) {return serialized_size_upperbound_he(self, context, mode); },
            py::arg("context") = nullptr, COMPRESSION_MODE_ARGUMENT)
            
        .def("mod_switch_to_next_inplace", [](Cipher2d& self, const Evaluator& evaluator, MemoryPoolHandleArgument pool) {
            self.mod_switch_to_next_inplace(evaluator, nullopt_default_pool(pool));
        }, py::arg("evaluator"), MEMORY_POOL_ARGUMENT)
        .def("mod_switch_to_next", [](const Cipher2d& self, const Evaluator& evaluator, MemoryPoolHandleArgument pool) {
            return self.mod_switch_to_next(evaluator, nullopt_default_pool(pool));
        }, py::arg("evaluator"), MEMORY_POOL_ARGUMENT)

        .def("relinearize_inplace", [](Cipher2d& self, const Evaluator& evaluator, const RelinKeys& relin_keys, MemoryPoolHandleArgument pool) {
            self.relinearize_inplace(evaluator, relin_keys, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("relin_keys"), MEMORY_POOL_ARGUMENT)
        .def("relinearize", [](const Cipher2d& self, const Evaluator& evaluator, const RelinKeys& relin_keys, MemoryPoolHandleArgument pool) {
            return self.relinearize(evaluator, relin_keys, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("relin_keys"), MEMORY_POOL_ARGUMENT)

        .def("add", [](const Cipher2d& self, const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandleArgument pool) {
            return self.add(evaluator, other, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("other"), MEMORY_POOL_ARGUMENT)
        .def("add_inplace", [](Cipher2d& self, const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandleArgument pool) {
            self.add_inplace(evaluator, other, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("other"), MEMORY_POOL_ARGUMENT)

        .def("add_plain", [](const Cipher2d& self, const Evaluator& evaluator, const Plain2d& plain, MemoryPoolHandleArgument pool) {
            return self.add_plain(evaluator, plain, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("add_plain_inplace", [](Cipher2d& self, const Evaluator& evaluator, const Plain2d& plain, MemoryPoolHandleArgument pool) {
            self.add_plain_inplace(evaluator, plain, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("plain"), MEMORY_POOL_ARGUMENT)

        .def("sub", [](const Cipher2d& self, const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandleArgument pool) {
            return self.sub(evaluator, other, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("other"), MEMORY_POOL_ARGUMENT)
        .def("sub_inplace", [](Cipher2d& self, const Evaluator& evaluator, const Cipher2d& other, MemoryPoolHandleArgument pool) {
            self.sub_inplace(evaluator, other, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("other"), MEMORY_POOL_ARGUMENT)

        .def("sub_plain", [](const Cipher2d& self, const Evaluator& evaluator, const Plain2d& plain, MemoryPoolHandleArgument pool) {
            return self.sub_plain(evaluator, plain, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("sub_plain_inplace", [](Cipher2d& self, const Evaluator& evaluator, const Plain2d& plain, MemoryPoolHandleArgument pool) {
            self.sub_plain_inplace(evaluator, plain, nullopt_default_pool(pool));
        }, py::arg("evaluator"), py::arg("plain"), MEMORY_POOL_ARGUMENT)
        .def("decrypt", [](const Cipher2d& self, const Decryptor& decryptor, MemoryPoolHandleArgument pool) {
            return self.decrypt(decryptor, nullopt_default_pool(pool));
        }, py::arg("decryptor"), MEMORY_POOL_ARGUMENT)

        .def("get", [](const Cipher2d& self, size_t i, size_t j) {
            return self[i][j];
        }, py::arg("i"), py::arg("j"), py::return_value_policy::reference)

    ;

    py::enum_<MatmulObjective>(m, "MatmulObjective")
        .value("EncryptLeft", MatmulObjective::EncryptLeft)
        .value("EncryptRight", MatmulObjective::EncryptRight)
        .value("Crossed", MatmulObjective::Crossed)
    ;


    auto encode_weights_uint64s = [](const MatmulHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& weights) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.input_dims * self.output_dims) 
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encode_weights_uint64s(encoder, get_pointer_from_buffer(weights));
    };
    auto encode_inputs_uint64s = [](const MatmulHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& inputs) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_dims)
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encode_inputs_uint64s(encoder, get_pointer_from_buffer(inputs));
    };
    
    auto encrypt_weights_uint64s = [](const MatmulHelper& self, const Encryptor& encryptor, const BatchEncoder& encoder, const py::array_t<uint64_t>& weights) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.input_dims * self.output_dims) 
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encrypt_weights_uint64s(encryptor, encoder, get_pointer_from_buffer(weights));
    };
    auto encrypt_inputs_uint64s = [](const MatmulHelper& self, const Encryptor& encryptor, const BatchEncoder& encoder, const py::array_t<uint64_t>& inputs) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_dims)
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encrypt_inputs_uint64s(encryptor, encoder, get_pointer_from_buffer(inputs));
    };

    auto encode_outputs_uint64s = [](const MatmulHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& outputs) {
        if (outputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be flattened.");
        if (outputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be contiguous.");
        if (static_cast<size_t>(outputs.size()) != self.batch_size * self.output_dims)
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be of size batch_size * output_dims.");
        return self.encode_outputs_uint64s(encoder, get_pointer_from_buffer(outputs));
    };
    auto decrypt_outputs_uint64s = [](const MatmulHelper& self, const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) {
        std::vector<uint64_t> result = self.decrypt_outputs_uint64s(encoder, decryptor, outputs);
        return get_buffer_from_vector(result);
    };

    auto encode_weights_doubles = [](const MatmulHelper& self, const CKKSEncoder& encoder, const py::array_t<double>& weights, std::optional<ParmsID> parms_id, double scale) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.input_dims * self.output_dims) 
            throw std::invalid_argument("[MatmulHelper::encode_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encode_weights_doubles(encoder, get_pointer_from_buffer(weights), parms_id, scale);
    };
    auto encode_inputs_doubles = [](const MatmulHelper& self, const CKKSEncoder& encoder, const py::array_t<double>& inputs, std::optional<ParmsID> parms_id, double scale) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_dims)
            throw std::invalid_argument("[MatmulHelper::encode_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encode_inputs_doubles(encoder, get_pointer_from_buffer(inputs), parms_id, scale);
    };

    auto encrypt_weights_doubles = [](const MatmulHelper& self, const Encryptor& encryptor, const CKKSEncoder& encoder, const py::array_t<double>& weights, std::optional<ParmsID> parms_id, double scale) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.input_dims * self.output_dims) 
            throw std::invalid_argument("[MatmulHelper::encrypt_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encrypt_weights_doubles(encryptor, encoder, get_pointer_from_buffer(weights), parms_id, scale);
    };
    auto encrypt_inputs_doubles = [](const MatmulHelper& self, const Encryptor& encryptor, const CKKSEncoder& encoder, const py::array_t<double>& inputs, std::optional<ParmsID> parms_id, double scale) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_dims)
            throw std::invalid_argument("[MatmulHelper::encrypt_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encrypt_inputs_doubles(encryptor, encoder, get_pointer_from_buffer(inputs), parms_id, scale);
    };

    auto encode_outputs_doubles = [](const MatmulHelper& self, const CKKSEncoder& encoder, const py::array_t<double>& outputs, std::optional<ParmsID> parms_id, double scale) {
        if (outputs.ndim() != 1) 
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be flattened.");
        if (outputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be contiguous.");
        if (static_cast<size_t>(outputs.size()) != self.batch_size * self.output_dims)
            throw std::invalid_argument("[MatmulHelper::encode_outputs] Binder - Outputs must be of size batch_size * output_dims.");
        return self.encode_outputs_doubles(encoder, get_pointer_from_buffer(outputs), parms_id, scale);
    };
    auto decrypt_outputs_doubles = [](const MatmulHelper& self, const CKKSEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) {
        std::vector<double> result = self.decrypt_outputs_doubles(encoder, decryptor, outputs);
        return get_buffer_from_vector(result);
    };

    auto pythonMatmulHelperClass = py::class_<MatmulHelper>(m, "MatmulHelper")
        .def(py::init([](
                size_t batch_size, size_t input_dims, size_t output_dims, size_t slot_count, 
                MatmulObjective objective, bool pack_lwe, MemoryPoolHandleArgument pool
            ) {
                return MatmulHelper(
                    batch_size, input_dims, output_dims, slot_count, objective, pack_lwe,
                    nullopt_default_pool(pool)
                );
            }),
            py::arg("batch_size"), py::arg("input_dims"), py::arg("output_dims"),
            py::arg("slot_count"), py::arg_v("objective", MatmulObjective::EncryptLeft, "MatmulObjective.EncryptLeft"),
            py::arg("pack_lwe") = true, MEMORY_POOL_ARGUMENT
        )
        .def("set_pool", &MatmulHelper::set_pool)
        .def("batch_size", [](const MatmulHelper& self) { return self.batch_size; })
        .def("input_dims", [](const MatmulHelper& self) { return self.input_dims; })
        .def("output_dims", [](const MatmulHelper& self) { return self.output_dims; })
        .def("slot_count", [](const MatmulHelper& self) { return self.slot_count; })
        .def("objective", [](const MatmulHelper& self) { return self.objective; })
        .def("pack_lwe", [](const MatmulHelper& self) { return self.pack_lwe; })
        .def("batch_block", [](const MatmulHelper& self) { return self.batch_block; })
        .def("input_block", [](const MatmulHelper& self) { return self.input_block; })
        .def("output_block", [](const MatmulHelper& self) { return self.output_block; })

        .def("matmul", &MatmulHelper::matmul)
        .def("matmul_reverse", &MatmulHelper::matmul_reverse)
        .def("matmul_cipher", &MatmulHelper::matmul_cipher)
        .def("pack_outputs", &MatmulHelper::pack_outputs)

        .def("serialize_outputs", [](const MatmulHelper& self, const Evaluator &evaluator, const Cipher2d& x, CompressionMode mode) {
            ostringstream ss; self.serialize_outputs(evaluator, x, ss, mode); return py::bytes(ss.str());
        }, py::arg("evaluator"), py::arg("x"), COMPRESSION_MODE_ARGUMENT)
        .def("deserialize_outputs", [](const MatmulHelper& self, const Evaluator &evaluator, const py::bytes& str) {
            istringstream ss(str); return self.deserialize_outputs(evaluator, ss);
        })
        
        .def("encode_weights", encode_weights_uint64s)
        .def("encode_inputs", encode_inputs_uint64s)
        .def("encrypt_weights", encrypt_weights_uint64s)
        .def("encrypt_inputs", encrypt_inputs_uint64s)
        .def("encode_outputs", encode_outputs_uint64s)
        .def("decrypt_outputs", decrypt_outputs_uint64s)
        .def("encode_weights_uint64s", encode_weights_uint64s)
        .def("encode_inputs_uint64s", encode_inputs_uint64s)
        .def("encrypt_weights_uint64s", encrypt_weights_uint64s)
        .def("encrypt_inputs_uint64s", encrypt_inputs_uint64s)
        .def("encode_outputs_uint64s", encode_outputs_uint64s)
        .def("decrypt_outputs_uint64s", decrypt_outputs_uint64s)
        .def("encode_weights_doubles", encode_weights_doubles)
        .def("encode_inputs_doubles", encode_inputs_doubles)
        .def("encrypt_weights_doubles", encrypt_weights_doubles)
        .def("encrypt_inputs_doubles", encrypt_inputs_doubles)
        .def("encode_outputs_doubles", encode_outputs_doubles)
        .def("decrypt_outputs_doubles", decrypt_outputs_doubles)
    ;

    register_methods_matmul_polynomial_encoder_ring2k<uint32_t>(pythonMatmulHelperClass, "32");
    register_methods_matmul_polynomial_encoder_ring2k<uint64_t>(pythonMatmulHelperClass, "64");

}

#pragma GCC diagnostic pop