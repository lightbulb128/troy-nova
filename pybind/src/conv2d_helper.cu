#include "header.h"


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"


template<typename T>
static void register_methods_conv2d_polynomial_encoder_ring2k(pybind11::class_<Conv2dHelper>& c, const char* bitwidth_name) {
    using Encoder = PolynomialEncoderRing2k<T>;
    auto encode_weights_ring2k = [](
        const Conv2dHelper& self, const Encoder& encoder, const py::array_t<T>& weights, std::optional<ParmsID> parms_id
    ) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(T))
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encode_weights_ring2k<T>(encoder, get_pointer_from_buffer(weights), parms_id);
    };
    auto encode_inputs_ring2k = [](
        const Conv2dHelper& self, const Encoder& encoder, const py::array_t<T>& inputs, std::optional<ParmsID> parms_id
    ) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(T))
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_channels * self.image_height * self.image_width)
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encode_inputs_ring2k<T>(encoder, get_pointer_from_buffer(inputs), parms_id);
    };
    
    auto encrypt_weights_ring2k = [](
        const Conv2dHelper& self, const Encryptor& encryptor, const Encoder& encoder, const py::array_t<T>& weights, std::optional<ParmsID> parms_id
    ) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(T))
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encrypt_weights_ring2k<T>(encryptor, encoder, get_pointer_from_buffer(weights), parms_id);
    };
    auto encrypt_inputs_ring2k = [](
        const Conv2dHelper& self, const Encryptor& encryptor, const Encoder& encoder, const py::array_t<T>& inputs, std::optional<ParmsID> parms_id
    ) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(T))
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_channels * self.image_height * self.image_width)
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encrypt_inputs_ring2k<T>(encryptor, encoder, get_pointer_from_buffer(inputs), parms_id);
    };

    auto encode_outputs_ring2k = [](
        const Conv2dHelper& self, const Encoder& encoder, const py::array_t<T>& outputs, std::optional<ParmsID> parms_id
    ) {
        if (outputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be flattened.");
        if (outputs.strides(0) != sizeof(T))
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be contiguous.");
        size_t output_height = self.image_height - self.kernel_height + 1;
        size_t output_width = self.image_width - self.kernel_width + 1;
        if (static_cast<size_t>(outputs.size()) != self.batch_size * self.output_channels * output_height * output_width)
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be of size batch_size * output_dims.");
        return self.encode_outputs_ring2k<T>(encoder, get_pointer_from_buffer(outputs), parms_id);
    };
    auto decrypt_outputs_ring2k = [](
        const Conv2dHelper& self, const Encoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs
    ) {
        std::vector<T> result = self.decrypt_outputs_ring2k<T>(encoder, decryptor, outputs);
        return get_buffer_from_vector(result);
    };

    c
        .def((std::string("encode_weights_ring2k") + bitwidth_name).c_str(), encode_weights_ring2k)
        .def((std::string("encode_inputs_ring2k") + bitwidth_name).c_str(), encode_inputs_ring2k)
        .def((std::string("encrypt_weights_ring2k") + bitwidth_name).c_str(), encrypt_weights_ring2k)
        .def((std::string("encrypt_inputs_ring2k") + bitwidth_name).c_str(), encrypt_inputs_ring2k)
        .def((std::string("encode_outputs_ring2k") + bitwidth_name).c_str(), encode_outputs_ring2k)
        .def((std::string("decrypt_outputs_ring2k") + bitwidth_name).c_str(), decrypt_outputs_ring2k)
    ;
}

void register_conv2d_helper(pybind11::module& m) {
    
    auto encode_weights_uint64s = [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& weights) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encode_weights_uint64s(encoder, get_pointer_from_buffer(weights));
    };
    auto encode_inputs_uint64s = [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& inputs) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_channels * self.image_height * self.image_width)
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encode_inputs_uint64s(encoder, get_pointer_from_buffer(inputs));
    };

    auto encrypt_weights_uint64s = [](const Conv2dHelper& self, const Encryptor& encryptor, const BatchEncoder& encoder, const py::array_t<uint64_t>& weights) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encrypt_weights_uint64s(encryptor, encoder, get_pointer_from_buffer(weights));
    };
    auto encrypt_inputs_uint64s = [](const Conv2dHelper& self, const Encryptor& encryptor, const BatchEncoder& encoder, const py::array_t<uint64_t>& inputs) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_channels * self.image_height * self.image_width)
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encrypt_inputs_uint64s(encryptor, encoder, get_pointer_from_buffer(inputs));
    };

    auto encode_outputs_uint64s = [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& outputs) {
        if (outputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be flattened.");
        if (outputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be contiguous.");
        size_t output_height = self.image_height - self.kernel_height + 1;
        size_t output_width = self.image_width - self.kernel_width + 1;
        if (static_cast<size_t>(outputs.size()) != self.batch_size * self.output_channels * output_height * output_width)
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be of size batch_size * output_dims.");
        return self.encode_outputs_uint64s(encoder, get_pointer_from_buffer(outputs));
    };
    auto decrypt_outputs_uint64s = [](const Conv2dHelper& self, const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) {
        std::vector<uint64_t> result = self.decrypt_outputs_uint64s(encoder, decryptor, outputs);
        return get_buffer_from_vector(result);
    };

    auto encode_weights_doubles = [](const Conv2dHelper& self, const CKKSEncoder& encoder, const py::array_t<double>& weights, std::optional<ParmsID> parms_id, double scale) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
            throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encode_weights_doubles(encoder, get_pointer_from_buffer(weights), parms_id, scale);
    };
    auto encode_inputs_doubles = [](const Conv2dHelper& self, const CKKSEncoder& encoder, const py::array_t<double>& inputs, std::optional<ParmsID> parms_id, double scale) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_channels * self.image_height * self.image_width)
            throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encode_inputs_doubles(encoder, get_pointer_from_buffer(inputs), parms_id, scale);
    };
    
    auto encrypt_weights_doubles = [](const Conv2dHelper& self, const Encryptor& encryptor, const CKKSEncoder& encoder, const py::array_t<double>& weights, std::optional<ParmsID> parms_id, double scale) {
        if (weights.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be flattened.");
        if (weights.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be contiguous.");
        if (static_cast<size_t>(weights.size()) != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_weights] Binder - Weights must be of size input_dims * output_dims.");
        return self.encrypt_weights_doubles(encryptor, encoder, get_pointer_from_buffer(weights), parms_id, scale);
    };
    auto encrypt_inputs_doubles = [](const Conv2dHelper& self, const Encryptor& encryptor, const CKKSEncoder& encoder, const py::array_t<double>& inputs, std::optional<ParmsID> parms_id, double scale) {
        if (inputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be flattened.");
        if (inputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be contiguous.");
        if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_channels * self.image_height * self.image_width)
            throw std::invalid_argument("[Conv2dHelper::encrypt_inputs] Binder - Inputs must be of size batch_size * input_dims.");
        return self.encrypt_inputs_doubles(encryptor, encoder, get_pointer_from_buffer(inputs), parms_id, scale);
    };

    auto encode_outputs_doubles = [](const Conv2dHelper& self, const CKKSEncoder& encoder, const py::array_t<double>& outputs, std::optional<ParmsID> parms_id, double scale) {
        if (outputs.ndim() != 1) 
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be flattened.");
        if (outputs.strides(0) != sizeof(uint64_t))
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be contiguous.");
        size_t output_height = self.image_height - self.kernel_height + 1;
        size_t output_width = self.image_width - self.kernel_width + 1;
        if (static_cast<size_t>(outputs.size()) != self.batch_size * self.output_channels * output_height * output_width)
            throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be of size batch_size * output_dims.");
        return self.encode_outputs_doubles(encoder, get_pointer_from_buffer(outputs), parms_id, scale);
    };
    auto decrypt_outputs_doubles = [](const Conv2dHelper& self, const CKKSEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) {
        std::vector<double> result = self.decrypt_outputs_doubles(encoder, decryptor, outputs);
        return get_buffer_from_vector(result);
    };


    auto pythonClass = py::class_<Conv2dHelper>(m, "Conv2dHelper")
        .def(py::init([](
                size_t batch_size, size_t input_channels, size_t output_channels,
                size_t image_height, size_t image_width,
                size_t kernel_height, size_t kernel_width,
                size_t slot_count, MatmulObjective objective,
                MemoryPoolHandleArgument pool
            ) {
                return Conv2dHelper(
                    batch_size, input_channels, output_channels,
                    image_height, image_width, kernel_height, kernel_width,
                    slot_count, objective, nullopt_default_pool(pool)
                );
            }), 
            py::arg("batch_size"), py::arg("input_channels"), py::arg("output_channels"),
            py::arg("image_height"), py::arg("image_width"),
            py::arg("kernel_height"), py::arg("kernel_width"),
            py::arg("slot_count"), py::arg("objective"), MEMORY_POOL_ARGUMENT
        )
        .def("set_pool", &Conv2dHelper::set_pool)
        .def("batch_size", [](const Conv2dHelper& self) { return self.batch_size; })
        .def("input_channels", [](const Conv2dHelper& self) { return self.input_channels; })
        .def("output_channels", [](const Conv2dHelper& self) { return self.output_channels; })
        .def("image_height", [](const Conv2dHelper& self) { return self.image_height; })
        .def("image_width", [](const Conv2dHelper& self) { return self.image_width; })
        .def("kernel_height", [](const Conv2dHelper& self) { return self.kernel_height; })
        .def("kernel_width", [](const Conv2dHelper& self) { return self.kernel_width; })
        .def("slot_count", [](const Conv2dHelper& self) { return self.slot_count; })
        .def("objective", [](const Conv2dHelper& self) { return self.objective; })
        .def("batch_block", [](const Conv2dHelper& self) { return self.batch_block; })
        .def("input_channel_block", [](const Conv2dHelper& self) { return self.input_channel_block; })
        .def("output_channel_block", [](const Conv2dHelper& self) { return self.output_channel_block; })
        .def("image_height_block", [](const Conv2dHelper& self) { return self.image_height_block; })
        .def("image_width_block", [](const Conv2dHelper& self) { return self.image_width_block; })
        
        .def("conv2d", &Conv2dHelper::conv2d)
        .def("conv2d_reverse", &Conv2dHelper::conv2d_reverse)
        .def("conv2d_cipher", &Conv2dHelper::conv2d_cipher)
        
        .def("serialize_outputs", [](const Conv2dHelper& self, const Evaluator &evaluator, const Cipher2d& x, CompressionMode mode) {
            ostringstream ss; self.serialize_outputs(evaluator, x, ss, mode); return py::bytes(ss.str());
        }, py::arg("evaluator"), py::arg("x"), COMPRESSION_MODE_ARGUMENT)
        .def("deserialize_outputs", [](const Conv2dHelper& self, const Evaluator &evaluator, const py::bytes& str) {
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

    register_methods_conv2d_polynomial_encoder_ring2k<uint32_t>(pythonClass, "32");
    register_methods_conv2d_polynomial_encoder_ring2k<uint64_t>(pythonClass, "64");
}


#pragma GCC diagnostic pop