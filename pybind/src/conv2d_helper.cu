#include "header.cuh"

void register_conv2d_helper(pybind11::module& m) {
    
    py::class_<Conv2dHelper>(m, "Conv2dHelper")
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
        .def("encode_weights", [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& weights) {
            if (weights.ndim() != 1) 
                throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be flattened.");
            if (weights.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be contiguous.");
            if (static_cast<size_t>(weights.size()) != self.output_channels * self.input_channels * self.kernel_height * self.kernel_width) 
                throw std::invalid_argument("[Conv2dHelper::encode_weights] Binder - Weights must be of size oc * ic * kh * kw.");
            return self.encode_weights(encoder, get_pointer_from_buffer(weights));
        })
        .def("encode_inputs", [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& inputs) {
            if (inputs.ndim() != 1) 
                throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be flattened.");
            if (inputs.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be contiguous.");
            if (static_cast<size_t>(inputs.size()) != self.batch_size * self.input_channels * self.image_height * self.image_width)
                throw std::invalid_argument("[Conv2dHelper::encode_inputs] Binder - Inputs must be of size bs * ic * ih * iw.");
            return self.encode_inputs(encoder, get_pointer_from_buffer(inputs));
        })
        .def("encode_outputs", [](const Conv2dHelper& self, const BatchEncoder& encoder, const py::array_t<uint64_t>& outputs) {
            if (outputs.ndim() != 1) 
                throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be flattened.");
            if (outputs.strides(0) != sizeof(uint64_t))
                throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be contiguous.");
            size_t output_height = self.image_height - self.kernel_height + 1;
            size_t output_width = self.image_width - self.kernel_width + 1;
            if (static_cast<size_t>(outputs.size()) != self.batch_size * self.output_channels * output_height * output_width)
                throw std::invalid_argument("[Conv2dHelper::encode_outputs] Binder - Outputs must be of size bs * oc * oh * ow.");
            return self.encode_outputs(encoder, get_pointer_from_buffer(outputs));
        })
        .def("conv2d", &Conv2dHelper::conv2d)
        .def("conv2d_reverse", &Conv2dHelper::conv2d_reverse)
        .def("conv2d_cipher", &Conv2dHelper::conv2d_cipher)
        .def("decrypt_outputs", [](const Conv2dHelper& self, const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) {
            std::vector<uint64_t> result = self.decrypt_outputs(encoder, decryptor, outputs);
            return get_buffer_from_vector(result);
        })
        .def("serialize_outputs", [](const Conv2dHelper& self, const Evaluator &evaluator, const Cipher2d& x) {
            ostringstream ss; self.serialize_outputs(evaluator, x, ss); return py::bytes(ss.str());
        })
        .def("deserialize_outputs", [](const Conv2dHelper& self, const Evaluator &evaluator, const py::bytes& str) {
            istringstream ss(str); return self.deserialize_outputs(evaluator, ss);
        })
    ;
}