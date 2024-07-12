#include "header.h"

template<typename T>
static void register_class_polynomial_encoder_ring2k(pybind11::module_& m, const char* name) {
    using Encoder = PolynomialEncoderRing2k<T>;
    py::class_<Encoder>(m, name)
        .def(py::init<HeContextPointer, size_t>())
        .def("context", &Encoder::context)
        .def("on_device", &Encoder::on_device)
        .def("t_bit_length", &Encoder::t_bit_length)

        .def("to_device_inplace", [](Encoder& self, MemoryPoolHandleArgument pool) {
            self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)

        .def("scale_up", [](const Encoder& self, const py::array_t<T>& values, std::optional<ParmsID> parms_id, Plaintext& p, MemoryPoolHandleArgument pool) {
            self.scale_up(get_vector_from_buffer(values), parms_id, p, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("scale_up_new", [](const Encoder& self, const py::array_t<T>& values, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            return self.scale_up_new(get_vector_from_buffer(values), parms_id, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("centralize", [](const Encoder& self, const py::array_t<T>& values, std::optional<ParmsID> parms_id, Plaintext& p, MemoryPoolHandleArgument pool) {
            self.centralize(get_vector_from_buffer(values), parms_id, p, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("centralize_new", [](const Encoder& self, const py::array_t<T>& values, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            return self.centralize_new(get_vector_from_buffer(values), parms_id, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)
        .def("scale_down_new", [](const Encoder& self, const Plaintext& p, MemoryPoolHandleArgument pool) {
            return get_buffer_from_vector(self.scale_down_new(p, nullopt_default_pool(pool)));
        }, py::arg("values"), MEMORY_POOL_ARGUMENT)
    ;
}

void register_polynomial_encoder_ring2k(pybind11::module& m) {
    register_class_polynomial_encoder_ring2k<uint32_t>(m, "PolynomialEncoderRing2k32");
    register_class_polynomial_encoder_ring2k<uint64_t>(m, "PolynomialEncoderRing2k64");
}