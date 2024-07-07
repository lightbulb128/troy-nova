#include "header.h"

void register_batch_encoder(pybind11::module& m) {
    
    py::class_<BatchEncoder>(m, "BatchEncoder")
        .def(py::init<HeContextPointer>())
        .def("context", &BatchEncoder::context)
        .def("on_device", &BatchEncoder::on_device)
        
        // to_device_inplace(pool)
        .def("to_device_inplace", [](BatchEncoder& self, MemoryPoolHandleArgument pool){
            return self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)

        .def("slot_count", &BatchEncoder::slot_count)
        .def("row_count", &BatchEncoder::row_count)
        .def("column_count", &BatchEncoder::column_count)
        .def("simd_encoding_supported", &BatchEncoder::simd_encoding_supported)

        // encode_simd
        .def("encode_simd", [](const BatchEncoder& self, const py::array_t<uint64_t>& values, Plaintext& p, MemoryPoolHandleArgument pool) {
            self.encode(get_vector_from_buffer(values), p, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_simd_new", [](const BatchEncoder& self, const py::array_t<uint64_t>& values, MemoryPoolHandleArgument pool) {
            return self.encode_new(get_vector_from_buffer(values), nullopt_default_pool(pool));
        }, py::arg("values"), MEMORY_POOL_ARGUMENT)

        // encode_polynomial
        .def("encode_polynomial", [](const BatchEncoder& self, const py::array_t<uint64_t>& values, Plaintext& p, MemoryPoolHandleArgument pool) {
            self.encode_polynomial(get_vector_from_buffer(values), p, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_polynomial_new", [](const BatchEncoder& self, const py::array_t<uint64_t>& values, MemoryPoolHandleArgument pool) {
            return self.encode_polynomial_new(get_vector_from_buffer(values), nullopt_default_pool(pool));
        }, py::arg("values"), MEMORY_POOL_ARGUMENT)

        // decode_simd
        .def("decode_simd_new", [](const BatchEncoder& self, const Plaintext& p, MemoryPoolHandleArgument pool) {
            return get_buffer_from_vector(self.decode_new(p, nullopt_default_pool(pool)));
        }, py::arg("source"), MEMORY_POOL_ARGUMENT)

        // decode_polynomial
        .def("decode_polynomial_new", [](const BatchEncoder& self, const Plaintext& p) {
            return get_buffer_from_vector(self.decode_polynomial_new(p));
        }, py::arg("source"))

        // scale_up
        .def("scale_up", [](const BatchEncoder& self, const Plaintext& p, Plaintext& destination, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            self.scale_up(p, destination, parms_id, nullopt_default_pool(pool));
        }, py::arg("source"), py::arg("destination"), OPTIONAL_PARMS_ID_ARGUMENT, MEMORY_POOL_ARGUMENT)
        .def("scale_up_inplace", [](BatchEncoder& self, Plaintext& p, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            self.scale_up_inplace(p, parms_id, nullopt_default_pool(pool));
        }, py::arg("source"), OPTIONAL_PARMS_ID_ARGUMENT, MEMORY_POOL_ARGUMENT)
        .def("scale_up_new", [](const BatchEncoder& self, const Plaintext& p, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            return self.scale_up_new(p, parms_id, nullopt_default_pool(pool));
        }, py::arg("source"), OPTIONAL_PARMS_ID_ARGUMENT, MEMORY_POOL_ARGUMENT)

        // centralize(same arguments as scale_up)
        .def("centralize", [](const BatchEncoder& self, const Plaintext& p, Plaintext& destination, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            self.centralize(p, destination, parms_id, nullopt_default_pool(pool));
        }, py::arg("source"), py::arg("destination"), OPTIONAL_PARMS_ID_ARGUMENT, MEMORY_POOL_ARGUMENT)
        .def("centralize_inplace", [](BatchEncoder& self, Plaintext& p, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            self.centralize_inplace(p, parms_id, nullopt_default_pool(pool));
        }, py::arg("source"), OPTIONAL_PARMS_ID_ARGUMENT, MEMORY_POOL_ARGUMENT)
        .def("centralize_new", [](const BatchEncoder& self, const Plaintext& p, std::optional<ParmsID> parms_id, MemoryPoolHandleArgument pool) {
            return self.centralize_new(p, parms_id, nullopt_default_pool(pool));
        }, py::arg("source"), OPTIONAL_PARMS_ID_ARGUMENT, MEMORY_POOL_ARGUMENT)

        // scale_down(const Plaintext& plain, Plaintext& destination, pool)
        .def("scale_down", [](const BatchEncoder& self, const Plaintext& p, Plaintext& destination, MemoryPoolHandleArgument pool) {
            self.scale_down(p, destination, nullopt_default_pool(pool));
        }, py::arg("source"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("scale_down_inplace", [](BatchEncoder& self, Plaintext& p, MemoryPoolHandleArgument pool) {
            self.scale_down_inplace(p, nullopt_default_pool(pool));
        }, py::arg("source"), MEMORY_POOL_ARGUMENT)
        .def("scale_down_new", [](const BatchEncoder& self, const Plaintext& p, MemoryPoolHandleArgument pool) {
            return self.scale_down_new(p, nullopt_default_pool(pool));
        }, py::arg("source"), MEMORY_POOL_ARGUMENT)
    ;

}