#include "header.h"

void register_ckks_encoder(pybind11::module& m) {


    py::class_<CKKSEncoder>(m, "CKKSEncoder")
        .def(py::init<HeContextPointer>())
        .def("context", &CKKSEncoder::context)
        .def("on_device", &CKKSEncoder::on_device)

        // to_device_inplace(pool)
        .def("to_device_inplace", [](CKKSEncoder& self, MemoryPoolHandleArgument pool) {
            self.to_device_inplace(nullopt_default_pool(pool));
        }, MEMORY_POOL_ARGUMENT)

        .def("slot_count", &CKKSEncoder::slot_count)
        .def("polynomial_modulus_degree", &CKKSEncoder::polynomial_modulus_degree)
        .def("poly_modulus_degree", &CKKSEncoder::polynomial_modulus_degree)

        .def("encode_complex64_simd", [](
            const CKKSEncoder& self, const py::array_t<std::complex<double>>& values, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p,
            MemoryPoolHandleArgument pool) {
            self.encode_complex64_simd(get_vector_from_buffer(values), parms_id, scale, p, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_complex64_simd_new", [](
            const CKKSEncoder& self, const py::array_t<std::complex<double>>& values, 
            std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandleArgument pool) {
            return self.encode_complex64_simd_new(get_vector_from_buffer(values), parms_id, scale, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"), MEMORY_POOL_ARGUMENT)

        .def("encode_float64_single", [](
            const CKKSEncoder& self, double value, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p,
            MemoryPoolHandleArgument pool) {
            self.encode_float64_single(value, parms_id, scale, p, nullopt_default_pool(pool));
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_float64_single_new", [](
            const CKKSEncoder& self, double value, 
            std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandleArgument pool) {
            return self.encode_float64_single_new(value, parms_id, scale, nullopt_default_pool(pool));
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"), MEMORY_POOL_ARGUMENT)

        .def("encode_float64_polynomial", [](
            const CKKSEncoder& self, const py::array_t<double>& values, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p,
            MemoryPoolHandleArgument pool) {
            self.encode_float64_polynomial(get_vector_from_buffer(values), parms_id, scale, p, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_float64_polynomial_new", [](
            const CKKSEncoder& self, const py::array_t<double>& values, 
            std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandleArgument pool) {
            return self.encode_float64_polynomial_new(get_vector_from_buffer(values), parms_id, scale, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), py::arg("scale"), MEMORY_POOL_ARGUMENT)
        
        .def("encode_complex64_single", [](
            const CKKSEncoder& self, std::complex<double> value, 
            std::optional<ParmsID> parms_id, double scale, Plaintext& p,
            MemoryPoolHandleArgument pool) {
            self.encode_complex64_single(value, parms_id, scale, p, nullopt_default_pool(pool));
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_complex64_single_new", [](
            const CKKSEncoder& self, std::complex<double> value, 
            std::optional<ParmsID> parms_id, double scale,
            MemoryPoolHandleArgument pool) {
            return self.encode_complex64_single_new(value, parms_id, scale, nullopt_default_pool(pool));
        }, py::arg("value"), py::arg("parms_id"), py::arg("scale"), MEMORY_POOL_ARGUMENT)

        .def("encode_integer64_single", [](
            const CKKSEncoder& self, int64_t value, 
            std::optional<ParmsID> parms_id, Plaintext& p,
            MemoryPoolHandleArgument pool) {
            self.encode_integer64_single(value, parms_id, p, nullopt_default_pool(pool));
        }, py::arg("value"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_integer64_single_new", [](
            const CKKSEncoder& self, int64_t value, 
            std::optional<ParmsID> parms_id,
            MemoryPoolHandleArgument pool) {
            return self.encode_integer64_single_new(value, parms_id, nullopt_default_pool(pool));
        }, py::arg("value"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)

        .def("encode_integer64_polynomial", [](
            const CKKSEncoder& self, const py::array_t<int64_t>& values, 
            std::optional<ParmsID> parms_id, Plaintext& p,
            MemoryPoolHandleArgument pool) {
            self.encode_integer64_polynomial(get_vector_from_buffer(values), parms_id, p, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), py::arg("destination"), MEMORY_POOL_ARGUMENT)
        .def("encode_integer64_polynomial_new", [](
            const CKKSEncoder& self, const py::array_t<int64_t>& values, 
            std::optional<ParmsID> parms_id,
            MemoryPoolHandleArgument pool) {
            return self.encode_integer64_polynomial_new(get_vector_from_buffer(values), parms_id, nullopt_default_pool(pool));
        }, py::arg("values"), py::arg("parms_id"), MEMORY_POOL_ARGUMENT)

        .def("decode_complex64_simd_new", [](const CKKSEncoder& self, const Plaintext& p, MemoryPoolHandleArgument pool) {
            return get_buffer_from_vector(self.decode_complex64_simd_new(p, nullopt_default_pool(pool)));
        }, py::arg("source"), MEMORY_POOL_ARGUMENT)
        .def("decode_float64_polynomial_new", [](const CKKSEncoder& self, const Plaintext& p, MemoryPoolHandleArgument pool) {
            return get_buffer_from_vector(self.decode_float64_polynomial_new(p, nullopt_default_pool(pool)));
        }, py::arg("source"), MEMORY_POOL_ARGUMENT)
    ;

}