#include "header.h"

void register_modulus(pybind11::module& m) {

    py::class_<Modulus>(m, "Modulus")
        .def(py::init<uint64_t>(), py::arg("value") = 0)
        .def("is_prime", &Modulus::is_prime)
        .def("is_zero", &Modulus::is_zero)
        .def("value", &Modulus::value)
        .def("bit_count", &Modulus::bit_count)
        .def("__str__", [](const Modulus& m){ return to_string(m); })
        .def("__repr__", [](const Modulus& m){ return to_string(m); })
        .def("reduce", &Modulus::reduce, py::arg("input_uint64"))
        .def("reduce_mul", [](const Modulus& m, uint64_t a, uint64_t b){
            return m.reduce_mul_uint64(a, b);
        }, "Output the reduction of a * b modulo the value of the Modulus.", py::arg("a"), py::arg("b"))
    ;

    py::class_<CoeffModulus>(m, "CoeffModulus")
        .def_static("max_bit_count", &CoeffModulus::max_bit_count,
            py::arg("poly_modulus_degree"), py::arg_v("sec_level", SecurityLevel::Classical128, "SecurityLevel.Classical128"))
        .def_static("bfv_default", [](size_t poly_modulus_degree, SecurityLevel sec){
            return get_buffer_from_array(CoeffModulus::bfv_default(poly_modulus_degree, sec));
        }, py::arg("poly_modulus_degree"), py::arg_v("sec_level", SecurityLevel::Classical128, "SecurityLevel.Classical128"))
        .def_static("create", [](size_t poly_modulus_degree, const std::vector<int>& bit_sizes){
            vector<size_t> bit_sizes_copy(bit_sizes.size(), 0);
            for (size_t i = 0; i < bit_sizes.size(); i++) {
                bit_sizes_copy[i] = bit_sizes[i];
            }
            return CoeffModulus::create(poly_modulus_degree, bit_sizes_copy).to_vector();
        }, py::arg("poly_modulus_degree"), py::arg("bit_sizes"))
    ;

    py::class_<PlainModulus>(m, "PlainModulus")
        .def_static("batching", &PlainModulus::batching,
            py::arg("poly_modulus_degree"), py::arg_v("sec_level", SecurityLevel::Classical128, "SecurityLevel.Classical128"))
        .def_static("batching_multiple", [](size_t poly_modulus_degree, const std::vector<int>& bit_sizes){
            vector<size_t> bit_sizes_copy(bit_sizes.size(), 0);
            for (size_t i = 0; i < bit_sizes.size(); i++) {
                bit_sizes_copy[i] = bit_sizes[i];
            }
            return PlainModulus::batching_multiple(poly_modulus_degree, bit_sizes_copy).to_vector();
        }, py::arg("poly_modulus_degree"), py::arg("bit_sizes"))
    ;
    
}