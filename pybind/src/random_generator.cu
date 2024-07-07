#include "header.h"

void register_random_generator(pybind11::module& m) {
    
    py::class_<utils::RandomGenerator>(m, "RandomGenerator")
        .def(py::init<>())
        .def(py::init<uint64_t>())
        .def("reset_seed", &utils::RandomGenerator::reset_seed)
        .def("sample_uint64", &utils::RandomGenerator::sample_uint64)
    ;
    
}