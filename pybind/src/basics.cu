#include "header.h"

void register_basics(pybind11::module& m) {

    py::enum_<SchemeType>(m, "SchemeType")
        .value("Nil", SchemeType::Nil)
        .value("BFV", SchemeType::BFV)
        .value("CKKS", SchemeType::CKKS)
        .value("BGV", SchemeType::BGV)
    ;

    py::enum_<SecurityLevel>(m, "SecurityLevel")
        .value("Nil", SecurityLevel::Nil)
        .value("Classical128", SecurityLevel::Classical128)
        .value("Classical192", SecurityLevel::Classical192)
        .value("Classical256", SecurityLevel::Classical256)
    ;

    py::enum_<CompressionMode>(m, "CompressionMode")
        .value("Nil", CompressionMode::Nil)
        .value("Zstd", CompressionMode::Zstd)
    ;

    py::class_<MemoryPool, std::shared_ptr<MemoryPool>>(m, "MemoryPool")
        .def_static("global_pool", &MemoryPool::GlobalPool)
        .def(py::init([](size_t device){
            return MemoryPool::create(device);
        }), py::arg("device_index") = 0)
        .def("destroy", &MemoryPool::destroy);
    ;
}