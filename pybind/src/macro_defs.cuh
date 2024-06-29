#pragma once
#include "header.cuh"

#define def_arg0_pool_const(T, NAME)                              \
    def(#NAME, [](const T& self, MemoryPoolHandleArgument pool) { \
        return self.NAME(nullopt_default_pool(pool));             \
    }, MEMORY_POOL_ARGUMENT)

#define def_arg0_pool(T, NAME)                                    \
    def(#NAME, [](T& self, MemoryPoolHandleArgument pool) {       \
        return self.NAME(nullopt_default_pool(pool));             \
    }, MEMORY_POOL_ARGUMENT)

#define def_arg1_pool_const(T, NAME, A1T, A1)                             \
    def(#NAME, [](const T& self, A1T A1, MemoryPoolHandleArgument pool) { \
        return self.NAME(A1, nullopt_default_pool(pool));               \
    }, py::arg(#A1), MEMORY_POOL_ARGUMENT)

#define def_arg1_pool(T, NAME, A1T, A1)                                 \
    def(#NAME, [](T& self, A1T A1, MemoryPoolHandleArgument pool) {     \
        return self.NAME(A1, nullopt_default_pool(pool));             \
    }, py::arg(#A1), MEMORY_POOL_ARGUMENT)

#define def_arg2_pool_const(T, NAME, A1T, A1, A2T, A2)                            \
    def(#NAME, [](const T& self, A1T A1, A2T A2, MemoryPoolHandleArgument pool) { \
        return self.NAME(A1, A2, nullopt_default_pool(pool));                 \
    }, py::arg(#A1), py::arg(#A2), MEMORY_POOL_ARGUMENT)

#define def_arg2_pool(T, NAME, A1T, A1, A2T, A2)                             \
    def(#NAME, [](T& self, A1T A1, A2T A2, MemoryPoolHandleArgument pool) {  \
        return self.NAME(A1, A2, nullopt_default_pool(pool));            \
    }, py::arg(#A1), py::arg(#A2), MEMORY_POOL_ARGUMENT)

