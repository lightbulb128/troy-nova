#pragma once
#include <memory>
#include "cuda_runtime.h"
#include "troy/troy.cuh"
#include "rust/cxx.h"

namespace troy_rust {

    typedef troy::SchemeType SchemeType;

    typedef troy::SecurityLevel SecurityLevel;

    inline size_t device_count() {
        return troy::utils::device_count();
    }

    inline rust::String scheme_type_to_string(SchemeType scheme) {
        switch (scheme) {
            case SchemeType::Nil: return "Nil";
            case SchemeType::BFV: return "BFV";
            case SchemeType::CKKS: return "CKKS";
            case SchemeType::BGV: return "BGV";
            default: return "Unknown";
        }
    }

    inline rust::String security_level_to_string(SecurityLevel sec) {
        switch (sec) {
            case SecurityLevel::Nil: return "Nil";
            case SecurityLevel::Classical128: return "Classical128";
            case SecurityLevel::Classical192: return "Classical192";
            case SecurityLevel::Classical256: return "Classical256";
            default: return "Unknown";
        }
    }

}