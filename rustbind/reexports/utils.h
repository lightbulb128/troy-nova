#pragma once
#include <memory>
#include <sstream>

namespace troy_rust {
    using uint128_t = __uint128_t;
}

namespace troy_rust::utils {

    template <typename T>
    std::string to_string(const T& object) {
        std::ostringstream ss;
        ss << object;
        return ss.str();
    }

}