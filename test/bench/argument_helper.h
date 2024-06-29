#pragma once

#include <string>
#include <vector>

namespace bench {

    inline std::string bool_to_string(bool b) {
        return b ? "true" : "false";
    }
    inline std::string concat_by_comma(const std::vector<std::string>& v, bool none_if_none = true) {
        if (v.size() == 0) return none_if_none ? "none" : "";
        std::string s;
        for (size_t i = 0; i < v.size(); i++) {
            s += v[i];
            if (i != v.size() - 1) s += ", ";
        }
        return s;
    }
    inline std::string list_usize_to_string(const std::vector<size_t>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); i++) {
            s += std::to_string(v[i]);
            if (i != v.size() - 1) s += ", ";
        }
        return s + "]";
    }

}