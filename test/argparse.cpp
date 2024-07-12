#include "argparse.h"
#include <cstdint>
#include <string>

using int128_t = __int128_t;
using uint128_t = __uint128_t;

ArgumentParser::ArgumentParser(int argc, char** argv) {
    std::vector<std::string> raw;
    for (int i = 0; i < argc; i++) {
        raw.push_back(argv[i]);
    }
    this->raw = std::move(raw);
}

static std::string strip_quotes(const std::string& str) {
    if (str.size() >= 2 && str[0] == '"' && str[str.size() - 1] == '"') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

std::optional<std::string> ArgumentParser::get_string(const std::string& name) const {
    for (size_t i = 0; i < this->raw.size(); i++) {
        // if it starts with the name followed by a "=", obtain the following string
        if (this->raw[i].find(name + "=") == 0) {
            return strip_quotes(this->raw[i].substr(name.size() + 1));
        }
        // if it is exactly the name, then get the next element
        if (this->raw[i] == name && i + 1 < this->raw.size()) {
            return strip_quotes(this->raw[i + 1]);
        }
    }
    return std::nullopt;
}

template <typename T>
std::optional<T> ArgumentParser::get_int(const std::string& name) const {
    auto str = this->get_string(name);
    if (!str) {
        return std::nullopt;
    }
    try {
        return static_cast<T>(std::stoll(*str));
    } catch (...) {
        return std::nullopt;
    }
}

template std::optional<int8_t> ArgumentParser::get_int(const std::string& name) const;
template std::optional<int16_t> ArgumentParser::get_int(const std::string& name) const;
template std::optional<int32_t> ArgumentParser::get_int(const std::string& name) const;
template std::optional<int64_t> ArgumentParser::get_int(const std::string& name) const;
template std::optional<int128_t> ArgumentParser::get_int(const std::string& name) const;


template <typename T>
std::optional<T> ArgumentParser::get_uint(const std::string& name) const {
    auto str = this->get_string(name);
    if (!str) {
        return std::nullopt;
    }
    try {
        return static_cast<T>(std::stoull(*str));
    } catch (...) {
        return std::nullopt;
    }
}

template std::optional<uint8_t> ArgumentParser::get_uint(const std::string& name) const;
template std::optional<uint16_t> ArgumentParser::get_uint(const std::string& name) const;
template std::optional<uint32_t> ArgumentParser::get_uint(const std::string& name) const;
template std::optional<uint64_t> ArgumentParser::get_uint(const std::string& name) const;
template std::optional<uint128_t> ArgumentParser::get_uint(const std::string& name) const;

template <typename T>
std::optional<T> ArgumentParser::get_float(const std::string& name) const {
    auto str = this->get_string(name);
    if (!str) {
        return std::nullopt;
    }
    try {
        return static_cast<T>(std::stod(*str));
    } catch (...) {
        return std::nullopt;
    }
}

template std::optional<float> ArgumentParser::get_float(const std::string& name) const;
template std::optional<double> ArgumentParser::get_float(const std::string& name) const;

std::optional<bool> ArgumentParser::get_bool_store_true(const std::string& name) const {
    for (size_t i = 0; i < this->raw.size(); i++) {
        if (this->raw[i] == name) {
            return true;
        }
    }
    return std::nullopt;
}

std::optional<std::vector<std::string>> ArgumentParser::get_list(const std::string& name) const {
    // get the string element, then split it by commas
    auto str = this->get_string(name);
    if (!str) {
        return std::nullopt;
    }
    std::vector<std::string> list;
    std::string current;
    for (char c : *str) {
        if (c == ',') {
            list.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    list.push_back(current);
    return list;
}