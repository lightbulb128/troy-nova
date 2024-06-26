#pragma once

#include <string>
#include <cstdarg>
#include <optional>
#include <vector>

class ArgumentParser {
    private:
        std::vector<std::string> raw;
    public:
        ArgumentParser(int argc, char** argv);
        std::optional<std::string> get_string(const std::string& name) const;
        
        template <typename T>
        std::optional<T> get_int(const std::string& name) const;

        template <typename T>
        std::optional<T> get_uint(const std::string& name) const;
        
        template <typename T>
        std::optional<T> get_float(const std::string& name) const;

        std::optional<bool> get_bool_store_true(const std::string& name) const;
        std::optional<std::vector<std::string>> get_list(const std::string& name) const;

        template <typename T, typename F>
        std::optional<std::vector<T>> get_list_as(const std::string& name, F f) const {
            auto list = this->get_list(name);
            if (!list) {
                return std::nullopt;
            }
            std::vector<T> result;
            for (const auto& item : *list) {
                try {
                    result.push_back(f(item));
                } catch (...) {
                    return std::nullopt;
                }
            }
            return result;
        }

        template <typename T>
        std::optional<std::vector<T>> get_int_list(const std::string& name) const {
            return this->get_list_as<T>(name, [](const std::string& s) {
                return static_cast<T>(std::stoll(s));
            });
        }

        template <typename T>
        std::optional<std::vector<T>> get_uint_list(const std::string& name) const {
            return this->get_list_as<T>(name, [](const std::string& s) {
                return static_cast<T>(std::stoull(s));
            });
        }

        template <typename T>
        std::optional<std::vector<T>> get_float_list(const std::string& name) const {
            return this->get_list_as<T>(name, [](const std::string& s) {
                return static_cast<T>(std::stod(s));
            });
        }
};