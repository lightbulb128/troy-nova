#pragma once
#include "key.cuh"

namespace troy {

    class KSwitchKeys {

    private:
        ParmsID parms_id_;
        std::vector<std::vector<PublicKey>> keys;

    public: 
        inline KSwitchKeys(): parms_id_(parms_id_zero) {}
        inline KSwitchKeys(ParmsID parms_id, std::vector<std::vector<PublicKey>> keys): parms_id_(parms_id), keys(keys) {}

        inline const ParmsID& parms_id() const {
            return parms_id_;
        }

        inline ParmsID& parms_id() {
            return parms_id_;
        }

        inline const std::vector<std::vector<PublicKey>>& data() const {
            return keys;
        }

        inline std::vector<std::vector<PublicKey>>& data() {
            return keys;
        }

        inline bool on_device() const {
            // find first non-empty vector
            for (auto& v : keys) {
                if (v.size() > 0) {
                    return v[0].on_device();
                }
            }
            throw std::runtime_error("[KSwitchKeys::on_device] KSwitchKeys is empty.");
        }

        inline size_t key_count() const {
            // find non-empty vector count
            size_t count = 0;
            for (auto& v : keys) {
                if (v.size() > 0) {
                    count++;
                }
            }
            return count;
        }

        inline const std::vector<PublicKey>& operator[](size_t index) const {
            return keys[index];
        }

        inline std::vector<PublicKey>& operator[](size_t index) {
            return keys[index];
        }

    };

    class RelinKeys {
    private:
        KSwitchKeys keys;
    
    public:
        inline RelinKeys() {}
        inline RelinKeys(KSwitchKeys&& keys): keys(std::move(keys)) {}
        
        static inline size_t get_index(size_t key_power) {
            if (key_power < 2) {
                throw std::invalid_argument("[RelinKeys::get_index] key_power must be at least 2.");
            }
            return key_power - 2;
        }

        inline bool has_key(size_t key_power) const {
            size_t index = get_index(key_power);
            return index < keys.data().size() && keys.data()[index].size() > 0;
        }

        inline const std::vector<PublicKey>& key(size_t key_power) const {
            size_t index = get_index(key_power);
            if (!has_key(key_power)) {
                throw std::invalid_argument("[RelinKeys::key] key_power is not valid.");
            }
            return keys.data()[index];
        }

        inline const ParmsID& parms_id() const {
            return keys.parms_id();
        }

        inline ParmsID& parms_id() {
            return keys.parms_id();
        }

        inline const KSwitchKeys& as_kswitch_keys() const {
            return keys;
        }

        inline KSwitchKeys& as_kswitch_keys() {
            return keys;
        }

    };

    class GaloisKeys {
    private:
        KSwitchKeys keys;
    
    public:
        inline GaloisKeys() {}
        inline GaloisKeys(KSwitchKeys&& keys): keys(std::move(keys)) {}
        
        static inline size_t get_index(size_t galois_element) {
            return utils::GaloisTool::get_index_from_element(galois_element);
        }

        inline bool has_key(size_t galois_element) const {
            size_t index = get_index(galois_element);
            return index < keys.data().size() && keys.data()[index].size() > 0;
        }

        inline const std::vector<PublicKey>& key(size_t galois_element) const {
            size_t index = get_index(galois_element);
            if (!has_key(galois_element)) {
                throw std::invalid_argument("[RelinKeys::key] key_power is not valid.");
            }
            return keys.data()[index];
        }

        inline const ParmsID& parms_id() const {
            return keys.parms_id();
        }

        inline ParmsID& parms_id() {
            return keys.parms_id();
        }

        inline const KSwitchKeys& as_kswitch_keys() const {
            return keys;
        }

        inline KSwitchKeys& as_kswitch_keys() {
            return keys;
        }

    };
}