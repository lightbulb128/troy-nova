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

        inline KSwitchKeys clone() const {
            KSwitchKeys result;
            result.parms_id_ = parms_id_;
            result.keys.resize(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                result.keys[i].resize(keys[i].size());
                for (size_t j = 0; j < keys[i].size(); j++) {
                    result.keys[i][j] = keys[i][j].clone();
                }
            }
            return result;
        }

        inline void to_device_inplace() {
            for (auto& v : keys) {
                for (auto& key : v) {
                    key.to_device_inplace();
                }
            }
        }

        inline void to_host_inplace() {
            for (auto& v : keys) {
                for (auto& key : v) {
                    key.to_host_inplace();
                }
            }
        }

        inline KSwitchKeys to_device() const {
            KSwitchKeys result = clone();
            result.to_device_inplace();
            return result;
        }

        inline KSwitchKeys to_host() const {
            KSwitchKeys result = clone();
            result.to_host_inplace();
            return result;
        }

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
        
        inline bool contains_seed() const {
            // iterate over all keys
            bool flag = false;
            for (auto& v : keys) {
                for (auto& key : v) {
                    if (key.contains_seed()) {
                        flag = true;
                        break;
                    }
                }
                if (flag) break;
            }
            return flag;
        }
        inline void expand_seed(HeContextPointer context) {
            // iterate over all keys
            bool flag = false;
            for (auto& v : keys) {
                for (auto& key : v) {
                    if (key.contains_seed()) {
                        key.expand_seed(context);
                        flag = true;
                    }
                }
            }
            if (!flag) {
                throw std::runtime_error("[KSwitchKeys::expand_seed] KSwitchKeys does not contain seed.");
            }
        }

        void save(std::ostream& stream, HeContextPointer context) const;
        void load(std::istream& stream, HeContextPointer context);
        inline static KSwitchKeys load_new(std::istream& stream, HeContextPointer context) {
            KSwitchKeys result;
            result.load(stream, context);
            return result;
        }
        size_t serialized_size(HeContextPointer context) const;

    };

    class RelinKeys {
    private:
        KSwitchKeys keys;
    
    public:
        inline RelinKeys() {}
        inline RelinKeys(KSwitchKeys&& keys): keys(std::move(keys)) {}

        inline RelinKeys clone() const {
            RelinKeys result;
            result.keys = keys.clone();
            return result;
        }
        inline bool on_device() const {
            return keys.on_device();
        }
        inline void to_device_inplace() {
            keys.to_device_inplace();
        }
        inline void to_host_inplace() {
            keys.to_host_inplace();
        }
        inline RelinKeys to_device() const {
            RelinKeys result = clone();
            result.to_device_inplace();
            return result;
        }
        inline RelinKeys to_host() const {
            RelinKeys result = clone();
            result.to_host_inplace();
            return result;
        }

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

        inline bool contains_seed() const {
            return keys.contains_seed();
        }
        inline void expand_seed(HeContextPointer context) {
            keys.expand_seed(context);
        }
        
        inline void save(std::ostream& stream, HeContextPointer context) const {
            keys.save(stream, context);
        }
        inline void load(std::istream& stream, HeContextPointer context) {
            keys.load(stream, context);
        }
        inline static RelinKeys load_new(std::istream& stream, HeContextPointer context) {
            RelinKeys result;
            result.load(stream, context);
            return result;
        }
        size_t serialized_size(HeContextPointer context) const {
            return keys.serialized_size(context);
        }
    };

    class GaloisKeys {
    private:
        KSwitchKeys keys;
    
    public:
        inline GaloisKeys() {}
        inline GaloisKeys(KSwitchKeys&& keys): keys(std::move(keys)) {}

        inline GaloisKeys clone() const {
            GaloisKeys result;
            result.keys = keys.clone();
            return result;
        }
        inline bool on_device() const {
            return keys.on_device();
        }
        inline void to_device_inplace() {
            keys.to_device_inplace();
        }
        inline void to_host_inplace() {
            keys.to_host_inplace();
        }
        inline GaloisKeys to_device() const {
            GaloisKeys result = clone();
            result.to_device_inplace();
            return result;
        }
        inline GaloisKeys to_host() const {
            GaloisKeys result = clone();
            result.to_host_inplace();
            return result;
        }
        
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

        inline bool contains_seed() const {
            return keys.contains_seed();
        }
        inline void expand_seed(HeContextPointer context) {
            keys.expand_seed(context);
        }

        inline void save(std::ostream& stream, HeContextPointer context) const {
            keys.save(stream, context);
        }
        inline void load(std::istream& stream, HeContextPointer context) {
            keys.load(stream, context);
        }
        inline static GaloisKeys load_new(std::istream& stream, HeContextPointer context) {
            GaloisKeys result;
            result.load(stream, context);
            return result;
        }
        size_t serialized_size(HeContextPointer context) const {
            return keys.serialized_size(context);
        }
    };
}