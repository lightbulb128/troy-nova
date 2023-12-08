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

}