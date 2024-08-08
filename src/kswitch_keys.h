#pragma once
#include "key.h"
#include "utils/memory_pool.h"

namespace troy {

    class KSwitchKeys {

    private:
        ParmsID parms_id_;
        std::vector<std::vector<PublicKey>> keys;
        std::vector<utils::Array<const uint64_t*>> key_data_ptrs;

    public: 
    
        KSwitchKeys(const KSwitchKeys& clone) {
            this->keys = clone.keys;
            this->parms_id_ = clone.parms_id_;
            this->key_data_ptrs.clear();
            build_key_data_ptrs(clone.pool());
        }

        KSwitchKeys& operator=(const KSwitchKeys& assign) {
            if (this == &assign) {
                return *this;
            }
            this->keys = assign.keys;
            this->parms_id_ = assign.parms_id_;
            this->key_data_ptrs.clear();
            build_key_data_ptrs(assign.pool());
            return *this;
        }

        inline void build_key_data_ptrs(MemoryPoolHandle pool) {
            key_data_ptrs.resize(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                key_data_ptrs[i] = utils::Array<const uint64_t*>(keys[i].size(), false, pool);
                for (size_t j = 0; j < keys[i].size(); j++) {
                    key_data_ptrs[i][j] = keys[i][j].as_ciphertext().data().raw_pointer();
                }
            }
            if (on_device()) {
                for (size_t i = 0; i < keys.size(); i++) {
                    key_data_ptrs[i].to_device_inplace(pool);
                }
            }
        }

        inline utils::ConstSlice<const uint64_t*> get_data_ptrs(size_t index) const {
            if (index >= key_data_ptrs.size()) {
                throw std::invalid_argument("[KSwitchKeys::get_data_ptrs] index is out of range.");
            }
            return key_data_ptrs[index].const_reference();
        }
    
        inline MemoryPoolHandle pool() const { 
            // find first non-empty vector
            for (auto& v : keys) {
                if (v.size() > 0) {
                    return v[0].pool();
                }
            }
            throw std::runtime_error("[KSwitchKeys::pool] KSwitchKeys is empty.");
        }
        inline size_t device_index() const { 
            // find first non-empty vector
            for (auto& v : keys) {
                if (v.size() > 0) {
                    return v[0].device_index();
                }
            }
            throw std::runtime_error("[KSwitchKeys::device_index] KSwitchKeys is empty.");
        }

        inline KSwitchKeys(): parms_id_(parms_id_zero) {}
        inline KSwitchKeys(ParmsID parms_id, std::vector<std::vector<PublicKey>> keys, MemoryPoolHandle pool): parms_id_(parms_id), keys(keys) {
            build_key_data_ptrs(pool);
        }

        inline KSwitchKeys clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            KSwitchKeys result;
            result.parms_id_ = parms_id_;
            result.keys.resize(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                result.keys[i].resize(keys[i].size());
                for (size_t j = 0; j < keys[i].size(); j++) {
                    result.keys[i][j] = keys[i][j].clone(pool);
                }
            }
            result.build_key_data_ptrs(pool);
            return result;
        }

        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            for (auto& v : keys) {
                for (auto& key : v) {
                    key.to_device_inplace(pool);
                }
            }
            for (auto& ptrs : key_data_ptrs) {
                ptrs.to_device_inplace(pool);
            }
        }

        inline void to_host_inplace() {
            for (auto& v : keys) {
                for (auto& key : v) {
                    key.to_host_inplace();
                }
            }
            for (auto& ptrs : key_data_ptrs) {
                ptrs.to_host_inplace();
            }
        }

        inline KSwitchKeys to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            KSwitchKeys result = clone(pool);
            result.to_device_inplace(pool);
            return result;
        }

        inline KSwitchKeys to_host() const {
            KSwitchKeys result = clone(pool());
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

        size_t save(std::ostream& stream, HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const;
        void load(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool());
        inline static KSwitchKeys load_new(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            KSwitchKeys result;
            result.load(stream, context, pool);
            return result;
        }
        size_t serialized_size_upperbound(HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const;

    };

    class RelinKeys {
    private:
        KSwitchKeys keys;
    
    public:
        inline MemoryPoolHandle pool() const { return keys.pool(); }
        inline size_t device_index() const { return keys.device_index(); }

        inline RelinKeys() {}
        inline RelinKeys(KSwitchKeys&& keys): keys(std::move(keys)) {}
        RelinKeys(const KSwitchKeys& keys) = delete;

        inline RelinKeys clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            RelinKeys result;
            result.keys = keys.clone(pool);
            return result;
        }
        inline bool on_device() const {
            return keys.on_device();
        }
        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            keys.to_device_inplace(pool);
        }
        inline void to_host_inplace() {
            keys.to_host_inplace();
        }
        inline RelinKeys to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            RelinKeys result = clone(pool);
            result.to_device_inplace(pool);
            return result;
        }
        inline RelinKeys to_host() const {
            RelinKeys result = clone(pool());
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
        
        inline size_t save(std::ostream& stream, HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const {
            return keys.save(stream, context, mode);
        }
        inline void load(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            keys.load(stream, context, pool);
        }
        inline static RelinKeys load_new(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            RelinKeys result;
            result.load(stream, context, pool);
            return result;
        }
        size_t serialized_size_upperbound(HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const {
            return keys.serialized_size_upperbound(context, mode);
        }
    };

    class GaloisKeys {
    private:
        KSwitchKeys keys;
    
    public:
        inline MemoryPoolHandle pool() const { return keys.pool(); }
        inline size_t device_index() const { return keys.device_index(); }

        inline GaloisKeys() {}
        inline GaloisKeys(KSwitchKeys&& keys): keys(std::move(keys)) {}
        GaloisKeys(const KSwitchKeys& keys) = delete;

        inline GaloisKeys clone(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            GaloisKeys result;
            result.keys = keys.clone(pool);
            return result;
        }
        inline bool on_device() const {
            return keys.on_device();
        }
        inline void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            keys.to_device_inplace(pool);
        }
        inline void to_host_inplace() {
            keys.to_host_inplace();
        }
        inline GaloisKeys to_device(MemoryPoolHandle pool = MemoryPool::GlobalPool()) const {
            GaloisKeys result = clone(pool);
            result.to_device_inplace(pool);
            return result;
        }
        inline GaloisKeys to_host() const {
            GaloisKeys result = clone(pool());
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

        inline size_t save(std::ostream& stream, HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const {
            return keys.save(stream, context, mode);
        }
        inline void load(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            keys.load(stream, context, pool);
        }
        inline static GaloisKeys load_new(std::istream& stream, HeContextPointer context, MemoryPoolHandle pool = MemoryPool::GlobalPool()) {
            GaloisKeys result;
            result.load(stream, context, pool);
            return result;
        }
        size_t serialized_size_upperbound(HeContextPointer context, CompressionMode mode = CompressionMode::Nil) const {
            return keys.serialized_size_upperbound(context, mode);
        }
    };
}