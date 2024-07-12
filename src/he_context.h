#pragma once
#include <unordered_map>
#include "context_data.h"
#include "utils/random_generator.h"

namespace troy {

    class HeContext {
    
    private:
        bool device;
        ParmsID key_parms_id_, first_parms_id_, last_parms_id_;
        std::unordered_map<ParmsID, ContextDataPointer, std::TroyHashParmsID> context_data_map_;
        SecurityLevel security_level_;
        bool using_keyswitching_;
        mutable utils::RandomGenerator random_generator_;

        ParmsID create_next_context_data(ParmsID prev_parms_id);

        HeContext(): device(false), context_data_map_(){}

    public:

        inline MemoryPoolHandle pool() const { return first_context_data_pointer()->pool(); }
        inline size_t device_index() const { return first_context_data_pointer()->device_index(); }

        inline bool on_device() const noexcept {
            return device;
        }

        inline ParmsID key_parms_id() const noexcept {
            return key_parms_id_;
        }

        inline ParmsID first_parms_id() const noexcept {
            return first_parms_id_;
        }

        inline ParmsID last_parms_id() const noexcept {
            return last_parms_id_;
        }

        inline std::optional<ContextDataPointer> get_context_data(const ParmsID& parms_id) const noexcept {
            auto it = context_data_map_.find(parms_id);
            if (it == context_data_map_.end()) {
                return std::nullopt;
            }
            return it->second;
        }

        // This definition is for those who have difficulty accessing std::optional.
        inline ContextDataPointer get_context_data_pointer(const ParmsID& parms_id) const noexcept {
            auto it = context_data_map_.find(parms_id);
            if (it == context_data_map_.end()) {
                return std::shared_ptr<ContextData>(nullptr);
            }
            return it->second;
        }

        inline std::optional<ContextDataPointer> key_context_data() const noexcept {
            return get_context_data(key_parms_id_);
        }

        inline std::optional<ContextDataPointer> first_context_data() const noexcept {
            return get_context_data(first_parms_id_);
        }

        inline std::optional<ContextDataPointer> last_context_data() const noexcept {
            return get_context_data(last_parms_id_);
        }

        inline ContextDataPointer key_context_data_pointer() const noexcept {
            return get_context_data_pointer(key_parms_id_);
        }

        inline ContextDataPointer first_context_data_pointer() const noexcept {
            return get_context_data_pointer(first_parms_id_);
        }

        inline ContextDataPointer last_context_data_pointer() const noexcept {
            return get_context_data_pointer(last_parms_id_);
        }

        inline SecurityLevel security_level() const noexcept {
            return security_level_;
        }

        inline bool using_keyswitching() const noexcept {
            return using_keyswitching_;
        }

        inline bool parameters_set() const {
            std::optional<ContextDataPointer> first_context_data = this->first_context_data();
            if (!first_context_data.has_value()) {
                return false;
            }
            return first_context_data.value()->qualifiers().parameters_set();
        }

        static std::shared_ptr<HeContext> create(
            EncryptionParameters parms, 
            bool expand_mod_chain, 
            SecurityLevel sec_level, 
            uint64_t random_seed = 0
        );

        inline utils::RandomGenerator& random_generator() const noexcept {
            return random_generator_;
        }

        void to_device_inplace(MemoryPoolHandle pool = MemoryPool::GlobalPool());

    };

    using HeContextPointer = std::shared_ptr<HeContext>;

}