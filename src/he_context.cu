#include "he_context.h"

namespace troy {

    ParmsID HeContext::create_next_context_data(ParmsID prev_parms_id) {
        // Create the next set of parameters by removing last modulus
        auto prev_parms_iter = context_data_map_.find(prev_parms_id);
        if (prev_parms_iter == context_data_map_.end()) {
            throw std::invalid_argument("ParmsID not found");
        }
        const EncryptionParameters& prev_parms = prev_parms_iter->second->parms();
        EncryptionParameters next_parms = prev_parms.clone(nullptr);
        next_parms.set_coeff_modulus(
            prev_parms.coeff_modulus().const_slice(0, prev_parms.coeff_modulus().size() - 1)
        );
        ParmsID next_parms_id = next_parms.parms_id();

        ContextData next_context_data(std::move(next_parms));
        next_context_data.validate(security_level_);
        
        // If not valid then return zero parms_id
        if (!next_context_data.qualifiers().parameters_set()) {
            return parms_id_zero;
        }
        
        // Add them to the context_data_map_
        context_data_map_.emplace(
            std::make_pair(
                next_parms_id,
                std::make_shared<const ContextData>(std::move(next_context_data))
            )
        );

        // Add pointer to next context_data to the previous one (linked list)
        // Add pointer to previous context_data to the next one (doubly linked list)
        // We need to remove constness first to modify this
        std::const_pointer_cast<ContextData>(context_data_map_.at(prev_parms_id))
            ->next_context_data_ = context_data_map_.at(next_parms_id);
        std::const_pointer_cast<ContextData>(context_data_map_.at(next_parms_id))
            ->prev_context_data_ = context_data_map_.at(prev_parms_id);

        return next_parms_id;

    }
    
    std::shared_ptr<HeContext> HeContext::create(EncryptionParameters parms, bool expand_mod_chain, SecurityLevel sec_level, uint64_t random_seed) {

        // std::cout << "creating he context" << std::endl;

        // Validate parameters and add new ContextData to the map
        // Note that this happens even if parameters are not valid
        HeContext he; 
        he.security_level_ = sec_level;
        if (parms.on_device()) {
            throw std::logic_error("[HeContext::create] Cannot create HeContext from device parameters");
        }
        
        // First create key_parms_id_.
        std::unordered_map<ParmsID, ContextDataPointer, std::TroyHashParmsID>& context_data_map
            = he.context_data_map_;
        ContextData key_context_data = ContextData(parms.clone(nullptr));
        key_context_data.validate(sec_level);
        ParmsID key_parms_id = parms.parms_id();
        context_data_map.emplace(
            std::make_pair(
                key_parms_id,
                std::make_shared<const ContextData>(std::move(key_context_data))
            )
        );
        he.key_parms_id_ = key_parms_id;

        // Then create first_parms_id_ if the parameters are valid and there is
        // more than one modulus in coeff_modulus. This is equivalent to expanding
        // the chain by one step. Otherwise, we set first_parms_id_ to equal
        // key_parms_id_.
        ParmsID first_parms_id;
        if (!context_data_map.at(key_parms_id)->qualifiers().parameters_set() || parms.coeff_modulus().size() == 1 || parms.use_special_prime_for_encryption()) {
            first_parms_id = key_parms_id;
        } else {
            ParmsID next_parms_id = he.create_next_context_data(key_parms_id);
            if (next_parms_id == parms_id_zero) {
                first_parms_id = key_parms_id;
            } else {
                first_parms_id = next_parms_id;
            }
        }
        he.first_parms_id_ = first_parms_id;

        // Set last_parms_id_ to point to first_parms_id_
        ParmsID last_parms_id = first_parms_id;

        // Check if keyswitching is available
        bool using_keyswitching = first_parms_id != key_parms_id;
        he.using_keyswitching_ = using_keyswitching;
        
        // If modulus switching chain is to be created, compute the remaining parameter sets as long as they are valid
        // to use (i.e., parameters_set() == true).
        if (expand_mod_chain && he.get_context_data(first_parms_id).value()->qualifiers().parameters_set()) {
            ParmsID prev_parms_id = first_parms_id;
            while (he.get_context_data(prev_parms_id).value()->parms().coeff_modulus().size() > 1) {
                ParmsID next_parms_id = he.create_next_context_data(prev_parms_id);
                if (next_parms_id == parms_id_zero) {
                    break;
                }
                last_parms_id = next_parms_id;
                prev_parms_id = next_parms_id;
            }
        }
        he.last_parms_id_ = last_parms_id;

        // Set the chain_index for each context_data
        int parms_count = context_data_map.size();
        ContextDataPointer context_data_ptr = he.get_context_data(key_parms_id).value();
        while (true) {
            std::const_pointer_cast<ContextData>(context_data_ptr)->chain_index_ = parms_count - 1;
            parms_count--;
            std::optional<ContextDataPointer> ptr = context_data_ptr->next_context_data();
            if (!ptr.has_value()) {
                break;
            } else {
                context_data_ptr = ptr.value();
            }
        }

        // Create random generator
        if (random_seed == 0) {
            random_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
        he.random_generator_.reset_seed(random_seed);

        return std::make_shared<HeContext>(std::move(he));
    }

    void HeContext::to_device_inplace(MemoryPoolHandle pool) {
        if (this->on_device()) {
            return;
        }
        // iterate over context datas
        for (auto& [parms_id, context_data_ptr] : this->context_data_map_) {
            // cast as mutable
            std::const_pointer_cast<ContextData>(context_data_ptr)->to_device_inplace(pool);
        }
        this->device = true;
    }

}