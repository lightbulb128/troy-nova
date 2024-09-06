#include "evaluator.h"
#include "evaluator_utils.h"
#include "batch_utils.h"

namespace troy {


    using utils::ConstSlice;
    using utils::GaloisTool;

    void Evaluator::apply_keyswitching(const Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        if (kswitch_keys.data().size() != 1) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys size must be 1.");
        }
        if (encrypted.polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Ciphertext polynomial count must be 2.");
        }
        if (kswitch_keys.data()[0][0].as_ciphertext().polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys polynomial count must be 2. Check the key switch key generation for problems.");
        }
        destination = Ciphertext::like(encrypted, false, pool);
        this->switch_key_internal(encrypted, encrypted.poly(1), kswitch_keys, 0, Evaluator::SwitchKeyDestinationAssignMethod::Overwrite, destination, pool);
        
        ContextDataPointer context_data = this->get_context_data("[Evaluator::switch_key_inplace_internal]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();

        utils::add_inplace_p(destination.poly(0), encrypted.poly(0), parms.poly_modulus_degree(), parms.coeff_modulus());
    }

    void Evaluator::apply_keyswitching_inplace(Ciphertext& encrypted, const KSwitchKeys& kswitch_keys, MemoryPoolHandle pool) const {
        if (kswitch_keys.data().size() != 1) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys size must be 1.");
        }
        if (encrypted.polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Ciphertext polynomial count must be 2.");
        }
        if (kswitch_keys.data()[0][0].as_ciphertext().polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys polynomial count must be 2. Check the key switch key generation for problems.");
        }
        this->switch_key_internal(encrypted, encrypted.poly(1), kswitch_keys, 0, Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst, encrypted, pool);
        
    }



    void Evaluator::apply_keyswitching_batched(const std::vector<const Ciphertext*>& encrypted, const KSwitchKeys& kswitch_keys, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Encrypted and destination size must be the same.");
        }
        if (encrypted.size() == 0) return;
        if (kswitch_keys.data().size() != 1) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys size must be 1.");
        }
        for (const Ciphertext* e: encrypted) {
            if (e->polynomial_count() != 2) {
                throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Ciphertext polynomial count must be 2.");
            }
        }
        if (kswitch_keys.data()[0][0].as_ciphertext().polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys polynomial count must be 2. Check the key switch key generation for problems.");
        }
        size_t n = encrypted.size();
        for (size_t i = 0; i < n; i++) *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        auto encrypted_poly1 = batch_utils::pcollect_const_poly(encrypted, 1);
        this->switch_key_internal_batched(encrypted, encrypted_poly1, kswitch_keys, 0, Evaluator::SwitchKeyDestinationAssignMethod::Overwrite, destination, pool);
        
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::switch_key_inplace_internal]", parms_id);
        const EncryptionParameters& parms = context_data->parms();

        utils::add_inplace_bp(batch_utils::pcollect_poly(destination, 0), batch_utils::pcollect_const_poly(encrypted, 0), parms.poly_modulus_degree(), parms.coeff_modulus());
    }

    void Evaluator::apply_keyswitching_inplace_batched(const std::vector<Ciphertext*>& encrypted, const KSwitchKeys& kswitch_keys, MemoryPoolHandle pool) const {
        if (kswitch_keys.data().size() != 1) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys size must be 1.");
        }
        for (const Ciphertext* e: encrypted) {
            if (e->polynomial_count() != 2) {
                throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Ciphertext polynomial count must be 2.");
            }
        }
        if (kswitch_keys.data()[0][0].as_ciphertext().polynomial_count() != 2) {
            throw std::invalid_argument("[Evaluator::apply_keyswitching_inplace] Key switch keys polynomial count must be 2. Check the key switch key generation for problems.");
        }
        std::vector<utils::ConstSlice<uint64_t>> encrypted_poly1; encrypted_poly1.reserve(encrypted.size());
        for (size_t i = 0; i < encrypted.size(); i++) {
            encrypted_poly1.push_back(encrypted[i]->const_poly(1));
        }
        std::vector<const Ciphertext*> encrypted_const(encrypted.begin(), encrypted.end());
        this->switch_key_internal_batched(encrypted_const, encrypted_poly1, kswitch_keys, 0, Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst, encrypted, pool);
        
    }


    void Evaluator::relinearize_inplace_internal(Ciphertext& encrypted, const RelinKeys& relin_keys, size_t destination_size, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::relinearize_inplace_internal]", encrypted);
        if (relin_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Relin keys has incorrect parms id.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::relinearize_inplace_internal]", encrypted.parms_id());
        size_t encrypted_size = encrypted.polynomial_count();
        if (encrypted_size < 2 || destination_size > encrypted_size) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Destination size must be at least 2 and less/equal to the size of the encrypted polynomial.");
        }
        if (destination_size == encrypted_size) {
            return;
        }
        size_t relins_needed = encrypted_size - destination_size;
        for (size_t i = 0; i < relins_needed; i++) {
            this->switch_key_internal(
                encrypted, encrypted.const_poly(encrypted_size - 1),
                relin_keys.as_kswitch_keys(), RelinKeys::get_index(encrypted_size - 1), Evaluator::SwitchKeyDestinationAssignMethod::AddInplace, encrypted, pool);
            encrypted_size -= 1;
        }
        encrypted.resize(this->context(), context_data->parms_id(), destination_size);
    }

    void Evaluator::relinearize_internal(const Ciphertext& encrypted, const RelinKeys& relin_keys, size_t destination_size, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::relinearize_inplace_internal]", encrypted);
        if (relin_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Relin keys has incorrect parms id.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::relinearize_inplace_internal]", encrypted.parms_id());
        size_t encrypted_size = encrypted.polynomial_count();
        if (encrypted_size < 2 || destination_size > encrypted_size) {
            throw std::invalid_argument("[Evaluator::relinearize_inplace_internal] Destination size must be at least 2 and less/equal to the size of the encrypted polynomial.");
        }
        if (destination_size == encrypted_size) {
            return;
        }
        size_t relins_needed = encrypted_size - destination_size;
        destination = Ciphertext::like(encrypted, destination_size, false, pool);
        for (size_t i = 0; i < relins_needed; i++) {
            this->switch_key_internal(
                encrypted, encrypted.const_poly(encrypted_size - 1),
                relin_keys.as_kswitch_keys(), RelinKeys::get_index(encrypted_size - 1), 
                i == 0 ? Evaluator::SwitchKeyDestinationAssignMethod::Overwrite : Evaluator::SwitchKeyDestinationAssignMethod::AddInplace, 
                destination, pool);
            encrypted_size -= 1;
        }
        const EncryptionParameters& parms = context_data->parms();
        utils::add_inplace_ps(destination.polys(0, destination_size), encrypted.const_polys(0, destination_size), destination_size, parms.poly_modulus_degree(), parms.coeff_modulus());
    }


    void Evaluator::apply_galois(const Ciphertext& encrypted, size_t galois_element, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_no_seed("[Evaluator::apply_galois_inplace]", encrypted);
        if (galois_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois keys has incorrect parms id.");
        }
        ContextDataPointer context_data = this->get_context_data("[Evaluator::apply_galois_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.polynomial_count();
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const GaloisTool& galois_tool = key_context_data->galois_tool();

        if (!galois_keys.has_key(galois_element)) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois key not present.");
        }
        size_t m = coeff_count * 2;
        if ((galois_element & 1) == 0 || galois_element > m) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois element is not valid.");
        }
        if (encrypted_size > 2) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Ciphertext size must be 2.");
        }

        destination = Ciphertext::like(encrypted, false, pool);
        if (!encrypted.is_ntt_form()) {
            galois_tool.apply_ps(encrypted.const_polys(0, 2), 2, galois_element, coeff_modulus, destination.polys(0, 2));
        } else {
            galois_tool.apply_ntt_ps(encrypted.const_polys(0, 2), 2, coeff_modulus_size, galois_element, destination.polys(0, 2), pool);
        }

        this->switch_key_internal(encrypted, destination.poly(1), galois_keys.as_kswitch_keys(), GaloisKeys::get_index(galois_element), Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst, destination, pool);
    }


    void Evaluator::apply_galois_batched(
        const std::vector<const Ciphertext*>& encrypted, size_t galois_element, const GaloisKeys& galois_keys, 
        const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool
    ) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Encrypted and destination size must be the same.");
        }
        if (encrypted.size() == 0) return;
        check_no_seed_vec("[Evaluator::apply_galois_inplace]", encrypted);
        if (galois_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois keys has incorrect parms id.");
        }
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::apply_galois_inplace]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = get_vec_polynomial_count(encrypted);
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const GaloisTool& galois_tool = key_context_data->galois_tool();

        if (!galois_keys.has_key(galois_element)) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois key not present.");
        }
        size_t m = coeff_count * 2;
        if ((galois_element & 1) == 0 || galois_element > m) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois element is not valid.");
        }
        if (encrypted_size > 2) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Ciphertext size must be 2.");
        }

        size_t n = encrypted.size();
        for (size_t i = 0; i < n; i++) *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        auto encrypted_batched = batch_utils::pcollect_const_polys(encrypted, 0, 2);
        auto destination_batched = batch_utils::pcollect_polys(destination, 0, 2);
        bool is_ntt_form = get_is_ntt_form_vec(encrypted);
        if (!is_ntt_form) {
            galois_tool.apply_bps(encrypted_batched, 2, galois_element, coeff_modulus, destination_batched);
        } else {
            galois_tool.apply_ntt_bps(encrypted_batched, 2, coeff_modulus_size, galois_element, destination_batched, pool);
        }

        std::vector<ConstSlice<uint64_t>> destination_batched_1; destination_batched_1.reserve(n);
        for (size_t i = 0; i < n; i++) {
            destination_batched_1.push_back(destination[i]->const_poly(1));
        }
        this->switch_key_internal_batched(encrypted, destination_batched_1, galois_keys.as_kswitch_keys(), GaloisKeys::get_index(galois_element), Evaluator::SwitchKeyDestinationAssignMethod::OverwriteExceptFirst, destination, pool);
    }

    
    void Evaluator::apply_galois_plain(const Plaintext& plain, size_t galois_element, Plaintext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = plain.is_ntt_form()
            ? this->get_context_data("[Evaluator::apply_galois_plain_inplace]", plain.parms_id())
            : this->context()->key_context_data().value();
        const EncryptionParameters& parms = context_data->parms();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        ContextDataPointer key_context_data = this->context()->key_context_data().value();
        const GaloisTool& galois_tool = key_context_data->galois_tool();
        
        size_t m = coeff_count * 2;
        if ((galois_element & 1) == 0 || galois_element > m) {
            throw std::invalid_argument("[Evaluator::apply_galois_inplace] Galois element is not valid.");
        }

        destination = Plaintext::like(plain, false, pool);
        if (!plain.is_ntt_form()) {
            if (context_data->is_ckks()) {
                galois_tool.apply_p(plain.const_poly(), galois_element, coeff_modulus, destination.reference());
            } else {
                galois_tool.apply(plain.const_poly(), galois_element, context_data->parms().plain_modulus(), destination.reference());
            }
        } else {
            galois_tool.apply_ntt_p(plain.const_poly(), coeff_modulus_size, galois_element, destination.reference(), pool);
        }
    }

    void Evaluator::rotate_internal(const Ciphertext& encrypted, int steps, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->get_context_data("[Evaluator::rotate_inplace_internal]", encrypted.parms_id());
        if (!context_data->qualifiers().using_batching) {
            throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Batching must be enabled to use rotate.");
        }
        if (galois_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Galois keys has incorrect parms id.");
        }
        if (steps == 0) return;
        const GaloisTool& galois_tool = context_data->galois_tool();
        if (galois_keys.has_key(galois_tool.get_element_from_step(steps))) {
            size_t element = galois_tool.get_element_from_step(steps);
            this->apply_galois(encrypted, element, galois_keys, destination, pool);
        } else {
            // Convert the steps to NAF: guarantees using smallest HW
            std::vector<int> naf_steps = utils::naf(steps);
            if (naf_steps.size() == 1) {
                throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Galois key not present.");
            }
            bool done_flag = false;
            for (int naf_step : naf_steps) {
                if (!done_flag) {
                    this->rotate_internal(encrypted, naf_step, galois_keys, destination, pool);
                    done_flag = true;
                } else {
                    Ciphertext temp;
                    this->rotate_internal(destination, naf_step, galois_keys, temp, pool);
                    destination = std::move(temp);
                }
            }
        }
    }
    

    void Evaluator::rotate_internal_batched(const std::vector<const Ciphertext*>& encrypted, int steps, const GaloisKeys& galois_keys, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Encrypted and destination size must be the same.");
        }
        if (encrypted.size() == 0) return;
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::rotate_inplace_internal]", parms_id);
        if (!context_data->qualifiers().using_batching) {
            throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Batching must be enabled to use rotate.");
        }
        if (galois_keys.parms_id() != this->context()->key_parms_id()) {
            throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Galois keys has incorrect parms id.");
        }
        if (steps == 0) return;
        const GaloisTool& galois_tool = context_data->galois_tool();
        if (galois_keys.has_key(galois_tool.get_element_from_step(steps))) {
            size_t element = galois_tool.get_element_from_step(steps);
            this->apply_galois_batched(encrypted, element, galois_keys, destination, pool);
        } else {
            // Convert the steps to NAF: guarantees using smallest HW
            std::vector<int> naf_steps = utils::naf(steps);
            if (naf_steps.size() == 1) {
                throw std::invalid_argument("[Evaluator::rotate_inplace_internal] Galois key not present.");
            }
            bool done_flag = false;
            std::vector<Ciphertext> temp(encrypted.size());
            auto temp_batched = batch_utils::collect_pointer(temp);
            auto destination_const_batched = batch_utils::pcollect_const_pointer(destination);
            for (int naf_step : naf_steps) {
                if (!done_flag) {
                    this->rotate_internal_batched(encrypted, naf_step, galois_keys, destination, pool);
                    done_flag = true;
                } else {
                    this->rotate_internal_batched(destination_const_batched, naf_step, galois_keys, temp_batched, pool);
                    for (size_t i = 0; i < encrypted.size(); i++) {
                        *destination[i] = std::move(temp[i]);
                    }
                }
            }
        }
    }

    void Evaluator::conjugate_internal(const Ciphertext& encrypted, const GaloisKeys& galois_keys, Ciphertext& destination, MemoryPoolHandle pool) const {
        ContextDataPointer context_data = this->get_context_data("Evaluator::conjugate_inplace_internal", encrypted.parms_id());
        if (!context_data->qualifiers().using_batching) {
            throw std::logic_error("[Evaluator::conjugate_inplace_internal] Batching is not enabled.");
        }
        const GaloisTool& galois_tool = context_data->galois_tool();
        this->apply_galois(encrypted, galois_tool.get_element_from_step(0), galois_keys, destination, pool);
    }


    void Evaluator::conjugate_internal_batched(const std::vector<const Ciphertext*>& encrypted, const GaloisKeys& galois_keys, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::conjugate_inplace_internal] Encrypted and destination size must be the same.");
        }
        if (encrypted.size() == 0) return;
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("Evaluator::conjugate_inplace_internal", parms_id);
        if (!context_data->qualifiers().using_batching) {
            throw std::logic_error("[Evaluator::conjugate_inplace_internal] Batching is not enabled.");
        }
        const GaloisTool& galois_tool = context_data->galois_tool();
        this->apply_galois_batched(encrypted, galois_tool.get_element_from_step(0), galois_keys, destination, pool);
    }


}