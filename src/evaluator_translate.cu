#include "evaluator.h"
#include "evaluator_utils.h"
#include "batch_utils.h"
#include "utils/constants.h"

namespace troy {

    using utils::Slice;
    using utils::ConstSlice;


    void Evaluator::negate_inplace(Ciphertext& encrypted) const {
        check_ciphertext("[Evaluator::negate_inplace]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate_inplace]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t poly_count = encrypted.polynomial_count();
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_inplace_ps(encrypted.data().reference(), poly_count, poly_degree, coeff_modulus);
    }

    void Evaluator::negate(const Ciphertext& encrypted, Ciphertext& destination, MemoryPoolHandle pool) const {
        check_ciphertext("[Evaluator::negate]", encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate]", encrypted.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        destination = Ciphertext::like(encrypted, false, pool);
        size_t poly_count = encrypted.polynomial_count();
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_ps(encrypted.data().const_reference(), poly_count, poly_degree, coeff_modulus, destination.data().reference());
    }

    
    void Evaluator::negate_inplace_batched(const std::vector<Ciphertext*>& encrypted, MemoryPoolHandle pool) const {
        if (encrypted.size() == 0) return;
        check_ciphertext_vec("[Evaluator::negate_inplace_batched]", encrypted);
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate_inplace_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t poly_count = get_vec_polynomial_count(encrypted);
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_inplace_bps(batch_utils::pcollect_reference(encrypted), poly_count, poly_degree, coeff_modulus, pool);
    }

    void Evaluator::negate_batched(const std::vector<const Ciphertext*>& encrypted, const std::vector<Ciphertext*>& destination, MemoryPoolHandle pool) const {
        if (encrypted.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::negate_batched] Size mismatch");
        }
        if (encrypted.size() == 0) return;
        check_ciphertext_vec("[Evaluator::negate_batched]", encrypted);
        ParmsID parms_id = get_vec_parms_id(encrypted);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::negate_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        for (size_t i = 0; i < encrypted.size(); i++) *destination[i] = Ciphertext::like(*encrypted[i], false, pool);
        size_t poly_count = get_vec_polynomial_count(encrypted);
        size_t poly_degree = parms.poly_modulus_degree();
        utils::negate_bps(batch_utils::pcollect_const_reference(encrypted), poly_count, poly_degree, coeff_modulus, batch_utils::pcollect_reference(destination));
    }




    void Evaluator::translate(const Ciphertext& encrypted1, const Ciphertext& encrypted2, Ciphertext& destination, bool subtract, MemoryPoolHandle pool) const {
        check_ciphertext("[Evaluator::translate_inplace]", encrypted1);
        check_ciphertext("[Evaluator::translate_inplace]", encrypted2);
        check_same_parms_id("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        check_same_scale("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        check_same_ntt_form("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::translate_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t enc1_size = encrypted1.polynomial_count();
        size_t enc2_size = encrypted2.polynomial_count();
        size_t max_size = std::max(enc1_size, enc2_size);
        size_t min_size = std::min(enc1_size, enc2_size);
        size_t coeff_count = parms.poly_modulus_degree();

        if (encrypted1.on_device()) destination.to_device_inplace(pool);
        if (!encrypted1.on_device()) destination.to_host_inplace();

        if (encrypted1.correction_factor() != encrypted2.correction_factor()) {
            // Balance correction factors and multiply by scalars before addition in BGV
            uint64_t f0, f1, f2;
            const Modulus& plain_modulus = parms.plain_modulus_host();
            balance_correction_factors(
                encrypted1.correction_factor(), encrypted2.correction_factor(),
                plain_modulus, f0, f1, f2
            );
            destination = Ciphertext::like(encrypted1, enc1_size, false, pool);
            utils::multiply_scalar_ps(encrypted1.const_reference(), f1, enc1_size, coeff_count, coeff_modulus, destination.data().reference());
            Ciphertext encrypted2_copy = Ciphertext::like(encrypted2, false, pool);
            utils::multiply_scalar_ps(encrypted2.data().const_reference(), f2, enc2_size, coeff_count, coeff_modulus, encrypted2_copy.data().reference()); 
            // Set new correction factor
            destination.correction_factor() = f0;
            encrypted2_copy.correction_factor() = f0;
            this->translate_inplace(destination, encrypted2_copy, subtract, pool);
        } else {
            // Prepare destination
            destination = Ciphertext::like(encrypted1, max_size, false, pool);
            if (!subtract) {
                utils::add_ps(encrypted1.const_reference(), encrypted2.const_reference(), min_size, coeff_count, coeff_modulus, destination.reference());
            } else {
                utils::sub_ps(encrypted1.const_reference(), encrypted2.const_reference(), min_size, coeff_count, coeff_modulus, destination.reference());
            }
            // Copy the remainding polys of the array with larger count into encrypted1
            if (enc1_size < enc2_size) {
                if (!subtract) {
                    destination.polys(enc1_size, enc2_size).copy_from_slice(encrypted2.polys(enc1_size, enc2_size));
                } else {
                    utils::negate_ps(encrypted2.polys(enc1_size, enc2_size), enc2_size - enc1_size, coeff_count, coeff_modulus, destination.polys(enc1_size, enc2_size));
                }
            } else if (enc1_size > enc2_size) {
                destination.polys(enc2_size, enc1_size).copy_from_slice(encrypted1.polys(enc2_size, enc1_size));
            }
        }
    }


    void Evaluator::translate_batched(
        const std::vector<const Ciphertext*>& encrypted1, const std::vector<const Ciphertext*>& encrypted2, const std::vector<Ciphertext*>& destination, bool subtract, MemoryPoolHandle pool
    ) const {
        check_ciphertext_vec("[Evaluator::translate_inplace_batched]", encrypted1);
        check_ciphertext_vec("[Evaluator::translate_inplace_batched]", encrypted2);
        check_same_parms_id_vec("[Evaluator::translate_inplace_batched]", encrypted1, encrypted2);
        check_same_scale_vec("[Evaluator::translate_inplace_batched]", encrypted1, encrypted2);
        check_same_ntt_form_vec("[Evaluator::translate_inplace_batched]", encrypted1, encrypted2);
        if (encrypted1.size() != encrypted2.size() || encrypted1.size() != destination.size()) {
            throw std::invalid_argument("[Evaluator::translate_inplace_batched] Arguments have different sizes.");
        }
        if (encrypted1.size() == 0) return;
        ParmsID parms_id = encrypted1[0]->parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::translate_inplace_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t enc1_size = get_vec_polynomial_count(encrypted1);
        size_t enc2_size = get_vec_polynomial_count(encrypted2);
        size_t max_size = std::max(enc1_size, enc2_size);
        size_t min_size = std::min(enc1_size, enc2_size);
        size_t coeff_count = parms.poly_modulus_degree();

        for (Ciphertext* c: destination) {
            *c = Ciphertext();
            if (encrypted1[0]->on_device()) c->to_device_inplace(pool);
            else c->to_host_inplace();
        }

        uint64_t enc1_correction_factor = get_vec_correction_factor(encrypted1);
        uint64_t enc2_correction_factor = get_vec_correction_factor(encrypted2);

        if (enc1_correction_factor != enc2_correction_factor) {

            // Balance correction factors and multiply by scalars before addition in BGV
            uint64_t f0, f1, f2;
            const Modulus& plain_modulus = parms.plain_modulus_host();
            balance_correction_factors(
                enc1_correction_factor, enc2_correction_factor,
                plain_modulus, f0, f1, f2
            );

            for (size_t i = 0; i < destination.size(); i++) *destination[i] = Ciphertext::like(*encrypted1[i], enc1_size, false, pool);
            {
                auto arg1 = batch_utils::pcollect_const_reference(encrypted1);
                auto arg2 = batch_utils::pcollect_reference(destination);
                utils::multiply_scalar_bps(arg1, f1, enc1_size, coeff_count, coeff_modulus, arg2);
            }

            std::vector<Ciphertext> encrypted2_copy; encrypted2_copy.reserve(encrypted2.size());
            for (const Ciphertext* c: encrypted2) encrypted2_copy.push_back(Ciphertext::like(*c, false, pool));

            {
                auto arg1 = batch_utils::pcollect_const_reference(encrypted2);
                auto arg2 = batch_utils::rcollect_reference(encrypted2_copy);
                utils::multiply_scalar_bps(arg1, f2, enc2_size, coeff_count, coeff_modulus, arg2);
            }
            
            // Set new correction factor
            for (Ciphertext* c: destination) c->correction_factor() = f0;
            for (Ciphertext& c: encrypted2_copy) c.correction_factor() = f0;

            {
                auto arg = batch_utils::collect_const_pointer(encrypted2_copy);
                this->translate_inplace_batched(destination, arg, subtract, pool);
            }

        } else {

            // Prepare destination
            for (size_t i = 0; i < destination.size(); i++) 
                *destination[i] = Ciphertext::like(*encrypted1[i], max_size, false, pool);

            {
                auto arg1 = batch_utils::pcollect_const_reference(encrypted1);
                auto arg2 = batch_utils::pcollect_const_reference(encrypted2);
                auto arg3 = batch_utils::pcollect_reference(destination);
                if (!subtract) {
                    utils::add_bps(arg1, arg2, min_size, coeff_count, coeff_modulus, arg3);
                } else {
                    utils::sub_bps(arg1, arg2, min_size, coeff_count, coeff_modulus, arg3);
                }
            }

            // Copy the remainding polys of the array with larger count into encrypted1
            {
                if (enc1_size < enc2_size) {
                    auto arg1 = batch_utils::pcollect_polys(destination, enc1_size, enc2_size);
                    auto arg2 = batch_utils::pcollect_const_polys(encrypted2, enc1_size, enc2_size);
                    for (const Ciphertext* c: encrypted2) arg2.push_back(c->polys(enc1_size, enc2_size));
                    if (!subtract) {
                        utils::copy_slice_b(arg2, arg1);
                    } else {
                        utils::negate_bps(arg2, enc2_size - enc1_size, coeff_count, coeff_modulus, arg1);
                    }
                } else if (enc1_size > enc2_size) {
                    auto arg1 = batch_utils::pcollect_polys(destination, enc2_size, enc1_size);
                    auto arg2 = batch_utils::pcollect_const_polys(encrypted1, enc2_size, enc1_size);
                    utils::copy_slice_b(arg2, arg1);
                }
            }
        }
    }

    void Evaluator::translate_inplace(Ciphertext& encrypted1, const Ciphertext& encrypted2, bool subtract, MemoryPoolHandle pool) const {
        check_ciphertext("[Evaluator::translate_inplace]", encrypted1);
        check_ciphertext("[Evaluator::translate_inplace]", encrypted2);
        check_same_parms_id("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        check_same_scale("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        check_same_ntt_form("[Evaluator::translate_inplace]", encrypted1, encrypted2);
        ContextDataPointer context_data = this->get_context_data("[Evaluator::translate_inplace]", encrypted1.parms_id());
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t enc1_size = encrypted1.polynomial_count();
        size_t enc2_size = encrypted2.polynomial_count();
        size_t max_size = std::max(enc1_size, enc2_size);
        size_t min_size = std::min(enc1_size, enc2_size);
        size_t coeff_count = parms.poly_modulus_degree();

        if (encrypted1.correction_factor() != encrypted2.correction_factor()) {
            // Balance correction factors and multiply by scalars before addition in BGV
            uint64_t f0, f1, f2;
            const Modulus& plain_modulus = parms.plain_modulus_host();
            balance_correction_factors(
                encrypted1.correction_factor(), encrypted2.correction_factor(),
                plain_modulus, f0, f1, f2
            );
            utils::multiply_scalar_inplace_ps(encrypted1.data().reference(), f1, enc1_size, coeff_count, coeff_modulus);
            Ciphertext encrypted2_copy = Ciphertext::like(encrypted2, false, pool);
            utils::multiply_scalar_ps(encrypted2.data().const_reference(), f2, enc2_size, coeff_count, coeff_modulus, encrypted2_copy.data().reference()); 
            // Set new correction factor
            encrypted1.correction_factor() = f0;
            encrypted2_copy.correction_factor() = f0;
            this->translate_inplace(encrypted1, encrypted2_copy, subtract, pool);
        } else {
            // Prepare destination
            encrypted1.resize(this->context(), context_data->parms_id(), max_size, false);
            if (!subtract) {
                utils::add_inplace_ps(encrypted1.data().reference(), encrypted2.data().const_reference(), min_size, coeff_count, coeff_modulus);
            } else {
                utils::sub_inplace_ps(encrypted1.data().reference(), encrypted2.data().const_reference(), min_size, coeff_count, coeff_modulus);
            }
            // Copy the remainding polys of the array with larger count into encrypted1
            if (enc1_size < enc2_size) {
                if (!subtract) {
                    encrypted1.polys(enc1_size, enc2_size).copy_from_slice(encrypted2.polys(enc1_size, enc2_size));
                } else {
                    utils::negate_ps(encrypted2.polys(enc1_size, enc2_size), enc2_size - enc1_size, coeff_count, coeff_modulus, encrypted1.polys(enc1_size, enc2_size));
                }
            }
        }
    }
    
    void Evaluator::translate_inplace_batched(
        const std::vector<Ciphertext*>& encrypted1, const std::vector<const Ciphertext*>& encrypted2, bool subtract, MemoryPoolHandle pool
    ) const {
        check_ciphertext_vec("[Evaluator::translate_inplace_batched]", encrypted1);
        check_ciphertext_vec("[Evaluator::translate_inplace_batched]", encrypted2);
        check_same_parms_id_vec("[Evaluator::translate_inplace_batched]", encrypted1, encrypted2);
        check_same_scale_vec("[Evaluator::translate_inplace_batched]", encrypted1, encrypted2);
        check_same_ntt_form_vec("[Evaluator::translate_inplace_batched]", encrypted1, encrypted2);
        if (encrypted1.size() != encrypted2.size()) {
            throw std::invalid_argument("[Evaluator::translate_inplace_batched] Arguments have different sizes.");
        }
        if (encrypted1.size() == 0) return;
        ParmsID parms_id = encrypted1[0]->parms_id();
        ContextDataPointer context_data = this->get_context_data("[Evaluator::translate_inplace_batched]", parms_id);
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t enc1_size = get_vec_polynomial_count(encrypted1);
        size_t enc2_size = get_vec_polynomial_count(encrypted2);
        size_t max_size = std::max(enc1_size, enc2_size);
        size_t min_size = std::min(enc1_size, enc2_size);
        size_t coeff_count = parms.poly_modulus_degree();

        uint64_t enc1_correction_factor = get_vec_correction_factor(encrypted1);
        uint64_t enc2_correction_factor = get_vec_correction_factor(encrypted2);

        if (enc1_correction_factor != enc2_correction_factor) {

            // Balance correction factors and multiply by scalars before addition in BGV
            uint64_t f0, f1, f2;
            const Modulus& plain_modulus = parms.plain_modulus_host();
            balance_correction_factors(
                enc1_correction_factor, enc2_correction_factor,
                plain_modulus, f0, f1, f2
            );
            
            {
                auto arg = batch_utils::pcollect_reference(encrypted1);
                utils::multiply_scalar_inplace_bps(arg, f1, enc1_size, coeff_count, coeff_modulus);
            }

            std::vector<Ciphertext> encrypted2_copy; encrypted2_copy.reserve(encrypted2.size());
            for (const Ciphertext* c : encrypted2) encrypted2_copy.push_back(Ciphertext::like(*c, false, pool));

            {
                auto arg1 = batch_utils::pcollect_const_reference(encrypted2);
                auto arg2 = batch_utils::rcollect_reference(encrypted2_copy);
                utils::multiply_scalar_bps(arg1, f2, enc2_size, coeff_count, coeff_modulus, arg2);
            }
            
            // Set new correction factor
            for (Ciphertext* c: encrypted1) c->correction_factor() = f0;
            for (Ciphertext& c: encrypted2_copy) c.correction_factor() = f0;

            {
                auto arg = batch_utils::collect_const_pointer(encrypted2_copy);
                this->translate_inplace_batched(encrypted1, arg, subtract, pool);
            }

        } else {

            // Prepare destination
            for (Ciphertext* c: encrypted1) c->resize(this->context(), context_data->parms_id(), max_size, false);
            {
                auto arg1 = batch_utils::pcollect_reference(encrypted1);
                auto arg2 = batch_utils::pcollect_const_reference(encrypted2);
                if (!subtract) {
                    utils::add_inplace_bps(arg1, arg2, min_size, coeff_count, coeff_modulus);
                } else {
                    utils::sub_inplace_bps(arg1, arg2, min_size, coeff_count, coeff_modulus);
                }
            }

            // Copy the remainding polys of the array with larger count into encrypted1
            {
                if (enc1_size < enc2_size) {
                    auto arg1 = batch_utils::pcollect_polys(encrypted1, enc1_size, enc2_size);
                    auto arg2 = batch_utils::pcollect_const_polys(encrypted2, enc1_size, enc2_size);
                    if (!subtract) {
                        utils::copy_slice_b(arg2, arg1);
                    } else {
                        utils::negate_bps(arg2, enc2_size - enc1_size, coeff_count, coeff_modulus, arg1);
                    }
                }
            }
        }
    }


}