#include "rlwe.h"
#include "../batch_utils.h"

namespace troy {namespace rlwe {

    using utils::Array;
    using utils::ConstSlice;
    using utils::RandomGenerator;
    using utils::NTTTables;

    void asymmetric_with_u_prng(
        const PublicKey& pk,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        RandomGenerator& u_prng,
        Ciphertext& destination,
        MemoryPoolHandle pool
    ) {

        destination.seed() = 0;

        bool device = pk.on_device();
        
        std::optional<ContextDataPointer> context_data_optional = context->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[rlwe::asymmetric_with_u_prng] parms_id is not valid for the current context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        const Ciphertext& public_key = pk.as_ciphertext();
        size_t encrypted_size = public_key.polynomial_count();
        SchemeType scheme_type = parms.scheme();
        RandomGenerator& context_prng = context->random_generator();

        // check device consistency
        if (context_data->on_device() != device) {
            throw std::invalid_argument("[rlwe::asymmetric_with_u_prng] context_data and public_key is not on the same device.");
        }

        // make destination have right size and parms_id
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        destination.resize(context, parms_id, encrypted_size, false, false);
        destination.is_ntt_form() = is_ntt_form;
        destination.scale() = 1.0;
        destination.correction_factor() = 1;
        
        // c[j] = public_key[j] * u + e[j] in BFV/CKKS = public_key[j] * u + p * e[j] in BGV
        // where e[j] <-- chi, u <-- R_3

        // Create u <-- Ring_3
        Array<uint64_t> u = Array<uint64_t>::create_uninitialized(coeff_count * coeff_modulus_size, device, pool);
        u_prng.sample_poly_ternary(u.reference(), coeff_count, coeff_modulus);
        
        // c[j] = u * public_key[j]
        utils::ntt_inplace_p(u.reference(), coeff_count, ntt_tables);
        for (size_t j = 0; j < encrypted_size; j++) {
            utils::dyadic_product_p(
                u.const_reference(),
                public_key.poly(j),
                coeff_count, coeff_modulus, 
                destination.poly(j)
            );
        }
        if (!is_ntt_form) {
            utils::intt_inplace_ps(destination.data().reference(), encrypted_size, coeff_count, ntt_tables);
        }

        // Create e[j] <-- chi
        // c[j] = public_key[j] * u + e[j] in BFV/CKKS, = public_key[j] * u + p * e[j] in BGV,
        for (size_t j = 0; j < encrypted_size; j++) {
            // Reuse u as e
            context_prng.sample_poly_centered_binomial(u.reference(), coeff_count, coeff_modulus); 
            if (is_ntt_form) {
                utils::ntt_inplace_p(u.reference(), coeff_count, ntt_tables);
            }
            if (scheme_type == SchemeType::BGV) {
                utils::multiply_scalar_inplace_p(
                    u.reference(), parms.plain_modulus_host().value(), coeff_count, coeff_modulus
                );
            }
            utils::add_inplace_p(
                destination.poly(j), u.const_reference(), coeff_count, coeff_modulus
            );
        }
    }


    void asymmetric_with_u_prng_batched(
        const PublicKey& pk,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        RandomGenerator& u_prng,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    ) {

        size_t n = destination.size();
        if (n == 0) return;

        bool device = pk.on_device();
        
        std::optional<ContextDataPointer> context_data_optional = context->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[rlwe::asymmetric_with_u_prng] parms_id is not valid for the current context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        const Ciphertext& public_key = pk.as_ciphertext();
        size_t encrypted_size = public_key.polynomial_count();
        SchemeType scheme_type = parms.scheme();
        RandomGenerator& context_prng = context->random_generator();

        // check device consistency
        if (context_data->on_device() != device) {
            throw std::invalid_argument("[rlwe::asymmetric_with_u_prng] context_data and public_key is not on the same device.");
        }

        // make destination have right size and parms_id
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Ciphertext();
            if (device) destination[i]->to_device_inplace(pool);
            else destination[i]->to_host_inplace();
            destination[i]->resize(context, parms_id, encrypted_size, false, false);
            destination[i]->is_ntt_form() = is_ntt_form;
            destination[i]->scale() = 1.0;
            destination[i]->correction_factor() = 1;
        }
        
        // c[j] = public_key[j] * u + e[j] in BFV/CKKS = public_key[j] * u + p * e[j] in BGV
        // where e[j] <-- chi, u <-- R_3

        // Create u <-- Ring_3
        std::vector<Array<uint64_t>> u(n);
        for (size_t i = 0; i < n; i++) u[i] = Array<uint64_t>::create_uninitialized(coeff_count * coeff_modulus_size, device, pool);
        auto u_reference = batch_utils::rcollect_reference(u);
        u_prng.sample_poly_ternary_batched(u_reference, coeff_count, coeff_modulus, pool);
        
        // c[j] = u * public_key[j]
        utils::ntt_inplace_bp(u_reference, coeff_count, ntt_tables, pool);
        auto u_const_reference = batch_utils::rcollect_const_reference(u);
        for (size_t j = 0; j < encrypted_size; j++) {
            auto destination_j = batch_utils::pcollect_poly(destination, j);
            std::vector<ConstSlice<uint64_t>> public_key_j_repeated(n, public_key.poly(j));
            utils::dyadic_product_bp(
                u_const_reference,
                public_key_j_repeated,
                coeff_count, coeff_modulus, 
                destination_j, pool
            );
        }
        if (!is_ntt_form) {
            auto destination_reference = batch_utils::pcollect_reference(destination);
            utils::intt_inplace_bps(destination_reference, encrypted_size, coeff_count, ntt_tables, pool);
        }

        // Create e[j] <-- chi
        // c[j] = public_key[j] * u + e[j] in BFV/CKKS, = public_key[j] * u + p * e[j] in BGV,
        for (size_t j = 0; j < encrypted_size; j++) {
            // Reuse u as e
            context_prng.sample_poly_centered_binomial_batched(u_reference, coeff_count, coeff_modulus, pool); 
            if (is_ntt_form) {
                utils::ntt_inplace_bp(u_reference, coeff_count, ntt_tables, pool);
            }
            if (scheme_type == SchemeType::BGV) {
                utils::multiply_scalar_inplace_bp(
                    u_reference, parms.plain_modulus_host().value(), coeff_count, coeff_modulus, pool
                );
            }
            auto destination_j = batch_utils::pcollect_poly(destination, j);
            utils::add_inplace_bp(
                destination_j, u_const_reference, coeff_count, coeff_modulus
            );
        }
    }




    void asymmetric(
        const PublicKey& public_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        Ciphertext& destination,
        MemoryPoolHandle pool
    ) {
        RandomGenerator& u_prng = context->random_generator();
        asymmetric_with_u_prng(
            public_key, context, parms_id, is_ntt_form, u_prng, destination, pool
        );
    }

    void asymmetric_batched(
        const PublicKey& public_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    ) {
        RandomGenerator& u_prng = context->random_generator();
        asymmetric_with_u_prng_batched(
            public_key, context, parms_id, is_ntt_form, u_prng, destination, pool
        );
    }
    
    void symmetric_with_c1_prng(
        const SecretKey& sk,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        utils::RandomGenerator& c1_prng,
        bool save_seed,
        Ciphertext& destination,
        MemoryPoolHandle pool
    ) {
        
        destination.seed() = 0;
        bool device = sk.on_device();
        
        std::optional<ContextDataPointer> context_data_optional = context->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[rlwe::asymmetric_with_u_prng] parms_id is not valid for the current context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        const Plaintext& secret_key = sk.as_plaintext();
        size_t encrypted_size = 2;
        SchemeType scheme_type = parms.scheme();
        RandomGenerator& context_prng = context->random_generator();

        // check device consistency
        if (context_data->on_device() != device) {
            throw std::invalid_argument("[rlwe::symmetric_with_c1_prng] context_data and public_key is not on the same device.");
        }

        // make destination have right size and parms_id
        if (device) destination.to_device_inplace(pool);
        else destination.to_host_inplace();

        destination.resize(context, parms_id, encrypted_size, false, false);
        destination.is_ntt_form() = is_ntt_form;
        destination.scale() = 1.0;
        destination.correction_factor() = 1;

        uint64_t seed = 0;
        while (seed == 0) seed = c1_prng.sample_uint64();

        RandomGenerator c1_new_prng(seed);
        
        // Generate ciphertext: (c[0], c[1]) = ([-(as+ e)]_q, a) in BFV/CKKS
        // Generate ciphertext: (c[0], c[1]) = ([-(as+pe)]_q, a) in BGV

        if (is_ntt_form || !save_seed) {
            // Directly sample NTT form
            c1_new_prng.sample_poly_uniform(destination.poly(1), coeff_count, coeff_modulus);
        } else if (save_seed) {
            // Sample non-NTT form and store the seed
            c1_new_prng.sample_poly_uniform(destination.poly(1), coeff_count, coeff_modulus);
            // Transform the c1 into NTT representation
            utils::ntt_inplace_p(destination.poly(1), coeff_count, ntt_tables);
        }
        if (save_seed) {
            destination.seed() = seed;
        }


        // Sample e <-- chi
        Array<uint64_t> noise(coeff_count * coeff_modulus_size, device, pool);
        context_prng.sample_poly_centered_binomial(noise.reference(), coeff_count, coeff_modulus);

        // Calculate -(as+ e) (mod q) and store in c[0] in BFV/CKKS
        // Calculate -(as+pe) (mod q) and store in c[0] in BGV
        utils::dyadic_product_p(
            secret_key.poly(), destination.poly(1).as_const(),
            coeff_count, coeff_modulus, destination.poly(0)
        );

        if (is_ntt_form) {
            // Transform the noise e into NTT representation
            utils::ntt_inplace_p(noise.reference(), coeff_count, ntt_tables);
        } else {
            utils::intt_inplace_p(destination.poly(0), coeff_count, ntt_tables);
        }
        if (scheme_type == SchemeType::BGV) {
            utils::multiply_scalar_inplace_p(
                noise.reference(), parms.plain_modulus_host().value(), coeff_count, coeff_modulus
            );
        }
        
        // c0 = as + noise
        utils::add_inplace_p(
            destination.poly(0), noise.const_reference(), coeff_count, coeff_modulus
        );
        
        // (as + noise, a) -> (-(as + noise), a),
        utils::negate_inplace_p(destination.poly(0), coeff_count, coeff_modulus);

        if (!is_ntt_form && !save_seed) {
            // Transform the c1 into non-NTT representation
            utils::intt_inplace_p(destination.poly(1), coeff_count, ntt_tables);
        }
    }


    void symmetric_with_c1_prng_batched(
        const SecretKey& sk,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        utils::RandomGenerator& c1_prng,
        bool save_seed,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    ) {
        
        size_t n = destination.size();
        if (n == 0) return;

        bool device = sk.on_device();
        if (!device) {
            // directly use symmetric_with_c1_prng
            for (size_t i = 0; i < n; i++) {
                symmetric_with_c1_prng(
                    sk, context, parms_id, is_ntt_form, c1_prng, save_seed, *destination[i], pool
                );
            }
            return;
        }
        
        std::optional<ContextDataPointer> context_data_optional = context->get_context_data(parms_id);
        if (!context_data_optional.has_value()) {
            throw std::invalid_argument("[rlwe::asymmetric_with_u_prng] parms_id is not valid for the current context.");
        }
        ContextDataPointer context_data = context_data_optional.value();
        const EncryptionParameters& parms = context_data->parms();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        ConstSlice<NTTTables> ntt_tables = context_data->small_ntt_tables();
        const Plaintext& secret_key = sk.as_plaintext();
        size_t encrypted_size = 2;
        SchemeType scheme_type = parms.scheme();
        RandomGenerator& context_prng = context->random_generator();

        // check device consistency
        if (context_data->on_device() != device) {
            throw std::invalid_argument("[rlwe::symmetric_with_c1_prng] context_data and public_key is not on the same device.");
        }

        // make destination have right size and parms_id
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Ciphertext();
            if (device) destination[i]->to_device_inplace(pool);
            else destination[i]->to_host_inplace();
            destination[i]->resize(context, parms_id, encrypted_size, false, false);
            destination[i]->is_ntt_form() = is_ntt_form;
            destination[i]->scale() = 1.0;
            destination[i]->correction_factor() = 1;
        }

        
        // Generate ciphertext: (c[0], c[1]) = ([-(as+ e)]_q, a) in BFV/CKKS
        // Generate ciphertext: (c[0], c[1]) = ([-(as+pe)]_q, a) in BGV

        utils::Array<uint64_t> seeds(n, false); 
        for (size_t i = 0; i < n; i++) {
            uint64_t seed = 0;
            while (seed == 0) seed = c1_prng.sample_uint64();
            seeds[i] = seed;
            if (save_seed) destination[i]->seed() = seed;
        }
        seeds.to_device_inplace(pool);
        auto destination_poly1 = batch_utils::pcollect_poly(destination, 1);
        RandomGenerator::sample_poly_uniform_many(
            seeds.const_reference(),
            destination_poly1,
            coeff_count, coeff_modulus, pool
        );

        if (!is_ntt_form && save_seed) {
            // Transform the c1 into NTT representation
            utils::ntt_inplace_bp(destination_poly1, coeff_count, ntt_tables, pool);
        }

        // Sample e <-- chi
        std::vector<Array<uint64_t>> noise(n);
        for (size_t i = 0; i < n; i++) noise[i] = Array<uint64_t>::create_uninitialized(coeff_count * coeff_modulus_size, device, pool);
        auto noise_reference = batch_utils::rcollect_reference(noise);
        context_prng.sample_poly_centered_binomial_batched(noise_reference, coeff_count, coeff_modulus, pool);

        // Calculate -(as+ e) (mod q) and store in c[0] in BFV/CKKS
        // Calculate -(as+pe) (mod q) and store in c[0] in BGV
        auto destination_poly1_const = utils::slice_vec_to_const(destination_poly1);
        auto destination_poly0 = batch_utils::pcollect_poly(destination, 0);
        std::vector<ConstSlice<uint64_t>> secret_key_repeated(n, secret_key.poly());
        utils::dyadic_product_bp(
            secret_key_repeated, destination_poly1_const,
            coeff_count, coeff_modulus, destination_poly0
        );

        if (is_ntt_form) {
            // Transform the noise e into NTT representation
            utils::ntt_inplace_bp(noise_reference, coeff_count, ntt_tables, pool);
        } else {
            utils::intt_inplace_bp(destination_poly0, coeff_count, ntt_tables, pool);
        }
        if (scheme_type == SchemeType::BGV) {
            utils::multiply_scalar_inplace_bp(
                noise_reference, parms.plain_modulus_host().value(), coeff_count, coeff_modulus, pool
            );
        }
        
        // c0 = as + noise
        auto noise_const_reference = batch_utils::rcollect_const_reference(noise);
        utils::add_inplace_bp(
            destination_poly0, noise_const_reference, coeff_count, coeff_modulus, pool
        );
        
        // (as + noise, a) -> (-(as + noise), a),
        utils::negate_inplace_bp(destination_poly0, coeff_count, coeff_modulus, pool);

        if (!is_ntt_form && !save_seed) {
            // Transform the c1 into non-NTT representation
            utils::intt_inplace_bp(destination_poly1, coeff_count, ntt_tables, pool);
        }
    }



    void symmetric(
        const SecretKey& secret_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        bool save_seed,
        Ciphertext& destination,
        MemoryPoolHandle pool
    ) {
        RandomGenerator& c1_prng = context->random_generator();
        symmetric_with_c1_prng(
            secret_key, context, parms_id, is_ntt_form, c1_prng, save_seed, destination, pool
        );
    }

    
    void symmetric_batched(
        const SecretKey& secret_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        bool save_seed,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    ) {
        RandomGenerator& c1_prng = context->random_generator();
        symmetric_with_c1_prng_batched(
            secret_key, context, parms_id, is_ntt_form, c1_prng, save_seed, destination, pool
        );
    }


}}