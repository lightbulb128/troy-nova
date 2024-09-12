#pragma once
#include "../he_context.h"
#include "../key.h"

namespace troy { namespace rlwe {

    void asymmetric_with_u_prng(
        const PublicKey& public_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        utils::RandomGenerator& u_prng,
        Ciphertext& destination,
        MemoryPoolHandle pool
    );

    void asymmetric_with_u_prng_batched(
        const PublicKey& public_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        utils::RandomGenerator& u_prng,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    );

    void asymmetric(
        const PublicKey& public_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        Ciphertext& destination,
        MemoryPoolHandle pool
    );

    void asymmetric_batched(
        const PublicKey& public_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    );

    void symmetric_with_c1_prng(
        const SecretKey& secret_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        utils::RandomGenerator& c1_prng,
        bool save_seed,
        Ciphertext& destination,
        MemoryPoolHandle pool
    );

    void symmetric_with_c1_prng_batched(
        const SecretKey& secret_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        utils::RandomGenerator& c1_prng,
        bool save_seed,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    );

    void symmetric(
        const SecretKey& secret_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        bool save_seed,
        Ciphertext& destination,
        MemoryPoolHandle pool
    );

    void symmetric_batched(
        const SecretKey& secret_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        bool save_seed,
        const std::vector<Ciphertext*>& destination,
        MemoryPoolHandle pool
    );

}}