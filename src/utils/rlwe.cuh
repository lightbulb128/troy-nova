#pragma once
#include "../he_context.cuh"
#include "../key.cuh"

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

    void asymmetric(
        const PublicKey& public_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        Ciphertext& destination,
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

    void symmetric(
        const SecretKey& secret_key,
        HeContextPointer context,
        const ParmsID& parms_id,
        bool is_ntt_form,
        bool save_seed,
        Ciphertext& destination,
        MemoryPoolHandle pool
    );

}}