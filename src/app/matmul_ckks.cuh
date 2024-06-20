#pragma once
#include <cassert>
#include "cipher2d.cuh"
#include "matmul.cuh"
#include "../ckks_encoder.cuh"

namespace troy { namespace linear {

    using utils::ceil_div;

    class CKKSMatmulHelper {
        // 0: encrypt inputs; 1: encrypt weights
        // 2: for calculating weight gradient

        void determine_block();

        Plaintext encode_weight_small(
            const CKKSEncoder& encoder,
            const double* weights,
            std::optional<ParmsID> parms_id,
            double scale,
            size_t li, size_t ui, size_t lj, size_t uj
        ) const;

    public:
    
        size_t batch_size, input_dims, output_dims;
        size_t slot_count;
        size_t batch_block, input_block, output_block;
        MatmulObjective objective; 
        bool pack_lwe;

        inline CKKSMatmulHelper(size_t batch_size, size_t input_dims, size_t output_dims, size_t slot_count, MatmulObjective objective = MatmulObjective::EncryptLeft, bool pack_lwe = true):
            batch_size(batch_size), input_dims(input_dims), output_dims(output_dims),
            slot_count(slot_count), objective(objective), pack_lwe(pack_lwe)
        {
            determine_block();
        }

        Plain2d encode_weights(
            const CKKSEncoder& encoder,
            const double* weights,
            std::optional<ParmsID> parms_id,
            double scale
        ) const;

        Plain2d encode_inputs(
            const CKKSEncoder& encoder,
            const double* inputs,
            std::optional<ParmsID> parms_id,
            double scale
        ) const;

        Cipher2d encrypt_inputs(
            const Encryptor& encryptor,
            const CKKSEncoder& encoder, 
            const double* inputs,
            std::optional<ParmsID> parms_id,
            double scale
        ) const;

        Cipher2d matmul(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& w) const;

        Cipher2d matmul_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& w) const;

        Cipher2d matmul_reverse(const Evaluator& evaluator, const Plain2d& a, const Cipher2d& w) const;

        Plain2d encode_outputs(
            const CKKSEncoder& encoder, 
            const double* outputs,
            std::optional<ParmsID> parms_id,
            double scale
        ) const;

        std::vector<double> decrypt_outputs(
            const CKKSEncoder& encoder,
            const Decryptor& decryptor,
            const Cipher2d& outputs
        ) const;

        Cipher2d pack_outputs(const Evaluator& evaluator, const GaloisKeys& autoKey, const Cipher2d& cipher) const;

        void serialize_encoded_weights(const Plain2d& w, std::ostream& stream) const;

        Plain2d deserialize_encoded_weights(std::istream& stream) const;

        void serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream) const;

        Cipher2d deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const;

    };

}}