#pragma once
#include <cassert>
#include "cipher2d.cuh"

namespace troy { namespace linear {

    using utils::ceil_div;

    enum class MatmulObjective : uint8_t {
        EncryptLeft = 0,
        EncryptRight = 1,
        Crossed = 2
    };

    class MatmulHelper {
        size_t batch_size, input_dims, output_dims;
        size_t slot_count;
        size_t batch_block, input_block, output_block;
        MatmulObjective objective; 
        bool pack_lwe;
        // 0: encrypt inputs; 1: encrypt weights
        // 2: for calculating weight gradient

        void determine_block();

        Plaintext encode_weight_small(
            const BatchEncoder& encoder,
            const uint64_t* weights,
            size_t li, size_t ui, size_t lj, size_t uj
        ) const;

    public:

        inline MatmulHelper(size_t batch_size, size_t input_dims, size_t output_dims, size_t slot_count, MatmulObjective objective = MatmulObjective::EncryptLeft, bool pack_lwe = true):
            batch_size(batch_size), input_dims(input_dims), output_dims(output_dims),
            slot_count(slot_count), objective(objective), pack_lwe(pack_lwe)
        {
            determine_block();
        }

        Plain2d encode_weights(
            const BatchEncoder& encoder,
            const uint64_t* weights
        ) const;

        Plain2d encode_inputs(
            const BatchEncoder& encoder,
            const uint64_t* inputs
        ) const;

        Cipher2d encrypt_inputs(
            const Encryptor& encryptor,
            const BatchEncoder& encoder, 
            const uint64_t* inputs
        ) const;

        Cipher2d matmul(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& w) const;

        Cipher2d matmul_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& w) const;

        Cipher2d matmul_reverse(const Evaluator& evaluator, const Plain2d& a, const Cipher2d& w) const;

        Plain2d encode_outputs(
            const BatchEncoder& encoder, 
            const uint64_t* outputs
        ) const;

        std::vector<uint64_t> decrypt_outputs(
            const BatchEncoder& encoder,
            const Decryptor& decryptor,
            const Cipher2d& outputs
        ) const;

        Cipher2d packOutputs(const Evaluator& evaluator, const GaloisKeys& autoKey, const Cipher2d& cipher) const;

        void serialize_encoded_weights(const Plain2d& w, std::ostream& stream) const;

        Plain2d deserialize_encoded_weights(std::istream& stream) const;

        void serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream) const;

        Cipher2d deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const;

    };

}}