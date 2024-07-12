#pragma once
#include <cassert>
#include "cipher2d.h"
#include "matmul.h"

namespace troy { namespace linear {

    using utils::ceil_div;

    class Conv2dHelper {
        // 0: encrypt inputs; 1: encrypt weights
        // 2: for calculating weight gradient

        void determine_block();

        size_t get_total_batch_size() const;

    public:
    
        size_t batch_size, input_channels, output_channels, image_height, image_width;
        size_t kernel_height, kernel_width;
        size_t slot_count;
        size_t batch_block, input_channel_block, output_channel_block;
        size_t image_height_block, image_width_block;
        MatmulObjective objective; 
        MemoryPoolHandle pool;
        bool pack_lwe;

        inline Conv2dHelper(
            size_t batch_size, 
            size_t input_channels, size_t output_channels,
            size_t image_height, size_t image_width,
            size_t kernel_height, size_t kernel_width,
            size_t poly_degree,
            MatmulObjective objective = MatmulObjective::EncryptLeft,
            MemoryPoolHandle pool = MemoryPool::GlobalPool()
        ):
            batch_size(batch_size), 
            input_channels(input_channels), output_channels(output_channels),
            image_height(image_height), image_width(image_width),
            kernel_height(kernel_height), kernel_width(kernel_width),
            slot_count(poly_degree), objective(objective),
            pool(pool)
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

        Cipher2d conv2d(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& w) const;

        Cipher2d conv2d_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& w) const;

        Cipher2d conv2d_reverse(const Evaluator& evaluator, const Plain2d& a, const Cipher2d& w) const;

        Plain2d encode_outputs(
            const BatchEncoder& encoder, 
            const uint64_t* outputs
        ) const;

        std::vector<uint64_t> decrypt_outputs(
            const BatchEncoder& encoder,
            const Decryptor& decryptor,
            const Cipher2d& outputs
        ) const;

        void serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream) const;

        Cipher2d deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const;

    };

}}