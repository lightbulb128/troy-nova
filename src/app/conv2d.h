#pragma once
#include <cassert>
#include "cipher2d.h"
#include "matmul.h"
#include "encoder_adapter.h"

namespace troy { namespace linear {

    using utils::ceil_div;

    class Conv2dHelper {
        // 0: encrypt inputs; 1: encrypt weights
        // 2: for calculating weight gradient

        void determine_block();

        size_t get_total_batch_size() const;

        template <typename E, typename T>
        void encode_weights(
            const E& encoder, const Encryptor* encryptor, const T* weights, 
            bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher
        ) const;
        
        template <typename E, typename T>
        void encode_inputs(
            const E& encoder, const Encryptor* encryptor, const T* inputs,
            bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher
        ) const;

        template <typename E, typename T>
        Plain2d encode_outputs(const E& encoder, const T* outputs) const;

        template <typename E, typename T>
        std::vector<T> decrypt_outputs(const E& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;


    public:
    
        size_t batch_size, input_channels, output_channels, image_height, image_width;
        size_t kernel_height, kernel_width;
        size_t slot_count;
        size_t batch_block, input_channel_block, output_channel_block;
        size_t image_height_block, image_width_block;
        MatmulObjective objective; 
        MemoryPoolHandle pool;
        bool pack_lwe;

        // When this is enabled, in the `matmul` function
        // we will invoke the batched operation. Note that this will
        // consume more memory (presumably by a ratio of O(1) constant) than the non-batched version.
        bool batched_mul = false;

        
        inline void set_pool(MemoryPoolHandle pool) {
            this->pool = pool;
        }

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
            pool(pool), batched_mul(true)
        {
            determine_block();
        }


        Plain2d encode_weights_uint64s(const BatchEncoder& encoder, const uint64_t* weights) const;
        Plain2d encode_weights_doubles(const CKKSEncoder& encoder, const double* weights, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Plain2d encode_weights_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* weights, std::optional<ParmsID> parms_id) const;

        Cipher2d encrypt_weights_uint64s(const Encryptor& encryptor, const BatchEncoder& encoder, const uint64_t* weights) const;
        Cipher2d encrypt_weights_doubles(const Encryptor& encryptor, const CKKSEncoder& encoder, const double* weights, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Cipher2d encrypt_weights_ring2k(const Encryptor& encryptor, const PolynomialEncoderRing2k<T>& encoder, const T* weights, std::optional<ParmsID> parms_id) const;

        Plain2d encode_inputs_uint64s(const BatchEncoder& encoder, const uint64_t* inputs) const;
        Plain2d encode_inputs_doubles(const CKKSEncoder& encoder, const double* inputs, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Plain2d encode_inputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* inputs, std::optional<ParmsID> parms_id) const;

        Cipher2d encrypt_inputs_uint64s(const Encryptor& encryptor, const BatchEncoder& encoder, const uint64_t* inputs) const;
        Cipher2d encrypt_inputs_doubles(const Encryptor& encryptor, const CKKSEncoder& encoder,  const double* inputs, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Cipher2d encrypt_inputs_ring2k(const Encryptor& encryptor, const PolynomialEncoderRing2k<T>& encoder, const T* inputs, std::optional<ParmsID> parms_id) const;

        Cipher2d conv2d(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& w) const;
        Cipher2d conv2d_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& w) const;
        Cipher2d conv2d_reverse(const Evaluator& evaluator, const Plain2d& a, const Cipher2d& w) const;

        Plain2d encode_outputs_uint64s(const BatchEncoder& encoder, const uint64_t* outputs) const;
        Plain2d encode_outputs_doubles(const CKKSEncoder& encoder, const double* outputs, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Plain2d encode_outputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* outputs, std::optional<ParmsID> parms_id) const;

        std::vector<uint64_t> decrypt_outputs_uint64s(const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;
        std::vector<double> decrypt_outputs_doubles(const CKKSEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;
        template <typename T>
        std::vector<T> decrypt_outputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;

        void serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream, CompressionMode mode = CompressionMode::Nil) const;
        Cipher2d deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const;

    };

    std::ostream& operator<<(std::ostream& os, const Conv2dHelper& helper);

}}