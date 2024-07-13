#pragma once
#include <cassert>
#include "../batch_encoder.h"
#include "../ckks_encoder.h"
#include "cipher2d.h"
#include "encoder_adapter.h"

namespace troy { namespace linear {

    using utils::ceil_div;

    enum class MatmulObjective : uint8_t {
        EncryptLeft = 0,
        EncryptRight = 1,
        Crossed = 2
    };

    class MatmulHelper {
        // 0: encrypt inputs; 1: encrypt weights
        // 2: for calculating weight gradient

        void determine_block();

        template <typename E, typename T>
        Plaintext encode_weights_small(const E& encoder, const T* weights, size_t li, size_t ui, size_t lj, size_t uj, bool for_cipher) const;

        template <typename E, typename T>
        Plaintext encode_inputs_small(const E& encoder, const T* inputs, size_t li, size_t ui, size_t lj, size_t uj, bool for_cipher) const;

        template <typename E, typename T>
        void encode_weights(const E& encoder, const Encryptor* encryptor, const T* weights, bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher) const;

        template <typename E, typename T>
        void encode_inputs(const E& encoder, const Encryptor* encryptor, const T* inputs, bool for_cipher, Plain2d* out_plain, Cipher2d* out_cipher) const;

        template <typename E, typename T>
        Plain2d encode_outputs(const E& encoder, const T* outputs) const;

        template <typename E, typename T>
        std::vector<T> decrypt_outputs(const E& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;

    public:
    
        size_t batch_size, input_dims, output_dims;
        size_t slot_count;
        size_t batch_block, input_block, output_block;
        MatmulObjective objective; 
        bool pack_lwe;
        bool store_in_host;
        MemoryPoolHandle pool;

        inline void set_pool(MemoryPoolHandle pool) {
            this->pool = pool;
        }

        inline void set_store_in_host(bool store_in_host) {
            if (store_in_host && !pack_lwe) {
                throw std::runtime_error("[MatmulHelper::set_store_in_host] Cannot store in host if pack_lwe is false");
            }
            this->store_in_host = store_in_host;
        }

        inline MatmulHelper(size_t batch_size, size_t input_dims, size_t output_dims, size_t slot_count, MatmulObjective objective = MatmulObjective::EncryptLeft, bool pack_lwe = true, MemoryPoolHandle pool = MemoryPool::GlobalPool()):
            batch_size(batch_size), input_dims(input_dims), output_dims(output_dims),
            slot_count(slot_count), objective(objective), pack_lwe(pack_lwe), store_in_host(false), pool(pool)
        {
            determine_block();
        }

        Plain2d encode_weights_uint64s(const BatchEncoder& encoder, const uint64_t* weights) const;
        Plain2d encode_weights_doubles(const CKKSEncoder& encoder, const double* weights, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Plain2d encode_weights_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* weights, std::optional<ParmsID> parms_id, bool for_cipher) const;

        Cipher2d encrypt_weights_uint64s(const Encryptor& encryptor, const BatchEncoder& encoder, const uint64_t* weights) const;
        Cipher2d encrypt_weights_doubles(const Encryptor& encryptor, const CKKSEncoder& encoder, const double* weights, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Cipher2d encrypt_weights_ring2k(const Encryptor& encryptor, const PolynomialEncoderRing2k<T>& encoder, const T* weights, std::optional<ParmsID> parms_id) const;

        Plain2d encode_inputs_uint64s(const BatchEncoder& encoder, const uint64_t* inputs) const;
        Plain2d encode_inputs_doubles(const CKKSEncoder& encoder, const double* inputs, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Plain2d encode_inputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* inputs, std::optional<ParmsID> parms_id, bool for_cipher) const;

        Cipher2d encrypt_inputs_uint64s(const Encryptor& encryptor, const BatchEncoder& encoder, const uint64_t* inputs) const;
        Cipher2d encrypt_inputs_doubles(const Encryptor& encryptor, const CKKSEncoder& encoder,  const double* inputs, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Cipher2d encrypt_inputs_ring2k(const Encryptor& encryptor, const PolynomialEncoderRing2k<T>& encoder, const T* inputs, std::optional<ParmsID> parms_id) const;

        Cipher2d matmul(const Evaluator& evaluator, const Cipher2d& a, const Plain2d& w) const;
        Cipher2d matmul_cipher(const Evaluator& evaluator, const Cipher2d& a, const Cipher2d& w) const;
        Cipher2d matmul_reverse(const Evaluator& evaluator, const Plain2d& a, const Cipher2d& w) const;

        Plain2d encode_outputs_uint64s(const BatchEncoder& encoder, const uint64_t* outputs) const;
        Plain2d encode_outputs_doubles(const CKKSEncoder& encoder, const double* outputs, std::optional<ParmsID> parms_id, double scale) const;
        template <typename T>
        Plain2d encode_outputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const T* outputs, std::optional<ParmsID> parms_id) const;

        std::vector<uint64_t> decrypt_outputs_uint64s(const BatchEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;
        std::vector<double> decrypt_outputs_doubles(const CKKSEncoder& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;
        template <typename T>
        std::vector<T> decrypt_outputs_ring2k(const PolynomialEncoderRing2k<T>& encoder, const Decryptor& decryptor, const Cipher2d& outputs) const;

        Cipher2d pack_outputs(const Evaluator& evaluator, const GaloisKeys& autoKey, const Cipher2d& cipher) const;

        void serialize_encoded_weights(const Plain2d& w, std::ostream& stream, CompressionMode mode = CompressionMode::Nil) const;
        Plain2d deserialize_encoded_weights(std::istream& stream) const;
        void serialize_outputs(const Evaluator &evaluator, const Cipher2d& x, std::ostream& stream, CompressionMode mode = CompressionMode::Nil) const;
        Cipher2d deserialize_outputs(const Evaluator &evaluator, std::istream& stream) const;

    };

    inline std::ostream& operator<<(std::ostream& os, const MatmulObjective& obj) {
        switch (obj) {
            case MatmulObjective::EncryptLeft: os << "EncryptLeft"; break;
            case MatmulObjective::EncryptRight: os << "EncryptRight"; break;
            case MatmulObjective::Crossed: os << "Crossed"; break;
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const MatmulHelper& helper) {
        os << "MatmulHelper(batch_size=" << helper.batch_size << ", input_dims=" << helper.input_dims << ", output_dims=" << helper.output_dims
           << ", slot_count=" << helper.slot_count << ", objective=" << helper.objective << ", pack_lwe=" << helper.pack_lwe << ", store_in_host=" << helper.store_in_host << ")";
        return os;
    }


}}