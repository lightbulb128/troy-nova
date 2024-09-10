#pragma once

#include <cassert>
#include "../batch_encoder.h"
#include "../ckks_encoder.h"
#include "../decryptor.h"
#include "bfv_ring2k.h"

namespace troy::linear {

    class BatchEncoderAdapter {
    private:
        const BatchEncoder& encoder;
    public:
        inline BatchEncoderAdapter(const BatchEncoder& encoder): encoder(encoder) {}
        inline Plaintext encode_for_cipher(const std::vector<uint64_t>& elements, MemoryPoolHandle pool) const {
            return encoder.encode_polynomial_new(elements, pool);
        }
        inline Plaintext encode_for_plain(const std::vector<uint64_t>& elements, MemoryPoolHandle pool) const {
            return encoder.encode_polynomial_new(elements, pool);
        }
        inline std::vector<uint64_t> decrypt_outputs(const Decryptor& decryptor, const Ciphertext& ciphertext, MemoryPoolHandle pool) const {
            return encoder.decode_polynomial_new(decryptor.decrypt_new(ciphertext, pool));
        }
        inline HeContextPointer context() const {return encoder.context();}
    };

    class CKKSEncoderAdapter {
    private:
        const CKKSEncoder& encoder;
        std::optional<ParmsID> parms_id;
        double scale;
    public:
        inline CKKSEncoderAdapter(const CKKSEncoder& encoder, std::optional<ParmsID> parms_id, double scale)
            : encoder(encoder), parms_id(parms_id), scale(scale) {}
        inline Plaintext encode_for_cipher(const std::vector<double>& elements, MemoryPoolHandle pool) const {
            return encoder.encode_float64_polynomial_new(elements, parms_id, scale, pool);
        }
        inline Plaintext encode_for_plain(const std::vector<double>& elements, MemoryPoolHandle pool) const {
            return encoder.encode_float64_polynomial_new(elements, parms_id, scale, pool);
        }
        inline std::vector<double> decrypt_outputs(const Decryptor& decryptor, const Ciphertext& ciphertext, MemoryPoolHandle pool) const {
            return encoder.decode_float64_polynomial_new(decryptor.decrypt_new(ciphertext, pool), pool);
        }
        inline HeContextPointer context() const {return encoder.context();}
    };

    template <typename T>
    class PolynomialEncoderRing2kAdapter {
        static_assert(is_compatible_ring2k<T>::value, "T must be uint32_t, uint64_t or uint128_t.");
    private:
        const PolynomialEncoderRing2k<T>& encoder;
        std::optional<ParmsID> parms_id;
    public:
        inline PolynomialEncoderRing2kAdapter(const PolynomialEncoderRing2k<T>& encoder, std::optional<ParmsID> parms_id)
            : encoder(encoder), parms_id(parms_id) {}
        inline Plaintext encode_for_cipher(const std::vector<T>& elements, MemoryPoolHandle pool) const {
            return encoder.scale_up_new(elements, parms_id, pool);
        }
        inline Plaintext encode_for_plain(const std::vector<T>& elements, MemoryPoolHandle pool) const {
            return encoder.centralize_new(elements, parms_id, pool);
        }
        inline std::vector<T> decrypt_outputs(const Decryptor& decryptor, const Ciphertext& ciphertext, MemoryPoolHandle pool) const {
            return encoder.scale_down_new(decryptor.bfv_decrypt_without_scaling_down_new(ciphertext, pool), pool);
        }
        inline HeContextPointer context() const {return encoder.context();}
    };

}