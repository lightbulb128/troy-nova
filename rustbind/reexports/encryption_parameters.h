#pragma once
#include <memory>
#include "cuda_runtime.h"
#include "memory_pool.h"
#include "troy/troy.cuh"
#include "rust/cxx.h"

namespace troy_rust {

    class ParmsID;
    typedef std::unique_ptr<ParmsID> UpParmsID;

    class ParmsID {
    private:
        troy::ParmsID p;
    public:
        explicit inline ParmsID(troy::ParmsID p) : p(p) {}
        ParmsID(const ParmsID& p) = default;
        ParmsID(ParmsID&& p) = default;
        ParmsID& operator=(const ParmsID& p) = default;
        ParmsID& operator=(ParmsID&& p) = default;
        inline bool equals_to(const ParmsID& other) const {
            return p == other.p;
        }
        inline std::array<uint64_t, 4> to_array() const {
            return p;
        }
        inline bool is_zero() const {
            return p == troy::parms_id_zero;
        }
    };

    inline UpParmsID parms_id_constructor_copy(const ParmsID& p) {
        return std::make_unique<ParmsID>(p);
    }
    inline UpParmsID parms_id_static_zero() {
        return std::make_unique<ParmsID>(troy::parms_id_zero);
    }

    // class EncryptionParameters;
    // typedef std::unique_ptr<EncryptionParameters> UpEncryptionParameters;
    
    // class EncryptionParameters {
    // private:
    //     troy::EncryptionParameters p;
    // public:

    //     explicit inline EncryptionParameters(troy::EncryptionParameters p) : p(p) {}
    //     explicit inline EncryptionParameters(troy::SchemeType scheme) {
    //         p = troy::EncryptionParameters(scheme);
    //     }

    //     inline UpMemoryPool pool() const {
    //         return std::make_unique<MemoryPool>(p.pool());
    //     }
    //     inline size_t device_index() const {
    //         return p.device_index();
    //     }
    //     inline bool on_device() const {
    //         return p.on_device();
    //     }

    //     inline void set_poly_modulus_degree(size_t poly_modulus_degree) {
    //         p.set_poly_modulus_degree(poly_modulus_degree);
    //     }
    //     inline void set_plain_modulus_u64(uint64_t plain_modulus) {
    //         p.set_plain_modulus(plain_modulus);
    //     }

    // };

    

}