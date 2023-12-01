#pragma once
#include <cstdint>

namespace troy {namespace he_standard_params {

    inline std::size_t params_classical_128(std::size_t poly_modulus_degree) {
        switch (poly_modulus_degree) {
            case  1024: return  27;
            case  2048: return  54;
            case  4096: return 109;
            case  8192: return 218;
            case 16384: return 438;
            case 32768: return 881;
        }
        return std::size_t(881.0 * poly_modulus_degree / 32768.0);
    }

    inline std::size_t params_classical_192(std::size_t poly_modulus_degree) {
        switch (poly_modulus_degree) {
            case  1024: return  19;
            case  2048: return  37;
            case  4096: return  75;
            case  8192: return 152;
            case 16384: return 305;
            case 32768: return 611;
        }
        return std::size_t(611.0 * poly_modulus_degree / 32768.0);
    }

    inline std::size_t params_classical_256(std::size_t poly_modulus_degree) {
        switch (poly_modulus_degree) {
            case  1024: return  14;
            case  2048: return  29;
            case  4096: return  58;
            case  8192: return 118;
            case 16384: return 237;
            case 32768: return 476;
        }
        return std::size_t(476.0 * poly_modulus_degree / 32768.0);
    }

    inline std::size_t params_quantum_128(std::size_t poly_modulus_degree) {
        switch (poly_modulus_degree) {
            case  1024: return  25;
            case  2048: return  51;
            case  4096: return 101;
            case  8192: return 202;
            case 16384: return 411;
            case 32768: return 827;
        }
        return std::size_t(827.0 * poly_modulus_degree / 32768.0);
    }

    inline std::size_t params_quantum_192(std::size_t poly_modulus_degree) {
        switch (poly_modulus_degree) {
            case  1024: return  17;
            case  2048: return  35;
            case  4096: return  70;
            case  8192: return 141;
            case 16384: return 284;
            case 32768: return 571;
        }
        return std::size_t(571.0 * poly_modulus_degree / 32768.0);
    }

    inline std::size_t params_quantum_256(std::size_t poly_modulus_degree) {
        switch (poly_modulus_degree) {
            case  1024: return  13;
            case  2048: return  27;
            case  4096: return  54;
            case  8192: return 109;
            case 16384: return 220;
            case 32768: return 443;
        }
        return std::size_t(443.0 * poly_modulus_degree / 32768.0);
    }

    const double HE_STANDARD_PARAMS_ERROR_STD_DEV = 3.2;

}}