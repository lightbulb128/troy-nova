#include "encoder_adapter.h"

namespace troy::linear {

    // provide instantiation for PolynomialEncoderRing2kAdapter
    template class PolynomialEncoderRing2kAdapter<uint32_t>;
    template class PolynomialEncoderRing2kAdapter<uint64_t>;
    template class PolynomialEncoderRing2kAdapter<__uint128_t>;

}