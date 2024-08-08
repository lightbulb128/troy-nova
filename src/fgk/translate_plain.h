#include "../evaluator.h"

namespace troy::utils::fgk::translate_plain {
    void multiply_translate_plain(utils::ConstSlice<uint64_t> from, const Plaintext& plain, ContextDataPointer context_data, utils::Slice<uint64_t> destination, bool subtract);
    void scatter_translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract);
    inline void translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract) {
        scatter_translate_copy(from, translation, degree, degree, moduli, destination, subtract);
    }
}