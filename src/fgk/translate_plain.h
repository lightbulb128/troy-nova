#include "../evaluator.h"

namespace troy::utils::fgk::translate_plain {

    void multiply_translate_plain(
        utils::ConstSlice<uint64_t> from, 
        const Plaintext& plain, 
        ContextDataPointer context_data, 
        utils::Slice<uint64_t> destination, 
        size_t destination_coeff_count, bool subtract
    );
    
    void multiply_translate_plain_batched(
        const utils::ConstSliceVec<uint64_t>& from, 
        const std::vector<const Plaintext*> plain, 
        ContextDataPointer context_data, 
        const utils::SliceVec<uint64_t>& destination, 
        size_t destination_coeff_count, bool subtract,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );

    void scatter_translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract);
    void scatter_translate_copy_batched(
        const ConstSliceVec<uint64_t>& from, 
        const ConstSliceVec<uint64_t>& translation, size_t from_degree, size_t translation_degree, ConstSlice<Modulus> moduli, 
        const SliceVec<uint64_t>& destination, bool subtract,
        MemoryPoolHandle pool = MemoryPool::GlobalPool()
    );

    inline void translate_copy(ConstSlice<uint64_t> from, ConstSlice<uint64_t> translation, size_t degree, ConstSlice<Modulus> moduli, Slice<uint64_t> destination, bool subtract) {
        scatter_translate_copy(from, translation, degree, degree, moduli, destination, subtract);
    }
}