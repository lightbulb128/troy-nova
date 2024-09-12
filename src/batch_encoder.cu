#include "batch_encoder.h"
#include "encryption_parameters.h"
#include "utils/basics.h"
#include "utils/constants.h"
#include "utils/memory_pool.h"
#include "utils/scaling_variant.h"

namespace troy {

    using utils::Array;
    using utils::Slice; 
    using utils::ConstSlice;

    BatchEncoder::BatchEncoder(HeContextPointer context) {
        if (context->on_device()) {
            throw std::invalid_argument("[BatchEncoder::BatchEncoder] Cannot create from device context.");
        }
        if (!context->parameters_set()) {
            throw std::invalid_argument("[BatchEncoder::BatchEncoder] Encryption parameters are not set correctly.");
        }
        
        ContextDataPointer context_data = context->first_context_data().value();
        const EncryptionParameters& parms = context_data->parms();

        if (parms.scheme() != SchemeType::BFV && parms.scheme() != SchemeType::BGV) {
            throw std::invalid_argument("[BatchEncoder::BatchEncoder] Unsupported scheme.");
        }

        size_t slots = parms.poly_modulus_degree();
        Array<uint64_t> roots_of_unity;
        Array<size_t> matrix_reps_index_map;

        if (context_data->qualifiers().using_batching) {
            roots_of_unity = Array<uint64_t>(slots, false, nullptr);
            const Modulus& modulus = *parms.plain_modulus();
            uint64_t root = context_data->plain_ntt_tables()->root();
            uint64_t generator_sq = utils::multiply_uint64_mod(root, root, modulus);
            roots_of_unity[0] = root;
            for (size_t i = 1; i < slots; i++) {
                roots_of_unity[i] = utils::multiply_uint64_mod(roots_of_unity[i - 1], generator_sq, modulus);
            }
            int logn_int = utils::get_power_of_two(static_cast<uint64_t>(slots));
            if (logn_int < 0) {
                throw std::invalid_argument("[BatchEncoder::BatchEncoder] Slots must be a power of two.");
            }
            size_t logn = static_cast<size_t>(logn_int);
            matrix_reps_index_map = Array<size_t>(slots, false, nullptr);
            size_t row_size = slots >> 1;
            size_t m = slots << 1;
            size_t gen = utils::GALOIS_GENERATOR; size_t pos = 1;
            for (size_t i = 0; i < row_size; i++) {
                size_t index1 = (pos - 1) >> 1;
                size_t index2 = (m - pos - 1) >> 1;
                matrix_reps_index_map[i] = utils::reverse_bits_uint64(static_cast<uint64_t>(index1), logn);
                matrix_reps_index_map[i + row_size] = utils::reverse_bits_uint64(static_cast<uint64_t>(index2), logn);
                pos = (pos * gen) & (m - 1);
            }
        }

        this->context_ = context;
        this->matrix_reps_index_map = std::move(matrix_reps_index_map);
        this->slots_ = slots;

    }

    __global__ static
    void kernel_reverse_bits(size_t logn, Slice<uint64_t> input) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < input.size()) {
            size_t j = utils::reverse_bits_uint64(static_cast<uint64_t>(i), logn);
            if (i < j) {
                uint64_t temp = input[i];
                input[i] = input[j];
                input[j] = temp;
            }
        }
    }

    void BatchEncoder::reverse_bits(utils::Slice<uint64_t> input) {
        size_t n = input.size();
        int logn_int = utils::get_power_of_two(static_cast<uint64_t>(n));
        if (logn_int < 0) {
            throw std::invalid_argument("[BatchEncoder::reverse_bits] input size must be a power of two.");
        }
        size_t logn = static_cast<size_t>(logn_int);
        bool device = input.on_device();
        if (!device) {
            for (size_t i = 0; i < n; i++) {
                size_t j = utils::reverse_bits_uint64(static_cast<uint64_t>(i), logn);
                if (i < j) {
                    std::swap(input[i], input[j]);
                }
            }
        } else {
            size_t block_count = utils::ceil_div(n, utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_reverse_bits<<<block_count, utils::KERNEL_THREAD_COUNT>>>(logn, input);
            utils::stream_sync();
        }
    }

    __device__ static void device_encode_set_values(
        ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < values.size()) {
            destination[index_map[i]] = values[i];
        } else if (i < destination.size()) {
            destination[index_map[i]] = 0;
        }
    }
    __global__ static void kernel_encode_set_values(
        ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination
    ) {
        device_encode_set_values(values, index_map, destination);
    }
    __global__ static void kernel_encode_set_values_batched(
        utils::ConstSliceArrayRef<uint64_t> values, ConstSlice<size_t> index_map, utils::SliceArrayRef<uint64_t> destination
    ) {
        size_t i = blockIdx.y;
        device_encode_set_values(values[i], index_map, destination[i]);
    }

    static void encode_set_values(ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination) {
        size_t device = index_map.on_device();
        if (!utils::device_compatible(values, index_map, destination)) {
            throw std::invalid_argument("[BatchEncoder::encode_set_values] All inputs must reside on same device.");
        }
        if (!device) {
            for (size_t i = 0; i < values.size(); i++) {
                destination[index_map[i]] = values[i];
            }
            for (size_t i = values.size(); i < destination.size(); i++) {
                destination[index_map[i]] = 0;
            }
        } else {
            size_t block_count = utils::ceil_div(destination.size(), utils::KERNEL_THREAD_COUNT);
            utils::set_device(values.device_index());
            kernel_encode_set_values<<<block_count, utils::KERNEL_THREAD_COUNT>>>(values, index_map, destination);
            utils::stream_sync();
        }
    }
    static void encode_set_values_batched(const utils::ConstSliceVec<uint64_t>& values, ConstSlice<size_t> index_map, const utils::SliceVec<uint64_t>& destination, MemoryPoolHandle pool) {
        if (values.size() != destination.size()) {
            throw std::invalid_argument("[BatchEncoder::encode_set_values_batched] Values and destination must have the same size.");
        }
        if (values.size() == 0) return;
        if (!index_map.on_device() || values.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < values.size(); i++) {
                encode_set_values(values[i], index_map, destination[i]);
            }
        } else {
            auto values_batched = batch_utils::construct_batch(values, pool, index_map);
            auto destination_batched = batch_utils::construct_batch(destination, pool, index_map);
            size_t destination_size = destination[0].size();
            for (size_t i = 0; i < values.size(); i++) {
                if (destination[i].size() != destination_size) {
                    throw std::invalid_argument("[BatchEncoder::encode_set_values_batched] All destination slices must have the same size.");
                }
            }
            size_t block_count = utils::ceil_div(destination_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(index_map.device_index());
            dim3 block_dims(block_count, values.size());
            kernel_encode_set_values_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(values_batched, index_map, destination_batched);
            utils::stream_sync();
        }
    }
    
    void BatchEncoder::encode_slice(utils::ConstSlice<uint64_t> values, Plaintext& destination, MemoryPoolHandle pool) const {

        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Memory pool is not compatible with device.");
        }

        // if values and this is not on the same device, convert first
        if (this->on_device() && (!values.on_device() || values.device_index() != this->device_index())) {
            Array<uint64_t> values_device = Array<uint64_t>::create_and_copy_from_slice(values, true, pool);
            encode_slice(values_device.const_reference(), destination, pool);
            return;
        } else if (!this->on_device() && values.on_device()) {
            Array<uint64_t> values_host = Array<uint64_t>::create_and_copy_from_slice(values, false, nullptr);
            encode_slice(values_host.const_reference(), destination, pool);
            return;
        }

        // check compatible
        if (!utils::device_compatible(values, *this)) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Values and destination are not compatible.");
        }

        if (this->matrix_reps_index_map.size() == 0) {
            throw std::logic_error("[BatchEncoder::encode_slice] The parameters does not support vector batching.");
        }
        ContextDataPointer context_data = this->context()->first_context_data().value();
        size_t value_size = values.size();
        if (value_size > this->slot_count()) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Values has size larger than the number of slots.");
        }
        // Set destination to full size
        size_t slots = this->slot_count();
        bool device = this->on_device();
        if (device) {destination.to_device_inplace(pool);}
        else {destination.to_host_inplace();}
        destination.parms_id() = parms_id_zero;
        destination.resize(slots, false, false);
        destination.poly_modulus_degree() = slots;
        destination.coeff_modulus_size() = 1;
        destination.is_ntt_form() = false;
        // First write the values to destination coefficients.
        // Read in top row, then bottom row.
        encode_set_values(
            values,
            this->matrix_reps_index_map.const_reference(), 
            destination.poly()
        );
        // Transform destination using inverse of negacyclic NTT
        // Note: We already performed bit-reversal when reading in the matrix
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Context and destination must reside on same device.");
        }
        utils::intt_inplace(
            destination.poly(),
            slots,
            context_data->plain_ntt_tables()
        );
    }


    void BatchEncoder::encode_slice_batched(const utils::ConstSliceVec<uint64_t>& values, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {

        if (values.size() != destination.size()) {
            throw std::invalid_argument("[BatchEncoder::encode_slice_batched] Values and destination must have the same size.");
        }
        if (values.size() == 0) return;
        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Memory pool is not compatible with device.");
        }
        if (!this->on_device() || values.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < values.size(); i++) {
                encode_slice(values[i], *destination[i], pool);
            }
            return;
        }
        
        // all slices must be on device and compatible with this
        for (size_t i = 0; i < values.size(); i++) {
            if (!values[i].on_device() || values[i].device_index() != this->device_index()) {
                throw std::invalid_argument("[BatchEncoder::encode_slice_batched] All slices must reside on device.");
            }
        }

        if (this->matrix_reps_index_map.size() == 0) {
            throw std::logic_error("[BatchEncoder::encode_slice] The parameters does not support vector batching.");
        }
        ContextDataPointer context_data = this->context()->first_context_data().value();
        size_t value_size = values.size();
        if (value_size > this->slot_count()) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Values has size larger than the number of slots.");
        }
        // Set destination to full size
        size_t slots = this->slot_count();
        bool device = this->on_device();

        size_t n = values.size();
        for (size_t i = 0; i < n; i++) {
            *destination[i] = Plaintext();
            if (device) {destination[i]->to_device_inplace(pool);}
            else {destination[i]->to_host_inplace();}
            destination[i]->parms_id() = parms_id_zero;
            destination[i]->resize(slots, false, false);
            destination[i]->poly_modulus_degree() = slots;
            destination[i]->coeff_modulus_size() = 1;
            destination[i]->is_ntt_form() = false;
        }

        // First write the values to destination coefficients.
        // Read in top row, then bottom row.
        auto destination_poly = batch_utils::pcollect_poly(destination);
        encode_set_values_batched(
            values,
            this->matrix_reps_index_map.const_reference(), 
            destination_poly, pool
        );
        
        utils::intt_inplace_b(
            destination_poly,
            slots,
            context_data->plain_ntt_tables(),
            pool
        );
    }




    void BatchEncoder::encode_polynomial_slice(utils::ConstSlice<uint64_t> values, Plaintext& destination, MemoryPoolHandle pool) const {
        
        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Memory pool is not compatible with device.");
        }

        // if values and this is not on the same device, convert first
        if (this->on_device() && (!values.on_device() || values.device_index() != this->device_index())) {
            Array<uint64_t> values_device = Array<uint64_t>::create_and_copy_from_slice(values, true, pool);
            encode_polynomial_slice(values_device.const_reference(), destination, pool);
            return;
        } else if (!this->on_device() && values.on_device()) {
            Array<uint64_t> values_host = Array<uint64_t>::create_and_copy_from_slice(values, false, nullptr);
            encode_polynomial_slice(values_host.const_reference(), destination, pool);
            return;
        }

        // check compatible
        if (!utils::device_compatible(values, *this)) {
            throw std::invalid_argument("[BatchEncoder::encode_polynomial_slice] Values and destination are not compatible.");
        }

        ContextDataPointer context_data = this->context()->first_context_data().value();
        size_t value_size = values.size();
        if (value_size > this->slot_count()) {
            throw std::invalid_argument("[BatchEncoder::encode_polynomial_slice] Values has size larger than the number of slots.");
        }
        // Set destination to full size
        bool device = this->on_device();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[BatchEncoder::encode_polynomial_slice] Context and destination must reside on same device.");
        }
        size_t slots = this->slot_count();
        if (device) {destination.to_device_inplace(pool);}
        else {destination.to_host_inplace();}
        destination.parms_id() = parms_id_zero;
        destination.resize(value_size, false, false);
        destination.poly_modulus_degree() = slots;
        destination.coeff_modulus_size() = 1;
        destination.is_ntt_form() = false;
        utils::ConstPointer<Modulus> plain_modulus = context_data->parms().plain_modulus();
        utils::modulo(
            values,
            plain_modulus, destination.poly().slice(0, value_size)
        );
    }

    __device__ static void device_decode_set_values(
        ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < values.size()) {
            destination[i] = values[index_map[i]];
        }
    }

    __global__ static void kernel_decode_set_values(
        ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination
    ) {
        device_decode_set_values(values, index_map, destination);
    }
    __global__ static void kernel_decode_set_values_batched(
        utils::ConstSliceArrayRef<uint64_t> values, ConstSlice<size_t> index_map, utils::SliceArrayRef<uint64_t> destination
    ) {
        size_t i = blockIdx.y;
        device_decode_set_values(values[i], index_map, destination[i]);
    }

    static void decode_set_values(ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination) {
        size_t device = index_map.on_device();
        if (!utils::device_compatible(values, index_map, destination)) {
            throw std::invalid_argument("[BatchEncoder::decode_set_values] All inputs must reside on same device.");
        }
        if (!device) {
            for (size_t i = 0; i < values.size(); i++) {
                destination[i] = values[index_map[i]];
            }
        } else {
            size_t block_count = utils::ceil_div(destination.size(), utils::KERNEL_THREAD_COUNT);
            utils::set_device(values.device_index());
            kernel_decode_set_values<<<block_count, utils::KERNEL_THREAD_COUNT>>>(values, index_map, destination);
            utils::stream_sync();
        }
    }
    
    static void decode_set_values_batched(const utils::ConstSliceVec<uint64_t>& values, ConstSlice<size_t> index_map, const utils::SliceVec<uint64_t>& destination, MemoryPoolHandle pool) {
        if (values.size() != destination.size()) {
            throw std::invalid_argument("[BatchEncoder::decode_set_values_batched] Values and destination must have the same size.");
        }
        if (values.size() == 0) return;
        if (!index_map.on_device() || values.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < values.size(); i++) {
                decode_set_values(values[i], index_map, destination[i]);
            }
        } else {
            auto values_batched = batch_utils::construct_batch(values, pool, index_map);
            auto destination_batched = batch_utils::construct_batch(destination, pool, index_map);            
            size_t destination_size = destination[0].size();
            for (size_t i = 0; i < values.size(); i++) {
                if (destination[i].size() != destination_size) {
                    throw std::invalid_argument("[BatchEncoder::encode_set_values_batched] All destination slices must have the same size.");
                }
            }
            size_t block_count = utils::ceil_div(destination_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(index_map.device_index());
            dim3 block_dims(block_count, values.size());
            kernel_decode_set_values_batched<<<block_dims, utils::KERNEL_THREAD_COUNT>>>(values_batched, index_map, destination_batched);
            utils::stream_sync();
        }
    }
    
    void BatchEncoder::decode_slice(const Plaintext& plain, Slice<uint64_t> destination, MemoryPoolHandle pool) const {

        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[BatchEncoder::encode_slice] Memory pool is not compatible with device.");
        }

        // if values and this is not on the same device, convert first
        if (this->on_device() && (!destination.on_device() || destination.device_index() != this->device_index())) {
            Array<uint64_t> destination_device = Array<uint64_t>(destination.size(), true, pool);
            decode_slice(plain, destination_device.reference(), pool);
            destination.copy_from_slice(destination_device.const_reference());
            return;
        } else if (!this->on_device() && destination.on_device()) {
            Array<uint64_t> destination_host = Array<uint64_t>(destination.size(), false, nullptr);
            decode_slice(plain, destination_host.reference(), pool);
            destination.copy_from_slice(destination_host.const_reference());
            return;
        }

        // check compatible
        if (!utils::device_compatible(destination, *this)) {
            throw std::invalid_argument("[BatchEncoder::decode_slice] Values and destination are not compatible.");
        }

        if (this->matrix_reps_index_map.size() == 0) {
            throw std::logic_error("[BatchEncoder::encode] The parameters does not support vector batching.");
        }
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("[BatchEncoder::decode] Plaintext is in NTT form.");
        }
        ContextDataPointer context_data = this->context()->first_context_data().value();
        size_t slots = this->slot_count();
        if (destination.size() != slots) {
            throw std::invalid_argument("[BatchEncoder::decode] Destination has incorrect size.");
        }
        size_t plain_coeff_count = std::min(plain.coeff_count(), slots);
        Array<uint64_t> temp_dest(slots, plain.on_device(), pool);
        temp_dest.slice(0, plain_coeff_count).copy_from_slice(plain.poly());
        // Transform destination using negacyclic NTT
        bool device = this->on_device();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[BatchEncoder::decode] Context and destination must reside on same device.");
        }
        utils::ntt_inplace(
            temp_dest.reference(),
            slots,
            context_data->plain_ntt_tables()
        );
        decode_set_values(
            temp_dest.const_reference(),
            this->matrix_reps_index_map.const_reference(), 
            destination
        );
    }
    

    void BatchEncoder::decode_slice_batched(const std::vector<const Plaintext*>& plain, const utils::SliceVec<uint64_t>& destination, MemoryPoolHandle pool) const {

        if (plain.size() != destination.size()) {
            throw std::invalid_argument("[BatchEncoder::decode_slice_batched] Values and destination must have the same size.");
        }
        if (plain.size() == 0) return;
        if (!this->on_device() || plain.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < plain.size(); i++) {
                decode_slice(*plain[i], destination[i], pool);
            }
            return;
        }

        if (!pool_compatible(pool)) {
            throw std::invalid_argument("[BatchEncoder::decode_slice_batched] Memory pool is not compatible with device.");
        }

        // all destination slices must be on device and compatible with this
        size_t slots = this->slot_count();
        for (size_t i = 0; i < destination.size(); i++) {
            if (!destination[i].on_device() || destination[i].device_index() != this->device_index()) {
                throw std::invalid_argument("[BatchEncoder::decode_slice_batched] All slices must reside on device.");
            }
            // destination size must be slots
            if (destination[i].size() != slots) {
                throw std::invalid_argument("[BatchEncoder::decode_slice_batched] Destination has incorrect size.");
            }
        }

        // all plain must not be ntt form
        for (size_t i = 0; i < plain.size(); i++) {
            if (plain[i]->is_ntt_form()) {
                throw std::invalid_argument("[BatchEncoder::decode_slice_batched] Plaintext is in NTT form.");
            }
        }

        if (this->matrix_reps_index_map.size() == 0) {
            throw std::logic_error("[BatchEncoder::decode_slice_batched] The parameters does not support vector batching.");
        }
        ContextDataPointer context_data = this->context()->first_context_data().value();

        size_t plain_coeff_count = plain[0]->coeff_count();
        // all must have same coeff count
        for (size_t i = 0; i < plain.size(); i++) {
            if (plain[i]->coeff_count() != plain_coeff_count) {
                throw std::invalid_argument("[BatchEncoder::decode_slice_batched] All plaintexts must have the same coefficient count.");
            }
        }
        
        plain_coeff_count = std::min(plain_coeff_count, slots);

        size_t n = plain.size();
        std::vector<Array<uint64_t>> temp_dest; temp_dest.reserve(n); 
        std::vector<Slice<uint64_t>> temp_copies; temp_copies.reserve(n);
        std::vector<Slice<uint64_t>> temp_set_zeros; temp_set_zeros.reserve(n);
        for (size_t i = 0; i < n; i++) {
            temp_dest.push_back(Array<uint64_t>::create_uninitialized(slots, true, pool));
            temp_copies.push_back(temp_dest[i].slice(0, plain_coeff_count));
            temp_set_zeros.push_back(temp_dest[i].slice(plain_coeff_count, slots));
        }

        utils::copy_slice_b(batch_utils::pcollect_const_poly(plain), temp_copies, pool);
        utils::set_slice_b(0, temp_set_zeros, pool);

        // Transform destination using negacyclic NTT
        bool device = this->on_device();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[BatchEncoder::decode] Context and destination must reside on same device.");
        }
        utils::ntt_inplace_b(
            batch_utils::rcollect_reference(temp_dest),
            slots,
            context_data->plain_ntt_tables(),
            pool
        );
        decode_set_values_batched(
            batch_utils::rcollect_const_reference(temp_dest),
            this->matrix_reps_index_map.const_reference(), 
            destination,
            pool
        );
    }
    


    void BatchEncoder::decode_polynomial_slice(const Plaintext& plaintext, utils::Slice<uint64_t> destination) const {
        if (plaintext.is_ntt_form() || plaintext.parms_id() != parms_id_zero) {
            throw std::invalid_argument("[BatchEncoder::decode_polynomial_slice] Plaintext is not in valid form.");
        }
        if (destination.size() != plaintext.data().size()) {
            throw std::invalid_argument("[BatchEncoder::decode_polynomial_slice] Destination has incorrect size.");
        }
        destination.copy_from_slice(plaintext.data().const_reference());
    }

    Plaintext BatchEncoder::scale_up_new(const Plaintext& plain, std::optional<ParmsID> parms_id, MemoryPoolHandle pool) const {
        if (this->context_->first_context_data().value()->parms().scheme() != SchemeType::BFV) {
            throw std::logic_error("[BatchEncoder::scale_up_new] Only BFV scheme is supported.");
        }
        if (plain.parms_id() != parms_id_zero) {
            throw std::invalid_argument("[BatchEncoder::scale_up_new] Plaintext is already at the desired level.");
        }
        ParmsID pid = parms_id.value_or(this->context_->first_parms_id());
        ContextDataPointer context_data = this->context_->get_context_data(pid).value();
        Plaintext destination;
        if (plain.on_device()) {
            destination.to_device_inplace(pool);
        } else {
            destination.to_host_inplace();
        }
        destination.resize_rns_partial(*this->context_, pid, plain.coeff_count());
        destination.is_ntt_form() = false;
        scaling_variant::scale_up(plain, context_data, destination.reference(), plain.coeff_count(), nullptr, false);
        return destination;
    }

    Plaintext BatchEncoder::scale_down_new(const Plaintext& plain, MemoryPoolHandle pool) const {
        if (this->context_->first_context_data().value()->parms().scheme() != SchemeType::BFV) {
            throw std::logic_error("[BatchEncoder::scale_down_new] Only BFV scheme is supported.");
        }
        if (plain.parms_id() == parms_id_zero) {
            throw std::invalid_argument("[BatchEncoder::scale_down_new] Plaintext not in RNS form.");
        }
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("[BatchEncoder::scale_down_new] Plaintext is in NTT form.");
        }
        Plaintext destination;
        if (plain.on_device()) {
            destination.to_device_inplace(pool);
        } else {
            destination.to_host_inplace();
        }
        destination.coeff_modulus_size() = plain.coeff_modulus_size();
        destination.poly_modulus_degree() = plain.poly_modulus_degree();
        destination.coeff_count() = plain.coeff_count();
        destination.parms_id() = parms_id_zero;
        destination.resize(plain.coeff_count());
        destination.is_ntt_form() = false;
        std::optional<ContextDataPointer> context_data_opt = this->context_->get_context_data(plain.parms_id());
        if (!context_data_opt.has_value()) {
            throw std::invalid_argument("[BatchEncoder::scale_down_new] Could not find context data.");
        }
        ContextDataPointer context_data = context_data_opt.value();
        scaling_variant::scale_down(plain, context_data, destination.reference(), pool);
        return destination;
    }

    Plaintext BatchEncoder::centralize_new(const Plaintext& plain, std::optional<ParmsID> parms_id, MemoryPoolHandle pool) const {
        SchemeType scheme = this->context_->first_context_data().value()->parms().scheme();
        if (scheme != SchemeType::BFV && scheme != SchemeType::BGV) {
            throw std::logic_error("[BatchEncoder::centralize_new] Only BFV/BGV scheme is supported.");
        }
        if (plain.parms_id() != parms_id_zero) {
            throw std::invalid_argument("[BatchEncoder::centralize_new] Plaintext is already at the desired level.");
        }
        ParmsID pid = parms_id.value_or(this->context_->first_parms_id());
        ContextDataPointer context_data = this->context_->get_context_data(pid).value();
        Plaintext destination;
        if (plain.on_device()) {
            destination.to_device_inplace(pool);
        } else {
            destination.to_host_inplace();
        }
        destination.resize_rns_partial(*this->context_, pid, plain.coeff_count());
        destination.is_ntt_form() = false;
        scaling_variant::centralize(plain, context_data, destination.reference(), plain.coeff_count(), pool);
        return destination;
    }

    Plaintext BatchEncoder::decentralize_new(const Plaintext& plain, uint64_t correction_factor, MemoryPoolHandle pool) const {
        SchemeType scheme = this->context_->first_context_data().value()->parms().scheme();
        if (scheme != SchemeType::BFV && scheme != SchemeType::BGV) {
            throw std::logic_error("[BatchEncoder::decentralize_new] Only BFV/BGV scheme is supported.");
        }
        if (plain.parms_id() == parms_id_zero) {
            throw std::invalid_argument("[BatchEncoder::decentralize_new] Plaintext not in RNS form.");
        }
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("[BatchEncoder::decentralize_new] Plaintext is in NTT form.");
        }
        Plaintext destination;
        if (plain.on_device()) {
            destination.to_device_inplace(pool);
        } else {
            destination.to_host_inplace();
        }
        destination.coeff_modulus_size() = plain.coeff_modulus_size();
        destination.poly_modulus_degree() = plain.poly_modulus_degree();
        destination.coeff_count() = plain.coeff_count();
        destination.parms_id() = parms_id_zero;
        destination.resize(plain.coeff_count());
        destination.is_ntt_form() = false;
        std::optional<ContextDataPointer> context_data_opt = this->context_->get_context_data(plain.parms_id());
        if (!context_data_opt.has_value()) {
            throw std::invalid_argument("[BatchEncoder::decentralize_new] Could not find context data.");
        }
        ContextDataPointer context_data = context_data_opt.value();
        scaling_variant::decentralize(plain, context_data, destination.reference(), correction_factor, pool);
        return destination;
    }

}
