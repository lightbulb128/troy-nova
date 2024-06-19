#include "batch_encoder.cuh"

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
            roots_of_unity = Array<uint64_t>(slots, false);
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
            matrix_reps_index_map = Array<size_t>(slots, false);
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
            kernel_reverse_bits<<<block_count, utils::KERNEL_THREAD_COUNT>>>(logn, input);
        }
    }

    __global__ static void kernel_encode_set_values(
        ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < values.size()) {
            destination[index_map[i]] = values[i];
        } else if (i < destination.size()) {
            destination[index_map[i]] = 0;
        }
    }

    static void encode_set_values(ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination) {
        size_t device = index_map.on_device();
        if (!utils::same(device, values.on_device(), destination.on_device())) {
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
            kernel_encode_set_values<<<block_count, utils::KERNEL_THREAD_COUNT>>>(values, index_map, destination);
        }
    }
    
    void BatchEncoder::encode(const std::vector<uint64_t>& values, Plaintext& destination) const {
        if (this->matrix_reps_index_map.size() == 0) {
            throw std::logic_error("[BatchEncoder::encode] The parameters does not support vector batching.");
        }
        ContextDataPointer context_data = this->context()->first_context_data().value();
        size_t value_size = values.size();
        if (value_size > this->slot_count()) {
            throw std::invalid_argument("[BatchEncoder::encode] Values has size larger than the number of slots.");
        }
        // Set destination to full size
        size_t slots = this->slot_count();
        bool device = this->on_device();
        if (device) {destination.to_device_inplace();}
        else {destination.to_host_inplace();}
        destination.parms_id() = parms_id_zero;
        destination.resize(slots);
        destination.poly_modulus_degree() = slots;
        destination.coeff_modulus_size() = 1;
        destination.is_ntt_form() = false;
        // First write the values to destination coefficients.
        // Read in top row, then bottom row.
        if (!device) {
            encode_set_values(
                ConstSlice(values.data(), values.size(), false),
                this->matrix_reps_index_map.const_reference(), 
                destination.poly()
            );
        } else {
            Array<uint64_t> values_device(value_size, false);
            for (size_t i = 0; i < value_size; i++) {
                values_device[i] = values[i];
            }
            values_device.to_device_inplace();
            encode_set_values(
                values_device.const_reference(),
                this->matrix_reps_index_map.const_reference(), 
                destination.poly()
            );
        }
        // Transform destination using inverse of negacyclic NTT
        // Note: We already performed bit-reversal when reading in the matrix
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[BatchEncoder::encode] Context and destination must reside on same device.");
        }
        utils::inverse_ntt_negacyclic_harvey(
            destination.poly(),
            slots,
            context_data->plain_ntt_tables()
        );
    }
    

    void BatchEncoder::encode_polynomial(const std::vector<uint64_t>& values, Plaintext& destination) const {
        ContextDataPointer context_data = this->context()->first_context_data().value();
        size_t value_size = values.size();
        if (value_size > this->slot_count()) {
            throw std::invalid_argument("[BatchEncoder::encode] Values has size larger than the number of slots.");
        }
        // Set destination to full size
        bool device = this->on_device();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[BatchEncoder::encode] Context and destination must reside on same device.");
        }
        size_t slots = this->slot_count();
        if (device) {destination.to_device_inplace();}
        else {destination.to_host_inplace();}
        destination.parms_id() = parms_id_zero;
        destination.resize(value_size);
        destination.poly_modulus_degree() = slots;
        destination.coeff_modulus_size() = 1;
        destination.is_ntt_form() = false;
        utils::ConstPointer<Modulus> plain_modulus = context_data->parms().plain_modulus();
        if (!device) {
            utils::modulo(
                ConstSlice(values.data(), values.size(), false),
                plain_modulus, destination.poly().slice(0, value_size)
            );
        } else {
            Array<uint64_t> values_device(value_size, true);
            values_device.copy_from_slice(ConstSlice(values.data(), values.size(), false));
            values_device.to_device_inplace();
            utils::modulo(
                values_device.const_reference(),
                plain_modulus, destination.poly().slice(0, value_size)
            );
        }
    }

    __global__ static void kernel_decode_set_values(
        ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < values.size()) {
            destination[i] = values[index_map[i]];
        }
    }

    static void decode_set_values(ConstSlice<uint64_t> values, ConstSlice<size_t> index_map, Slice<uint64_t> destination) {
        size_t device = index_map.on_device();
        if (!utils::same(device, values.on_device(), destination.on_device())) {
            throw std::invalid_argument("[BatchEncoder::decode_set_values] All inputs must reside on same device.");
        }
        if (!device) {
            for (size_t i = 0; i < values.size(); i++) {
                destination[i] = values[index_map[i]];
            }
        } else {
            size_t block_count = utils::ceil_div(destination.size(), utils::KERNEL_THREAD_COUNT);
            kernel_decode_set_values<<<block_count, utils::KERNEL_THREAD_COUNT>>>(values, index_map, destination);
        }
    }
    
    void BatchEncoder::decode(const Plaintext& plain, std::vector<uint64_t>& destination) const {
        if (this->matrix_reps_index_map.size() == 0) {
            throw std::logic_error("[BatchEncoder::encode] The parameters does not support vector batching.");
        }
        if (plain.is_ntt_form()) {
            throw std::invalid_argument("[BatchEncoder::decode] Plaintext is in NTT form.");
        }
        ContextDataPointer context_data = this->context()->first_context_data().value();
        size_t slots = this->slot_count();
        destination.resize(slots);
        size_t plain_coeff_count = std::min(plain.coeff_count(), slots);
        Array<uint64_t> temp_dest(slots, plain.on_device());
        temp_dest.slice(0, plain_coeff_count).copy_from_slice(plain.poly());
        // Transform destination using negacyclic NTT
        bool device = this->on_device();
        if (device != context_data->on_device()) {
            throw std::invalid_argument("[BatchEncoder::decode] Context and destination must reside on same device.");
        }
        utils::ntt_negacyclic_harvey(
            temp_dest.reference(),
            slots,
            context_data->plain_ntt_tables()
        );
        // Read in top row, then bottom row.
        if (!device) {
            decode_set_values(
                temp_dest.const_reference(),
                this->matrix_reps_index_map.const_reference(), 
                Slice<uint64_t>(destination.data(), destination.size(), false)
            );
        } else {
            Array<uint64_t> temp_dest_host(slots, true);
            decode_set_values(
                temp_dest.const_reference(),
                this->matrix_reps_index_map.const_reference(), 
                temp_dest_host.reference()
            );
            Slice<uint64_t>(destination.data(), destination.size(), false).copy_from_slice(temp_dest_host.const_reference());
        }
    }

    void BatchEncoder::decode_polynomial(const Plaintext& plaintext, std::vector<uint64_t>& destination) const {
        destination.resize(plaintext.data().size());
        Slice<uint64_t>(destination.data(), destination.size(), false)
            .copy_from_slice(plaintext.data().const_reference());
    }
}