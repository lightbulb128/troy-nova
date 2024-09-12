#include "bfv_ring2k.h"

#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <unordered_map>
#include "../batch_utils.h"

namespace troy::linear {

    using utils::Array;
    using utils::MultiplyUint64Operand;
    using utils::ConstSlice;
    using utils::Slice;
    using uint128_t = __uint128_t;
    using utils::KERNEL_THREAD_COUNT;
    using utils::ceil_div;

    static void custom_assert(bool condition, const char* message = "") {
        if (!condition) {
            throw std::invalid_argument(message);
        }
    }

    template <typename T>
    static constexpr typename std::enable_if<is_compatible_ring2k<T>::value, size_t>::type
    type_bits() {
        return sizeof(T) * 8;
    }

    template <typename T>
    static constexpr typename std::enable_if<is_compatible_ring2k<T>::value, T>::type
    inverse_ring2k(T x) {
        if ((x & 1) == 0) throw std::invalid_argument("[bfv_ring2k::inverse_ring2k] x must be odd");
        T inv = 1; T p = x;
        for (size_t i = 1; i < type_bits<T>(); i++) {
            inv *= p; p *= p;
        }
        return inv;
    }

    template <typename T>
    __host__ __device__ static constexpr typename std::enable_if<is_compatible_ring2k<T>::value, uint64_t>::type
    general_reduce(T x, const Modulus& modulus) {
        if constexpr (std::is_same<T, uint32_t>::value) {
            return modulus.reduce(static_cast<uint64_t>(x));
        } else if constexpr (std::is_same<T, uint64_t>::value) {
            return modulus.reduce(x);
        } else if constexpr (std::is_same<T, uint128_t>::value) {
            return modulus.reduce_uint128(x);
        }
    }

    static inline uint128_t assemble_from_limbs(ConstSlice<uint64_t> limbs) {
        if (limbs.size() == 0) return 0;
        if (limbs.size() == 1) return static_cast<uint128_t>(limbs[0]);
        return (
            (static_cast<uint128_t>(limbs[1]) << 64) | 
            (static_cast<uint128_t>(limbs[0]))
        );
    }

    __host__ __device__ static inline uint128_t uint128_from_uint64s(uint64_t low, uint64_t high) {
        return (static_cast<uint128_t>(high) << 64) | static_cast<uint128_t>(low);
    }

    __host__ __device__ static inline void set_uint64s_with_uint128(uint64_t* target, uint128_t source) {
        target[0] = static_cast<uint64_t>(source);
        target[1] = static_cast<uint64_t>(source >> 64);
    }

    template <typename T>
    static constexpr typename std::enable_if<is_compatible_ring2k<T>::value, T>::type
    modulo_mask(size_t bit_length) {
        if (bit_length > type_bits<T>()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::modulo_mask] bit_length must be less than or equal to type_bits<T>()");
        }
        if (bit_length == type_bits<T>()) {
            return static_cast<T>(-1);
        } else {
            return (static_cast<T>(1) << bit_length) - 1;
        }
    }

    template <typename T>
    static typename std::enable_if<is_compatible_ring2k<T>::value, T>::type
    modulo_from_limbs(ConstSlice<uint64_t> limbs, size_t mod_bit_length) {
        return static_cast<T>(assemble_from_limbs(limbs) & modulo_mask<T>(mod_bit_length));
    }

    template <typename T>
    PolynomialEncoderRNSHelper<T>::PolynomialEncoderRNSHelper(ContextDataPointer context_data, size_t t_bit_length) {
        if (t_bit_length <= type_bits<T>() / 2) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::PolynomialEncoderRNSHelper] t_bit_length must be greater than type_bits<T>() / 2");
        }
        EncryptionParameters parms = context_data->parms();
        if (parms.scheme() != SchemeType::BFV && parms.scheme() != SchemeType::BGV) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::PolynomialEncoderRNSHelper] scheme must be BFV or BGV");
        }
        if (context_data->parms().on_device()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::PolynomialEncoderRNSHelper] context_data must be on host. Please turn context to device after the encoder is created.");
        }
        
        this->parms_id_ = context_data->parms_id();
        this->t_bit_length_ = t_bit_length;
        size_t log_Q = context_data->total_coeff_modulus_bit_count();

        size_t poly_degree = parms.poly_modulus_degree();
        ConstSlice<Modulus> coeff_modulus = parms.coeff_modulus();
        size_t num_modulus = coeff_modulus.size();
        gamma_host_ = utils::get_prime(poly_degree, utils::HE_INTERNAL_MOD_BIT_COUNT);
        Modulus* gamma_space = reinterpret_cast<Modulus*>(std::malloc(sizeof(Modulus)));
        *gamma_space = gamma_host_;
        gamma_ = utils::Box<Modulus>(gamma_space, false, nullptr);
        for (size_t i = 0; i < coeff_modulus.size(); i++) {
            if (coeff_modulus[i].value() == gamma_host_.value()) {
                throw std::invalid_argument("[PolynomialEncoderRNSHelper::PolynomialEncoderRNSHelper] gamma is in coeff_modulus");
            }
        }
        mod_t_mask_ = modulo_mask<T>(t_bit_length);
        t_half_ = static_cast<T>(1) << (t_bit_length - 1);

        ConstSlice<uint64_t> Q = context_data->total_coeff_modulus();
        Array<uint64_t> Q_div_t(num_modulus, false, nullptr); Q_div_t.set_zero();
        if (log_Q > t_bit_length) {
            utils::right_shift_uint(Q, t_bit_length, num_modulus, Q_div_t.reference());
        } else {
            Q_div_t.set_zero();
        }

        Q_mod_t_ = modulo_from_limbs<T>(Q, t_bit_length);

        const utils::RNSTool& rns_tool = context_data->rns_tool();
        rns_tool.base_q().decompose_single(Q_div_t.reference());
        Q_div_t_mod_qi_ = Array<MultiplyUint64Operand>(num_modulus, false, nullptr);
        for (size_t i = 0; i < num_modulus; i++) {
            Q_div_t_mod_qi_[i] = MultiplyUint64Operand(Q_div_t[i], coeff_modulus[i]);
        }

        const utils::RNSBase& base_Q = rns_tool.base_q();
        utils::RNSBase base_gamma = utils::RNSBase(ConstSlice(&gamma_host_, 1, false, nullptr));
        base_Q_to_gamma_ = std::move(utils::BaseConverter(base_Q, base_gamma));

        punctured_q_mod_t_ = Array<T>(num_modulus, false, nullptr);
        for (size_t i = 0; i < num_modulus; i++) {
            punctured_q_mod_t_[i] = modulo_from_limbs<T>(base_Q.punctured_product().const_slice(i * num_modulus, (i + 1) * num_modulus), t_bit_length);
        }

        if (t_bit_length <= 64) {
            neg_inv_Q_mod_t_ = (-inverse_ring2k<T>(static_cast<T>(base_Q.base_product()[0]))) & mod_t_mask_;
            inv_gamma_mod_t_ = inverse_ring2k<T>(static_cast<T>(gamma_host_.value())) & mod_t_mask_;
        } else {
            T base_Q_128 = static_cast<T>(assemble_from_limbs(base_Q.base_product()));
            T base_gamma_128 = static_cast<T>(gamma_host_.value());
            neg_inv_Q_mod_t_ = (-inverse_ring2k<T>(base_Q_128)) & mod_t_mask_;
            inv_gamma_mod_t_ = inverse_ring2k<T>(base_gamma_128) & mod_t_mask_;
            if (((-(base_Q_128 * neg_inv_Q_mod_t_)) & mod_t_mask_) != 1) {
                throw std::invalid_argument("[PolynomialEncoderRNSHelper::PolynomialEncoderRNSHelper] -(base_Q * neg_inv_Q_mod_t) != 1");
            }
            if (((base_gamma_128 * inv_gamma_mod_t_) & mod_t_mask_) != 1) {
                throw std::invalid_argument("[PolynomialEncoderRNSHelper::PolynomialEncoderRNSHelper] base_gamma * inv_gamma_mod_t != 1");
            }
        };

        {
            uint64_t Q_mod_gamma = utils::modulo_uint(base_Q.base_product(), gamma_host_);
            uint64_t inv = 0;
            bool success = utils::try_invert_uint64_mod(Q_mod_gamma, gamma_host_, inv);
            if (!success) {
                throw std::invalid_argument("[PolynomialEncoderRNSHelper::PolynomialEncoderRNSHelper] failed to invert Q_mod_gamma");
            }
            neg_inv_Q_mod_gamma_ = utils::Box(
                reinterpret_cast<MultiplyUint64Operand*>(std::malloc(sizeof(MultiplyUint64Operand))),
                false, nullptr
            );
            *neg_inv_Q_mod_gamma_ = MultiplyUint64Operand(utils::negate_uint64_mod(inv, gamma_host_), gamma_host_);
        }

        gamma_t_mod_Q_ = Array<MultiplyUint64Operand>(num_modulus, false, nullptr);
        uint64_t t0[2]; set_uint64s_with_uint128(t0, static_cast<uint128_t>(1) << (t_bit_length / 2));
        uint64_t t1[2]; set_uint64s_with_uint128(t1, static_cast<uint128_t>(1) << (t_bit_length - t_bit_length / 2));

        for (size_t i = 0; i < num_modulus; i++) {
            const Modulus& prime = coeff_modulus[i];
            uint64_t t = prime.reduce_uint128_limbs(ConstSlice<uint64_t>(t0, 2, false, nullptr)); 
            t = utils::multiply_uint64_mod(t, prime.reduce_uint128_limbs(ConstSlice<uint64_t>(t1, 2, false, nullptr)), prime);
            uint64_t g = prime.reduce(gamma_host_.value());
            gamma_t_mod_Q_[i] = MultiplyUint64Operand(utils::multiply_uint64_mod(g, t, prime), prime);
        }
    }

    template <typename T>
    void PolynomialEncoderRNSHelper<T>::to_device_inplace(MemoryPoolHandle pool) {
        gamma_.to_device_inplace(pool);
        punctured_q_mod_t_.to_device_inplace(pool);
        gamma_t_mod_Q_.to_device_inplace(pool);
        base_Q_to_gamma_.to_device_inplace(pool);
        Q_div_t_mod_qi_.to_device_inplace(pool);
        neg_inv_Q_mod_gamma_.to_device_inplace(pool);
    }

    template <typename T>
    __device__ static void device_scale_up(
        ConstSlice<T> source,
        ConstSlice<Modulus> modulus, ConstSlice<MultiplyUint64Operand> Q_div_t_mod_qi,
        uint128_t Q_mod_t, uint128_t t_half, uint32_t base_mod_bitlen,
        Slice<uint64_t> out
    ) {
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j < source.size()) {
            T x = source[j];
            for (size_t i = 0; i < modulus.size(); i++) {
                // u = (Q mod t)*x mod qi
                uint64_t x64 = modulus[i].reduce(x);
                uint64_t u = utils::multiply_uint64operand_mod(x64, Q_div_t_mod_qi[i], modulus[i]);
                // uint128_t can conver uint32_t/uint64_t mult here
                T v = ((Q_mod_t * x + t_half) >> base_mod_bitlen);
                out[i * source.size() + j] = modulus[i].reduce(u + v);
            }
        }
    }

    template <typename T>
    __global__ static void kernel_scale_up(
        ConstSlice<T> source,
        ConstSlice<Modulus> modulus, ConstSlice<MultiplyUint64Operand> Q_div_t_mod_qi,
        uint128_t Q_mod_t, uint128_t t_half, uint32_t base_mod_bitlen,
        Slice<uint64_t> out
    ) {
        device_scale_up(source, modulus, Q_div_t_mod_qi, Q_mod_t, t_half, base_mod_bitlen, out);
    }

    template <typename T>
    __global__ static void kernel_scale_up_batched(
        utils::ConstSliceArrayRef<T> source,
        ConstSlice<Modulus> modulus, ConstSlice<MultiplyUint64Operand> Q_div_t_mod_qi,
        uint128_t Q_mod_t, uint128_t t_half, uint32_t base_mod_bitlen,
        utils::SliceArrayRef<uint64_t> out
    ) {
        size_t i = blockIdx.y;
        device_scale_up(source[i], modulus, Q_div_t_mod_qi, Q_mod_t, t_half, base_mod_bitlen, out[i]);
    }


    __device__ static void device_scale_up_uint128(
        ConstSlice<uint128_t> source,
        ConstSlice<Modulus> modulus, ConstSlice<MultiplyUint64Operand> Q_div_t_mod_qi,
        uint128_t Q_mod_t, uint128_t t_half, uint32_t base_mod_bitlen,
        Slice<uint64_t> out
    ) {
        size_t j = blockIdx.x * blockDim.x + threadIdx.x; // global_index
        if (j < source.size()) {
            uint128_t x = source[j];
            for (size_t i = 0; i < modulus.size(); i++) {
                uint64_t x64 = modulus[i].reduce_uint128(x);
                uint64_t u = troy::utils::multiply_uint64operand_mod(x64, Q_div_t_mod_qi[i], modulus[i]);
                
                // ensure 8-byte alignment
                uint64_t Q_mod_t_arr[2]; set_uint64s_with_uint128(Q_mod_t_arr, Q_mod_t);
                uint64_t t_half_arr[2]; set_uint64s_with_uint128(t_half_arr, t_half);
                uint64_t x_arr[2]; set_uint64s_with_uint128(x_arr, x);

                ConstSlice<uint64_t> Q_mod_t(Q_mod_t_arr, 2, true, nullptr);
                ConstSlice<uint64_t> xlimbs(x_arr, 2, true, nullptr);
                ConstSlice<uint64_t> t_half(t_half_arr, 2, true, nullptr);

                // Compute round(x * Q_mod_t / t) for 2^64 < x, t <= 2^128
                // round(x * Q_mod_t / t) = floor((x * Q_mod_t + t_half) / t)
                // We need 4 limbs to store the product x * Q_mod_t
                uint64_t mul_limbs[4]; Slice<uint64_t> mul_limbs_slice(mul_limbs, 4, true, nullptr);
                uint64_t add_limbs[4]; Slice<uint64_t> add_limbs_slice(add_limbs, 4, true, nullptr);
                uint64_t rs_limbs[3]; Slice<uint64_t> rs_limbs_slice(rs_limbs, 3, true, nullptr);
                utils::multiply_uint(Q_mod_t, xlimbs, mul_limbs_slice);
                utils::add_uint_carry(mul_limbs_slice.as_const(), t_half,
                        0, add_limbs_slice);
                // NOTE(juhou) base_mod_bitlen_ > 64, we can direct drop the LSB here.
                utils::right_shift_uint192(add_limbs_slice.const_slice(1, 4), base_mod_bitlen - 64,
                                    rs_limbs_slice);
                out[i * source.size() + j] = modulus[i].reduce_uint128(u + uint128_from_uint64s(rs_limbs[0], rs_limbs[1]));
            }
        }
    }

    __global__ static void kernel_scale_up_uint128(
        ConstSlice<uint128_t> source,
        ConstSlice<Modulus> modulus, ConstSlice<MultiplyUint64Operand> Q_div_t_mod_qi,
        uint128_t Q_mod_t, uint128_t t_half, uint32_t base_mod_bitlen,
        Slice<uint64_t> out
    ) {
        device_scale_up_uint128(source, modulus, Q_div_t_mod_qi, Q_mod_t, t_half, base_mod_bitlen, out);
    }

    __global__ static void kernel_scale_up_uint128_batched(
        utils::ConstSliceArrayRef<uint128_t> source,
        ConstSlice<Modulus> modulus, ConstSlice<MultiplyUint64Operand> Q_div_t_mod_qi,
        uint128_t Q_mod_t, uint128_t t_half, uint32_t base_mod_bitlen,
        utils::SliceArrayRef<uint64_t> out
    ) {
        size_t i = blockIdx.y;
        device_scale_up_uint128(source[i], modulus, Q_div_t_mod_qi, Q_mod_t, t_half, base_mod_bitlen, out[i]);
    }

    template<typename T>
    void PolynomialEncoderRNSHelper<T>::scale_up(utils::ConstSlice<T> source, const HeContext& context, Plaintext& destination, MemoryPoolHandle pool) const {
        static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value, "[PolynomialEncoderRNSHelper::scale_up_component] T must be uint32_t or uint64_t");
        // This implementation is only for uint32 and uint64.
        // Uint128 is implemented with a specialized version.
        if (!utils::device_compatible(source, *this)) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source and helper must be on the same device");
        }
        if (source.size() > context.key_context_data_pointer()->parms().poly_modulus_degree()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source size is larger than poly_modulus_degree");
        }
        destination = Plaintext();
        if (on_device()) destination.to_device_inplace(pool);
        destination.is_ntt_form() = false;
        destination.resize_rns_partial(context, parms_id_, source.size(), false, false);
        ContextDataPointer context_data = context.get_context_data(this->parms_id_).value();
        if (!source.on_device()) {
            for (size_t i = 0; i < destination.coeff_modulus_size(); i++) {
                Slice<uint64_t> destination_i = destination.component(i);
                custom_assert(source.on_device() == destination.on_device(), "[PolynomialEncoderRNSHelper::scale_up_component] source and destination are not in the same device");
                const EncryptionParameters& parms = context_data->parms();
                custom_assert(i < parms.coeff_modulus().size(), "[PolynomialEncoderRNSHelper::scale_up_component] modulus_index is out of range");
                const Modulus& modulus = parms.coeff_modulus()[i];
                for (size_t j = 0; j < source.size(); j++) {
                    uint64_t x64 = modulus.reduce(source[j]);
                    uint64_t u = utils::multiply_uint64operand_mod(x64, Q_div_t_mod_qi_[i], modulus);
                    // uint128_t can conver uint32_t/uint64_t mult here
                    uint64_t v = static_cast<uint64_t>(((static_cast<uint128_t>(Q_mod_t_) * static_cast<uint128_t>(source[j])) + static_cast<uint128_t>(t_half_)) >> t_bit_length_);
                    destination_i[j] = modulus.reduce(u + v);
                }
            }
        } else {
            size_t block_count = ceil_div(source.size(), KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            ConstSlice<Modulus> coeff_modulus = context_data->parms().coeff_modulus();
            custom_assert(coeff_modulus.size() == destination.coeff_modulus_size(), "[PolynomialEncoderRNSHelper::scale_up] coeff_modulus.size() != destination.coeff_modulus_size()");
            kernel_scale_up<T><<<block_count, KERNEL_THREAD_COUNT>>>(
                source, coeff_modulus, 
                Q_div_t_mod_qi_.const_reference(), Q_mod_t_, 
                t_half_, t_bit_length_, destination.reference()
            );
            utils::stream_sync();
        }
    }

    
    template<typename T>
    void PolynomialEncoderRNSHelper<T>::scale_up_batched(const utils::ConstSliceVec<T>& source, const HeContext& context, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value, "[PolynomialEncoderRNSHelper::scale_up_component] T must be uint32_t or uint64_t");
        if (source.size() != destination.size()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::scale_up_batched] source and destination must have the same size");
        }
        if (source.size() == 0) return;
        if (!context.on_device() || source.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < source.size(); i++) {
                scale_up(source[i], context, *destination[i], pool);
            }
        } else {
            size_t poly_degree = context.key_context_data_pointer()->parms().poly_modulus_degree();
            size_t n = source.size();
            ContextDataPointer context_data = context.get_context_data(this->parms_id_).value();
            ConstSlice<Modulus> coeff_modulus = context_data->parms().coeff_modulus();
            for (size_t i = 0; i < n; i++) {
                if (source[i].size() > poly_degree) {
                    throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source size is larger than poly_modulus_degree");
                }
                *destination[i] = Plaintext();
                if (on_device()) destination[i]->to_device_inplace(pool);
                destination[i]->is_ntt_form() = false;
                destination[i]->resize_rns_partial(context, parms_id_, source[i].size(), false, false);
                custom_assert(coeff_modulus.size() == destination[i]->coeff_modulus_size(), "[PolynomialEncoderRNSHelper::scale_up] coeff_modulus.size() != destination.coeff_modulus_size()");
            }

            size_t max_source_size = 0;
            for (size_t i = 0; i < n; i++) {
                max_source_size = std::max(max_source_size, source[i].size());
            }
            size_t block_count = ceil_div(max_source_size, KERNEL_THREAD_COUNT);
            utils::set_device(context.device_index());
            dim3 block_dims(block_count, n);
            auto source_batched = batch_utils::construct_batch(source, pool, coeff_modulus);
            auto destination_batched = batch_utils::construct_batch(batch_utils::pcollect_reference(destination), pool, coeff_modulus);
            kernel_scale_up_batched<T><<<block_dims, KERNEL_THREAD_COUNT>>>(
                source_batched, coeff_modulus, 
                Q_div_t_mod_qi_.const_reference(), Q_mod_t_, 
                t_half_, t_bit_length_, destination_batched
            );
            utils::stream_sync();
        }
    }

    template<>
    void PolynomialEncoderRNSHelper<uint128_t>::scale_up(utils::ConstSlice<uint128_t> source, const HeContext& context, Plaintext& destination, MemoryPoolHandle pool) const {
        // This implementation is only for uint32 and uint64.
        // Uint128 is implemented with a specialized version.
        if (!utils::device_compatible(source, *this)) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source and helper must be on the same device");
        }
        if (source.size() > context.key_context_data_pointer()->parms().poly_modulus_degree()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source size is larger than poly_modulus_degree");
        }
        if (on_device()) destination.to_device_inplace(pool);
        else destination.to_host_inplace();
        destination.is_ntt_form() = false;
        destination.resize_rns_partial(context, parms_id_, source.size(), false, false);
        ContextDataPointer context_data = context.get_context_data(this->parms_id_).value();
        if (!source.on_device()) {
            const EncryptionParameters& parms = context_data->parms();
            for (size_t i = 0; i < destination.coeff_modulus_size(); i++) {
                Slice<uint64_t> destination_i = destination.component(i);
                const Modulus& modulus = parms.coeff_modulus()[i];
                ConstSlice<uint64_t> t_half = ConstSlice<uint64_t>(reinterpret_cast<const uint64_t*>(&this->t_half_), 2, false, nullptr);
                ConstSlice<uint64_t> q_mod_t = ConstSlice<uint64_t>(reinterpret_cast<const uint64_t*>(&this->Q_mod_t_), 2, false, nullptr);
                for (size_t j = 0; j < source.size(); j++) {
                    uint128_t x = source[j];
                    uint64_t x64 = modulus.reduce_uint128(x);
                    uint64_t u = utils::multiply_uint64operand_mod(x64, Q_div_t_mod_qi_[i], modulus);
                    
                    uint64_t mul_limbs[4] = {0, 0, 0, 0}; Slice<uint64_t> mul_limbs_slice = Slice<uint64_t>(mul_limbs, 4, false, nullptr);
                    uint64_t add_limbs[4] = {0, 0, 0, 0}; Slice<uint64_t> add_limbs_slice = Slice<uint64_t>(add_limbs, 4, false, nullptr);
                    uint64_t rs_limbs[3] = {0, 0, 0}; Slice<uint64_t> rs_limbs_slice = Slice<uint64_t>(rs_limbs, 3, false, nullptr);
                    uint64_t x_limbs[2] = {static_cast<uint64_t>(x), static_cast<uint64_t>(x >> 64)}; ConstSlice<uint64_t> x_limbs_slice = ConstSlice<uint64_t>(x_limbs, 2, false, nullptr);
                    utils::multiply_uint(q_mod_t, x_limbs_slice, mul_limbs_slice);
                    utils::add_uint_carry(mul_limbs_slice.as_const(), t_half, 0, add_limbs_slice);
                    utils::right_shift_uint192(add_limbs_slice.const_slice(1, 4), t_bit_length_ - 64, rs_limbs_slice);
                    destination_i[j] = modulus.reduce_uint128(static_cast<uint128_t>(u) + assemble_from_limbs(rs_limbs_slice.as_const()));
                }
            }
        } else {
            size_t block_count = ceil_div(source.size(), KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            ConstSlice<Modulus> coeff_modulus = context_data->parms().coeff_modulus();
            custom_assert(coeff_modulus.size() == destination.coeff_modulus_size(), "[PolynomialEncoderRNSHelper::scale_up] coeff_modulus.size() != destination.coeff_modulus_size()");
            kernel_scale_up_uint128<<<block_count, KERNEL_THREAD_COUNT>>>(
                source, coeff_modulus, 
                Q_div_t_mod_qi_.const_reference(), Q_mod_t_, 
                t_half_, t_bit_length_, destination.reference()
            );
            utils::stream_sync();
        }
    }

    template<>
    void PolynomialEncoderRNSHelper<uint128_t>::scale_up_batched(const utils::ConstSliceVec<uint128_t>& source, const HeContext& context, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (source.size() != destination.size()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::scale_up_batched] source and destination must have the same size");
        }
        if (source.size() == 0) return;
        if (!context.on_device() || source.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < source.size(); i++) {
                scale_up(source[i], context, *destination[i], pool);
            }
        } else {
            size_t poly_degree = context.key_context_data_pointer()->parms().poly_modulus_degree();
            size_t n = source.size();
            ContextDataPointer context_data = context.get_context_data(this->parms_id_).value();
            ConstSlice<Modulus> coeff_modulus = context_data->parms().coeff_modulus();
            for (size_t i = 0; i < n; i++) {
                if (source[i].size() > poly_degree) {
                    throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source size is larger than poly_modulus_degree");
                }
                *destination[i] = Plaintext();
                if (on_device()) destination[i]->to_device_inplace(pool);
                destination[i]->is_ntt_form() = false;
                destination[i]->resize_rns_partial(context, parms_id_, source[i].size(), false, false);
                custom_assert(coeff_modulus.size() == destination[i]->coeff_modulus_size(), "[PolynomialEncoderRNSHelper::scale_up] coeff_modulus.size() != destination.coeff_modulus_size()");
            }

            size_t max_source_size = 0;
            for (size_t i = 0; i < n; i++) {
                max_source_size = std::max(max_source_size, source[i].size());
            }
            size_t block_count = ceil_div(max_source_size, KERNEL_THREAD_COUNT);
            utils::set_device(context.device_index());
            dim3 block_dims(block_count, n);
            auto source_batched = batch_utils::construct_batch(source, pool, coeff_modulus);
            auto destination_batched = batch_utils::construct_batch(batch_utils::pcollect_reference(destination), pool, coeff_modulus);
            kernel_scale_up_uint128_batched<<<block_dims, KERNEL_THREAD_COUNT>>>(
                source_batched, coeff_modulus, 
                Q_div_t_mod_qi_.const_reference(), Q_mod_t_, 
                t_half_, t_bit_length_, destination_batched
            );
            utils::stream_sync();
        }
    }


    template <typename T>
    __device__ static void device_centralize(
        ConstSlice<T> source,
        utils::ConstSlice<troy::Modulus> mod_qs,
        uint128_t t_half, uint128_t mod_t_mask,
        Slice<uint64_t> out
    ) {
        size_t j = blockIdx.x * blockDim.x + threadIdx.x; // global_index
        if (j < source.size()) {
            T x = source[j];
            for (size_t i = 0; i < mod_qs.size(); i++) {
                const Modulus* mod_qi = &mod_qs[i];
                auto x128 = static_cast<uint128_t>(x);
                if (x128 > t_half) {
                    uint64_t u = general_reduce(-x128 & mod_t_mask, *mod_qi);
                    out[i * source.size() + j] = troy::utils::negate_uint64_mod(u, *mod_qi);
                } else {
                    out[i * source.size() + j] = general_reduce(x, *mod_qi);
                }
            }
        }
    }

    template <typename T>
    __global__ static void kernel_centralize(
        ConstSlice<T> source,
        utils::ConstSlice<troy::Modulus> mod_qs,
        uint128_t t_half, uint128_t mod_t_mask,
        Slice<uint64_t> out
    ) {
        device_centralize(source, mod_qs, t_half, mod_t_mask, out);
    }

    template <typename T>
    __global__ static void kernel_centralize_batched(
        utils::ConstSliceArrayRef<T> source,
        utils::ConstSlice<troy::Modulus> mod_qs,
        uint128_t t_half, uint128_t mod_t_mask,
        utils::SliceArrayRef<uint64_t> out
    ) {
        size_t i = blockIdx.y;
        device_centralize(source[i], mod_qs, t_half, mod_t_mask, out[i]);
    }

    template <typename T>
    void PolynomialEncoderRNSHelper<T>::centralize(utils::ConstSlice<T> source, const HeContext& context, Plaintext& destination, MemoryPoolHandle pool) const {
        if (!utils::device_compatible(source, *this)) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source and helper must be on the same device");
        }
        if (source.size() > context.key_context_data_pointer()->parms().poly_modulus_degree()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper:scale_up] source size is larger than poly_modulus_degree");
        }
        destination = Plaintext();
        if (on_device()) destination.to_device_inplace(pool);
        destination.is_ntt_form() = false;
        destination.resize_rns_partial(context, parms_id_, source.size(), false, false);
        if (!source.on_device()) {
            const EncryptionParameters& parms = context.get_context_data(this->parms_id_).value()->parms();
            for (size_t i = 0; i < destination.coeff_modulus_size(); i++) {
                Slice<uint64_t> destination_i = destination.component(i);
                const Modulus& modulus = parms.coeff_modulus()[i];
                for (size_t j = 0; j < source.size(); j++) {
                    T x = source[j];
                    if (x > t_half_) {
                        uint64_t u = general_reduce((-x) & mod_t_mask_, modulus);
                        destination_i[j] = utils::negate_uint64_mod(u, modulus);
                    } else {
                        destination_i[j] = general_reduce(x, modulus);
                    }
                }
            }
        } else {
            size_t block_count = ceil_div(source.size(), KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            ConstSlice<Modulus> coeff_modulus = context.get_context_data(this->parms_id_).value()->parms().coeff_modulus();
            custom_assert(coeff_modulus.size() == destination.coeff_modulus_size(), "[PolynomialEncoderRNSHelper::centralize] coeff_modulus.size() != destination.coeff_modulus_size()");
            kernel_centralize<<<block_count, KERNEL_THREAD_COUNT>>>(
                source, coeff_modulus, t_half_, mod_t_mask_, destination.reference()
            );
            utils::stream_sync();
        }
    }


    template<typename T>
    void PolynomialEncoderRNSHelper<T>::centralize_batched(const utils::ConstSliceVec<T>& source, const HeContext& context, const std::vector<Plaintext*>& destination, MemoryPoolHandle pool) const {
        if (source.size() != destination.size()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::centralize_batched] source and destination must have the same size");
        }
        if (source.size() == 0) return;
        if (!context.on_device() || source.size() < utils::BATCH_OP_THRESHOLD) {
            for (size_t i = 0; i < source.size(); i++) {
                centralize(source[i], context, *destination[i], pool);
            }
        } else {
            size_t poly_degree = context.key_context_data_pointer()->parms().poly_modulus_degree();
            size_t n = source.size();
            ContextDataPointer context_data = context.get_context_data(this->parms_id_).value();
            ConstSlice<Modulus> coeff_modulus = context_data->parms().coeff_modulus();
            for (size_t i = 0; i < n; i++) {
                if (source[i].size() > poly_degree) {
                    throw std::invalid_argument("[PolynomialEncoderRNSHelper:centralize_batched] source size is larger than poly_modulus_degree");
                }
                *destination[i] = Plaintext();
                if (on_device()) destination[i]->to_device_inplace(pool);
                destination[i]->is_ntt_form() = false;
                destination[i]->resize_rns_partial(context, parms_id_, source[i].size(), false, false);
                custom_assert(coeff_modulus.size() == destination[i]->coeff_modulus_size(), "[PolynomialEncoderRNSHelper::scale_up] coeff_modulus.size() != destination.coeff_modulus_size()");
            }

            size_t max_source_size = 0;
            for (size_t i = 0; i < n; i++) {
                max_source_size = std::max(max_source_size, source[i].size());
            }
            size_t block_count = ceil_div(max_source_size, KERNEL_THREAD_COUNT);
            utils::set_device(context.device_index());
            dim3 block_dims(block_count, n);
            auto source_batched = batch_utils::construct_batch(source, pool, coeff_modulus);
            auto destination_batched = batch_utils::construct_batch(batch_utils::pcollect_reference(destination), pool, coeff_modulus);
            kernel_centralize_batched<T><<<block_dims, KERNEL_THREAD_COUNT>>>(
                source_batched, coeff_modulus, t_half_, mod_t_mask_, destination_batched
            );
            utils::stream_sync();
        }
    }


    template <typename T>
    __global__ static void kernel_scale_down(
        size_t num_modulus, size_t coeff_count,
        ConstSlice<uint64_t> tmp,
        ConstSlice<T> punctured_base_mod_t,
        uint128_t neg_inv_Q_mod_t,
        uint128_t inv_gamma_mod_t,
        uint128_t mod_t_mask,
        ConstSlice<uint64_t> base_on_gamma,
        utils::ConstPointer<troy::Modulus> gamma,
        Slice<T> out
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x; // global_index
        if (i >= coeff_count) return;

        T base_on_t = 0;
        // sum_i (x * (Q/qi)^{-1} mod qi) * (Q/qi) mod t
        for (size_t l = 0; l < num_modulus; ++l) {
            const T factor = punctured_base_mod_t[l];
            base_on_t += tmp[l * coeff_count + i] * factor;
        }

        // 3-2 Then multiply with -Q^{-1} mod t
        base_on_t = (base_on_t * neg_inv_Q_mod_t) & mod_t_mask;
        
        // clang-format off
        // 4 Correct sign: (base_on_t - [base_on_gamma]_gamma) * gamma^{-1} mod t
        // NOTE(juhou):
        // `base_on_gamma` and `base_on_t` together gives
        // `gamma*(x + t*r) + round(gamma*v/q) - e` mod gamma*t for some unknown v and e.
        // (Key point): Taking `base_on_gamma` along equals to
        //    `round(gamma*v/q) - e mod gamma`
        // When gamma > v, e, we can have the centered remainder
        // [round(gamma*v/q) - e mod gamma]_gamma = round(gamma*v/q) - e.
        // As a result, `base_on_t - [base_on_gamma]_gamma mod t` will cancel out the
        // last term and gives `gamma*(x + t*r) mod t`.
        // Finally, multiply with `gamma^{-1} mod t` gives `x mod t`.
        // clang-format on
        uint64_t gamma_div_2 = gamma->value() >> 1;
        uint64_t on_gamma = base_on_gamma[i];
        // [0, gamma) -> [-gamma/2, gamma/2]
        if (on_gamma > gamma_div_2) {
            out[i] = ((base_on_t + gamma->value() - on_gamma) * inv_gamma_mod_t) & mod_t_mask;
        } else {
            out[i] = ((base_on_t - on_gamma) * inv_gamma_mod_t) & mod_t_mask;
        }
    }


    template <typename T>
    void PolynomialEncoderRNSHelper<T>::scale_down(const Plaintext& input, const HeContext& context, utils::Slice<T> destination, MemoryPoolHandle pool) const {
        // Ref: Bajard et al. "A Full RNS Variant of FV like Somewhat Homomorphic
        // Encryption Schemes" (Section 3.2 & 3.3)
        // NOTE(juhou): Basically the same code in seal/util/rns.cpp instead we
        // use the plain modulus `t` as 2^k here.
        ParmsID parms_id = input.parms_id();
        if (parms_id == parms_id_zero) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::scale_down] input is not valid");
        }
        if (input.on_device() != this->on_device() || input.on_device() != destination.on_device()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::scale_down] self, input, destination are not in the same device");
        }
        ContextDataPointer context_data = context.get_context_data(parms_id).value();
        const EncryptionParameters& parms = context_data->parms();
        size_t num_modulus = parms.coeff_modulus().size();
        size_t coeff_count = destination.size();
        custom_assert(input.coeff_modulus_size() == num_modulus);
        custom_assert(input.coeff_count() == coeff_count);
        custom_assert(input.data().size() == num_modulus * coeff_count);
        const utils::RNSBase &base_Q = context_data->rns_tool().base_q();
        ConstSlice<troy::Modulus> coeff_modulus = parms.coeff_modulus();
        custom_assert(input.on_device() == destination.on_device(), "[PolynomialEncoderRNSHelper::scale_down] input and destination are not in the same device");
        bool device = input.on_device();

        Array<uint64_t> tmp = Array<uint64_t>::create_uninitialized(input.data().size(), device, pool);

        // TODO: the following steps could use a kernel fusing.

        // 1. multiply with gamma*t
        troy::utils::multiply_uint64operand_p(
            input.const_reference(), gamma_t_mod_Q_.const_reference(), 
            coeff_count, coeff_modulus, tmp.reference()
        );

        // 2-1 FastBase convert from baseQ to {gamma}
        Array<uint64_t> base_on_gamma = Array<uint64_t>::create_uninitialized(coeff_count, device, pool);
        base_Q_to_gamma_.fast_convert_array(tmp.const_reference(), base_on_gamma.reference(), pool);
        // 2-2 Then multiply with -Q^{-1} mod gamma
        troy::utils::multiply_uint64operand_inplace(
            base_on_gamma.reference(), neg_inv_Q_mod_gamma_.as_const_pointer(),
            gamma_.as_const_pointer()
        );

        // 3-1 FastBase convert from baseQ to {t}
        // NOTE: overwrite the `tmp` (tmp is gamma*t*x mod Q)
        ConstSlice<troy::utils::MultiplyUint64Operand> inv_punctured = base_Q.inv_punctured_product_mod_base();
        utils::multiply_uint64operand_inplace_p(
            tmp.reference(), inv_punctured, coeff_count, coeff_modulus
        );

        if (!device) {
            for (size_t i = 0; i < coeff_count; i++) {
                T base_on_t = 0;
                // sum_i (x * (Q/qi)^{-1} mod qi) * (Q/qi) mod t
                for (size_t l = 0; l < num_modulus; ++l) {
                    const T factor = punctured_q_mod_t_[l];
                    base_on_t += tmp[l * coeff_count + i] * factor;
                }

                // 3-2 Then multiply with -Q^{-1} mod t
                base_on_t = (base_on_t * neg_inv_Q_mod_t_) & mod_t_mask_;

                uint64_t gamma_div_2 = gamma_->value() >> 1;
                uint64_t on_gamma = base_on_gamma[i];
                // [0, gamma) -> [-gamma/2, gamma/2]
                if (on_gamma > gamma_div_2) {
                    destination[i] = ((base_on_t + gamma_->value() - on_gamma) * inv_gamma_mod_t_) & mod_t_mask_;
                } else {
                    destination[i] = ((base_on_t - on_gamma) * inv_gamma_mod_t_) & mod_t_mask_;
                }
            }
        } else {
            size_t block_count = ceil_div(coeff_count, KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_scale_down<T><<<block_count, KERNEL_THREAD_COUNT>>>(
                num_modulus, coeff_count, tmp.const_reference(),
                punctured_q_mod_t_.const_reference(),
                neg_inv_Q_mod_t_, inv_gamma_mod_t_, mod_t_mask_,
                base_on_gamma.const_reference(), gamma_.as_const_pointer(),
                destination
            );
            utils::stream_sync();
        }
    }

    void host_decentralize_step1(const utils::RNSBase& ibase, ConstSlice<uint64_t> input, Slice<uint64_t> temp, Slice<double> v) {
        size_t count = input.size() / ibase.size();
        for (size_t i = 0; i < ibase.size(); i++) {
            const MultiplyUint64Operand& op = ibase.inv_punctured_product_mod_base()[i];
            const Modulus& base = ibase.base()[i];
            double divisor = static_cast<double>(base.value());
            if (op.operand == 1) {
                for (size_t j = 0; j < count; j++) {
                    temp[j * ibase.size() + i] = utils::barrett_reduce_uint64(input[i * count + j], base);
                    double dividend = static_cast<double>(temp[j * ibase.size() + i]);
                    v[j * ibase.size() + i] = dividend / divisor;
                }
            } else {
                for (size_t j = 0; j < count; j++) {
                    temp[j * ibase.size() + i] = utils::multiply_uint64operand_mod(input[i * count + j], op, base);
                    double dividend = static_cast<double>(temp[j * ibase.size() + i]);
                    v[j * ibase.size() + i] = dividend / divisor;
                }
            }
        }
    }

    __global__ void kernel_decentralize_step1(
        ConstSlice<Modulus> ibase, 
        ConstSlice<MultiplyUint64Operand> ibase_inv_punctured_product_mod_base,
        ConstSlice<uint64_t> input, Slice<uint64_t> temp, Slice<double> v
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= input.size()) {
            return;
        }
        size_t ibase_size = ibase.size();
        size_t count = input.size() / ibase_size;
        size_t i = global_index / count;
        size_t j = global_index % count;
        const MultiplyUint64Operand& op = ibase_inv_punctured_product_mod_base[i];
        const Modulus& base = ibase[i];
        if (op.operand == 1) {
            temp[j * ibase_size + i] = utils::barrett_reduce_uint64(input[global_index], base);
        } else {
            temp[j * ibase_size + i] = utils::multiply_uint64operand_mod(input[global_index], op, base);
        }
        double dividend = static_cast<double>(temp[j * ibase_size + i]);
        v[j * ibase_size + i] = dividend / static_cast<double>(base.value());
    }

    template <typename T>
    __host__ __device__
    inline T general_dot_product_mod(ConstSlice<uint64_t> operand1, ConstSlice<T> operand2, T t_mask) {
        T accumulated = 0;
        for (size_t i = 0; i < operand1.size(); i++) {
            accumulated += static_cast<T>(operand1[i]) * operand2[i];
        }
        return accumulated & t_mask;
    }

    template <typename T>
    void host_decentralize_step2(const utils::RNSBase& base_q, ConstSlice<T> base_change_matrix, T q_mod_t, T t_mask, ConstSlice<uint64_t> temp, ConstSlice<double> v, Slice<T> output) {
        size_t base_q_size = base_q.size();
        size_t count = temp.size() / base_q_size;
        for (size_t j = 0; j < count; j++) {
            double aggregated_v = 0;
            for (size_t i = j*base_q_size; i < (j+1)*base_q_size; i++) {
                aggregated_v += v[i];
            }
            T aggregated_rounded_v = std::round(aggregated_v);
            T sum_mod_obase = general_dot_product_mod(
                temp.const_slice(j * base_q_size, (j + 1) * base_q_size),
                base_change_matrix, // because output has only one modulus, the row is the matrix itself.
                t_mask
            );
            T v_q_mod_p = (aggregated_rounded_v * q_mod_t) & t_mask;
            output[j] = (sum_mod_obase - v_q_mod_p) & t_mask;
        }
    }

    template <typename T>
    __global__ void kernel_decentralize_step2(
        size_t base_q_size,
        ConstSlice<T> base_change_matrix,
        T q_mod_t, T t_mask, ConstSlice<uint64_t> temp, ConstSlice<double> v, Slice<T> output
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= output.size()) {
            return;
        }
        size_t j = global_index;
        double aggregated_v = 0;
        for (size_t i = j*base_q_size; i < (j+1)*base_q_size; i++) {
            aggregated_v += v[i];
        }
        T aggregated_rounded_v = std::round(aggregated_v);
        T sum_mod_obase = general_dot_product_mod(
            temp.const_slice(j * base_q_size, (j + 1) * base_q_size),
            base_change_matrix, // because output has only one modulus, the row is the matrix itself.
            t_mask
        );
        T v_q_mod_p = (aggregated_rounded_v * q_mod_t) & t_mask;
        output[j] = (sum_mod_obase - v_q_mod_p) & t_mask;
    }

    template <typename T>
    void host_multiply_scalar_mask_inplace(Slice<T> target, T scalar, T mask) {
        for (size_t i = 0; i < target.size(); i++) {
            target[i] = (target[i] * scalar) & mask;
        }
    }

    template <typename T>
    __global__ static void kernel_multiply_scalar_mask_inplace(Slice<T> target, T scalar, T mask) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_index >= target.size()) {
            return;
        }
        target[global_index] = (target[global_index] * scalar) & mask;
    }



    template <typename T>
    void PolynomialEncoderRNSHelper<T>::decentralize(const Plaintext& input, const HeContext& context, utils::Slice<T> destination, T correction_factor, MemoryPoolHandle pool) const {
        // Ref: Bajard et al. "A Full RNS Variant of FV like Somewhat Homomorphic
        // Encryption Schemes" (Section 3.2 & 3.3)
        // NOTE(juhou): Basically the same code in seal/util/rns.cpp instead we
        // use the plain modulus `t` as 2^k here.
        ParmsID parms_id = input.parms_id();
        if (parms_id == parms_id_zero) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::scale_down] input is not valid");
        }
        if (input.on_device() != this->on_device() || input.on_device() != destination.on_device()) {
            throw std::invalid_argument("[PolynomialEncoderRNSHelper::scale_down] self, input, destination are not in the same device");
        }
        ContextDataPointer context_data = context.get_context_data(parms_id).value();
        const EncryptionParameters& parms = context_data->parms();
        size_t num_modulus = parms.coeff_modulus().size();
        size_t coeff_count = destination.size();
        custom_assert(input.coeff_modulus_size() == num_modulus);
        custom_assert(input.coeff_count() == coeff_count);
        custom_assert(input.data().size() == num_modulus * coeff_count);
        const utils::RNSBase &base_Q = context_data->rns_tool().base_q();
        ConstSlice<troy::Modulus> coeff_modulus = parms.coeff_modulus();
        custom_assert(input.on_device() == destination.on_device(), "[PolynomialEncoderRNSHelper::scale_down] input and destination are not in the same device");
        bool device = input.on_device();

        Array<uint64_t> temp(coeff_count * num_modulus, device, pool);
        Array<double> v(coeff_count * num_modulus, device, pool);

        if (!device) {
            host_decentralize_step1(base_Q, input.const_reference(), temp.reference(), v.reference());
            host_decentralize_step2(base_Q, this->punctured_q_mod_t(), Q_mod_t_, mod_t_mask_, temp.const_reference(), v.const_reference(), destination);
        } else {
            size_t block_count = ceil_div(input.data().size(), KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_decentralize_step1<<<block_count, KERNEL_THREAD_COUNT>>>(
                coeff_modulus, base_Q.inv_punctured_product_mod_base(),
                input.const_reference(), temp.reference(), v.reference()
            );
            block_count = ceil_div(coeff_count, KERNEL_THREAD_COUNT);
            kernel_decentralize_step2<<<block_count, KERNEL_THREAD_COUNT>>>(
                base_Q.size(), this->punctured_q_mod_t(),
                Q_mod_t_, mod_t_mask_, temp.const_reference(), v.const_reference(), destination
            );
        }

        if (correction_factor != 1) {
            T fix = inverse_ring2k(correction_factor) & mod_t_mask_;
            if (!device) {
                host_multiply_scalar_mask_inplace(destination, fix, mod_t_mask_);
            } else {
                size_t block_count = ceil_div(destination.size(), KERNEL_THREAD_COUNT);
                utils::set_device(destination.device_index());
                kernel_multiply_scalar_mask_inplace<<<block_count, KERNEL_THREAD_COUNT>>>(destination, fix, mod_t_mask_);
            }
        }
    }

    template <typename T>
    PolynomialEncoderRing2k<T>::PolynomialEncoderRing2k(HeContextPointer context, size_t t_bit_length) {
        context_ = context;
        t_bit_length_ = t_bit_length;
        std::optional<ContextDataPointer> context_data = context->key_context_data();
        std::unordered_map<ParmsID, std::shared_ptr<PolynomialEncoderRNSHelper<T>>, std::TroyHashParmsID> helpers;
        while (context_data.has_value()) {
            ContextDataPointer c = context_data.value();
            ParmsID parms_id = c->parms_id();
            helpers[parms_id] = std::make_shared<PolynomialEncoderRNSHelper<T>>(c, t_bit_length);
            context_data = c->next_context_data();
        }
        helpers_ = helpers;
    }
    
    // Instantiate the template class implementations
    
    template class PolynomialEncoderRNSHelper<uint32_t>;
    template class PolynomialEncoderRNSHelper<uint64_t>;
    template class PolynomialEncoderRNSHelper<uint128_t>;

    template class PolynomialEncoderRing2k<uint32_t>;
    template class PolynomialEncoderRing2k<uint64_t>;
    template class PolynomialEncoderRing2k<uint128_t>;
    

}