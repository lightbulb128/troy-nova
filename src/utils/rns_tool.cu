#include "rns_tool.h"
#include "uint_small_mod.h"
#include "polynomial_buffer.h"

#include "../fgk/rns_tool.h"

namespace troy {namespace utils {

    static void print_array(ConstSlice<uint64_t> array, bool end_line = true) {
        if (array.on_device()) {
            Array<uint64_t> host = Array<uint64_t>::create_and_copy_from_slice(array, false, nullptr);
            print_array(host.const_reference(), end_line);
            return;
        }
        std::cout << "[";
        for (size_t i = 0; i < array.size(); i++) {
            std::cout << array[i];
            if (i != array.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (end_line) {
            std::cout << std::endl;
        }
    }
    
    RNSTool::RNSTool(size_t poly_modulus_degree, const RNSBase& q, const Modulus& t) {

        if (q.on_device()) {
            throw std::invalid_argument("[RNSTool::RNSTool] Cannot create RNSTool from device RNSBase q.");
        }

        if (q.size() < utils::HE_COEFF_MOD_COUNT_MIN || q.size() > utils::HE_COEFF_MOD_COUNT_MAX) {
            throw std::invalid_argument("[RNSTool::RNSTool] RNSBase length is invalid.");
        }

        int coeff_count_power = utils::get_power_of_two(poly_modulus_degree);

        if (
            coeff_count_power < 0 || 
            poly_modulus_degree > utils::HE_POLY_MOD_DEGREE_MAX ||
            poly_modulus_degree < utils::HE_POLY_MOD_DEGREE_MIN
        ) {
            throw std::invalid_argument("[RNSTool::RNSTool] Polynomial modulus degree is invalid.");
        }
        
        // Allocate memory for the bases q, B, Bsk, Bsk U m_tilde, t_gamma
        size_t base_q_size = q.size();

        // In some cases we might need to increase the size of the base B by one, namely we require
        // K * n * t * q^2 < q * prod(B) * m_sk, where K takes into account cross terms when larger size ciphertexts
        // are used, and n is the "delta factor" for the ring. We reserve 32 bits for K * n. Here the coeff modulus
        // primes q_i are bounded to be HE_USER_MOD_BIT_COUNT_MAX (60) bits, and all primes in B and m_sk are
        // HE_INTERNAL_MOD_BIT_COUNT (61) bits.
        size_t total_coeff_bit_count = utils::get_significant_bit_count_uint(q.base_product());

        size_t base_B_size = base_q_size;
        if (32 + t.bit_count() + total_coeff_bit_count >= utils::HE_INTERNAL_MOD_BIT_COUNT * base_q_size + utils::HE_INTERNAL_MOD_BIT_COUNT) {
            base_B_size++;
        }

        size_t base_Bsk_size = base_B_size + 1;
        size_t base_Bsk_m_tilde_size = base_Bsk_size + 1;

        // Sample primes for B and two more primes: m_sk and gamma
        size_t coeff_count = poly_modulus_degree;
        
        std::vector<Modulus> baseconv_primes = utils::get_primes(
            static_cast<uint64_t>(2 * coeff_count),
            utils::HE_INTERNAL_MOD_BIT_COUNT,
            base_Bsk_m_tilde_size
        );
        Modulus m_sk = baseconv_primes[0];
        Modulus gamma = baseconv_primes[1];
        std::vector<Modulus> base_B_primes; base_B_primes.reserve(baseconv_primes.size() - 2);
        for (size_t i = 2; i < baseconv_primes.size(); i++) {
            base_B_primes.push_back(baseconv_primes[i]);
        }

        // Set m_tilde to a non-prime value
        Modulus m_tilde(1ul << 32);
        uint64_t m_tilde_value = m_tilde.value();
        
        // Populate the base arrays
        RNSBase base_q = q.clone(nullptr);
        RNSBase base_B(ConstSlice(base_B_primes.data(), base_B_primes.size(), false, nullptr));
        RNSBase base_Bsk = base_B.extend_modulus(m_sk);
        RNSBase base_Bsk_m_tilde = base_Bsk.extend_modulus(m_tilde);

        // Set up t-gamma base if t_ is non-zero (using BFV)
        std::optional<RNSBase> base_t_gamma = std::nullopt;
        std::optional<BaseConverter> base_q_to_t_conv = std::nullopt;
        if (!t.is_zero()) {
            Modulus t_gamma[2]{ t, gamma };
            base_t_gamma = std::optional(RNSBase(ConstSlice(t_gamma, 2, false, nullptr)));
            base_q_to_t_conv = std::optional(BaseConverter(base_q, RNSBase(ConstSlice(&t, 1, false, nullptr))));
        }
        
        // Generate the Bsk NTTTables; these are used for NTT after base extension to Bsk
        Array<NTTTables> base_Bsk_ntt_tables = NTTTables::create_ntt_tables(
            coeff_count_power,
            base_Bsk.base()
        );

        BaseConverter base_q_to_Bsk_conv = BaseConverter(base_q, base_Bsk);
        BaseConverter base_q_to_m_tilde_conv = BaseConverter(base_q, RNSBase(ConstSlice(&m_tilde, 1, false, nullptr)));
        BaseConverter base_B_to_q_conv = BaseConverter(base_B, base_q);
        BaseConverter base_B_to_m_sk_conv = BaseConverter(base_B, RNSBase(ConstSlice(&m_sk, 1, false, nullptr)));
        std::optional<BaseConverter> base_q_to_t_gamma_conv = std::nullopt;
        if (base_q_to_t_conv.has_value()) {
            base_q_to_t_gamma_conv = std::optional(BaseConverter(base_q, base_t_gamma.value()));
        }

        // Compute prod(B) mod q
        Array<uint64_t> prod_B_mod_q(base_q.size(), false, nullptr);
        for (size_t i = 0; i < base_q.size(); i++) {
            prod_B_mod_q[i] = utils::modulo_uint(base_B.base_product(), base_q.base()[i]);
        }

        // Compute prod(q)^(-1) mod Bsk
        Array<MultiplyUint64Operand> inv_prod_q_mod_Bsk(base_Bsk.size(), false, nullptr);
        for (size_t i = 0; i < base_Bsk.size(); i++) {
            const Modulus& modulus = base_Bsk.base()[i];
            uint64_t temp = utils::modulo_uint(base_q.base_product(), modulus);
            bool try_invert = utils::try_invert_uint64_mod(temp, modulus, temp);
            if (!try_invert) {
                throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert base_q product.");
            }
            inv_prod_q_mod_Bsk[i] = MultiplyUint64Operand(temp, modulus);
        }

        // Compute prod(B)^(-1) mod m_sk
        uint64_t temp = utils::modulo_uint(base_B.base_product(), m_sk);
        bool try_invert = utils::try_invert_uint64_mod(temp, m_sk, temp);
        if (!try_invert) {
            throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert base_B product.");
        }
        MultiplyUint64Operand inv_prod_B_mod_m_sk(temp, m_sk);

        // Compute m_tilde^(-1) mod Bsk
        Array<MultiplyUint64Operand> inv_m_tilde_mod_Bsk(base_Bsk.size(), false, nullptr);
        for (size_t i = 0; i < base_Bsk.size(); i++) {
            const Modulus& modulus = base_Bsk.base()[i];
            try_invert = utils::try_invert_uint64_mod(modulus.reduce(m_tilde.value()), modulus, temp);
            if (!try_invert) {
                throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert m_tilde.");
            }
            inv_m_tilde_mod_Bsk[i] = MultiplyUint64Operand(temp, modulus);
        }
        
        // Compute prod(q)^(-1) mod m_tilde
        temp = utils::modulo_uint(base_q.base_product(), m_tilde);
        try_invert = utils::try_invert_uint64_mod(temp, m_tilde, temp);
        if (!try_invert) {
            throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert base_q product.");
        }
        MultiplyUint64Operand neg_inv_prod_q_mod_m_tilde(
            utils::negate_uint64_mod(temp, m_tilde), m_tilde
        );

        Array<uint64_t> prod_q_mod_Bsk(base_Bsk.size(), false, nullptr);
        for (size_t i = 0; i < base_Bsk.size(); i++) {
            prod_q_mod_Bsk[i] = utils::modulo_uint(base_q.base_product(), base_Bsk.base()[i]);
        }

        std::optional<MultiplyUint64Operand> inv_gamma_mod_t = std::nullopt;
        std::optional<Array<MultiplyUint64Operand>> prod_t_gamma_mod_q = std::nullopt;
        std::optional<Array<MultiplyUint64Operand>> neg_inv_q_mod_t_gamma = std::nullopt;
        uint64_t inv_q_last_mod_t =1;
        uint64_t q_last_mod_t = 1;
        if (base_t_gamma.has_value()) {

            // Compute gamma^(-1) mod t
            try_invert = utils::try_invert_uint64_mod(t.reduce(gamma.value()), t, temp);
            if (!try_invert) {
                throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert gamma mod t.");
            }
            inv_gamma_mod_t = std::optional(MultiplyUint64Operand(temp, t));
            
            // Compute prod({t, gamma}) mod q
            prod_t_gamma_mod_q = std::optional(Array<MultiplyUint64Operand>(base_q.size(), false, nullptr));
            for (size_t i = 0; i < base_q.size(); i++) {
                const Modulus& modulus = base_q.base()[i];
                prod_t_gamma_mod_q.value()[i] = MultiplyUint64Operand(
                    utils::multiply_uint64_mod(
                        base_t_gamma.value().base()[0].value(),
                        base_t_gamma.value().base()[1].value(),
                        modulus
                    ),
                    modulus
                );
            }

            // Compute -prod(q)^(-1) mod {t, gamma}
            neg_inv_q_mod_t_gamma = std::optional(Array<MultiplyUint64Operand>(2, false, nullptr));
            for (size_t i = 0; i < 2; i++) {
                const Modulus& modulus = base_t_gamma.value().base()[i];
                temp = utils::modulo_uint(base_q.base_product(), modulus);
                try_invert = utils::try_invert_uint64_mod(temp, modulus, temp);
                if (!try_invert) {
                    throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert base_q mod t_gamma.");
                }
                neg_inv_q_mod_t_gamma.value()[i] = MultiplyUint64Operand(
                    utils::negate_uint64_mod(temp, modulus),
                    modulus
                );
            }
        }

        // Compute q[last]^(-1) mod q[i] for i = 0..last-1
        // This is used by modulus switching and rescaling
        Array<MultiplyUint64Operand> inv_q_last_mod_q(base_q.size() - 1, false, nullptr);
        const Modulus& last_q = base_q.base()[base_q.size() - 1];
        for (size_t i = 0; i < base_q.size() - 1; i++) {
            const Modulus& modulus = base_q.base()[i];
            try_invert = utils::try_invert_uint64_mod(last_q.value(), modulus, temp);
            if (!try_invert) {
                throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert q[last] mod q[i].");
            }
            inv_q_last_mod_q[i] = MultiplyUint64Operand(temp, modulus);
        }

        if (!t.is_zero()) {
            try_invert = utils::try_invert_uint64_mod(last_q.value(), t, temp);
            if (!try_invert) {
                throw std::invalid_argument("[RNSTool::RNSTool] Unable to invert q[last] mod t.");
            }
            inv_q_last_mod_t = temp;
            q_last_mod_t = t.reduce(last_q.value());
        }

        // set the members

        this->coeff_count_ = coeff_count;

        this->base_q_ = std::move(base_q);
        this->base_B_ = std::move(base_B);
        this->base_Bsk_ = std::move(base_Bsk);
        this->base_Bsk_m_tilde_ = std::move(base_Bsk_m_tilde);
        this->base_t_gamma_ = std::move(base_t_gamma);

        this->base_q_to_Bsk_conv_ = std::move(base_q_to_Bsk_conv);
        this->base_q_to_m_tilde_conv_ = std::move(base_q_to_m_tilde_conv);
        this->base_B_to_q_conv_ = std::move(base_B_to_q_conv);
        this->base_B_to_m_sk_conv_ = std::move(base_B_to_m_sk_conv);
        this->base_q_to_t_gamma_conv_ = std::move(base_q_to_t_gamma_conv);
        this->base_q_to_t_conv_ = std::move(base_q_to_t_conv);

        this->inv_prod_q_mod_Bsk_ = std::move(inv_prod_q_mod_Bsk);
        this->neg_inv_prod_q_mod_m_tilde_ = std::move(neg_inv_prod_q_mod_m_tilde);
        this->inv_prod_B_mod_m_sk_ = std::move(inv_prod_B_mod_m_sk);
        this->inv_gamma_mod_t_ = std::move(inv_gamma_mod_t);
        this->prod_B_mod_q_ = std::move(prod_B_mod_q);
        this->inv_m_tilde_mod_Bsk_ = std::move(inv_m_tilde_mod_Bsk);
        this->prod_q_mod_Bsk_ = std::move(prod_q_mod_Bsk);
        this->neg_inv_q_mod_t_gamma_ = std::move(neg_inv_q_mod_t_gamma);
        this->prod_t_gamma_mod_q_ = std::move(prod_t_gamma_mod_q);
        this->inv_q_last_mod_q_ = std::move(inv_q_last_mod_q);
        this->base_Bsk_ntt_tables_ = std::move(base_Bsk_ntt_tables);

        this->m_tilde_ = Box(new Modulus(m_tilde), false, nullptr);
        this->m_sk_ = Box(new Modulus(m_sk), false, nullptr);
        this->t_ = Box(new Modulus(t), false, nullptr);
        this->gamma_ = Box(new Modulus(gamma), false, nullptr);

        this->m_tilde_value_ = m_tilde_value;
        this->inv_q_last_mod_t_ = inv_q_last_mod_t;
        this->q_last_mod_t_ = q_last_mod_t;
        this->q_last_half_ = last_q.value() >> 1;

        this->device = false;
        
    }

    template <typename T>
    static std::optional<T> optional_clone(const std::optional<T>& opt, MemoryPoolHandle pool) {
        if (opt.has_value()) {
            return std::optional<T>(opt.value().clone(pool));
        }
        return std::nullopt;
    }

    RNSTool RNSTool::clone(MemoryPoolHandle pool) const {
        RNSTool cloned;

        cloned.coeff_count_ = this->coeff_count_;

        cloned.base_q_ = this->base_q_.clone(pool);
        cloned.base_B_ = this->base_B_.clone(pool);
        cloned.base_Bsk_ = this->base_Bsk_.clone(pool);
        cloned.base_Bsk_m_tilde_ = this->base_Bsk_m_tilde_.clone(pool);
        cloned.base_t_gamma_ = optional_clone(this->base_t_gamma_, pool);

        cloned.base_q_to_Bsk_conv_ = this->base_q_to_Bsk_conv_.clone(pool);
        cloned.base_q_to_m_tilde_conv_ = this->base_q_to_m_tilde_conv_.clone(pool);
        cloned.base_B_to_q_conv_ = this->base_B_to_q_conv_.clone(pool);
        cloned.base_B_to_m_sk_conv_ = this->base_B_to_m_sk_conv_.clone(pool);
        cloned.base_q_to_t_gamma_conv_ = optional_clone(this->base_q_to_t_gamma_conv_, pool);
        cloned.base_q_to_t_conv_ = optional_clone(this->base_q_to_t_conv_, pool);

        cloned.inv_prod_q_mod_Bsk_ = this->inv_prod_q_mod_Bsk_.clone(pool);
        cloned.neg_inv_prod_q_mod_m_tilde_ = this->neg_inv_prod_q_mod_m_tilde_;
        cloned.inv_prod_B_mod_m_sk_ = this->inv_prod_B_mod_m_sk_;
        cloned.inv_gamma_mod_t_ = this->inv_gamma_mod_t_;
        cloned.prod_B_mod_q_ = this->prod_B_mod_q_.clone(pool);
        cloned.inv_m_tilde_mod_Bsk_ = this->inv_m_tilde_mod_Bsk_.clone(pool);
        cloned.prod_q_mod_Bsk_ = this->prod_q_mod_Bsk_.clone(pool);
        cloned.neg_inv_q_mod_t_gamma_ = optional_clone(this->neg_inv_q_mod_t_gamma_, pool);
        cloned.prod_t_gamma_mod_q_ = optional_clone(this->prod_t_gamma_mod_q_, pool);
        cloned.inv_q_last_mod_q_ = this->inv_q_last_mod_q_.clone(pool);
        cloned.base_Bsk_ntt_tables_ = this->base_Bsk_ntt_tables_.clone(pool);
        cloned.m_tilde_ = this->m_tilde_.clone(pool);
        cloned.m_sk_ = this->m_sk_.clone(pool);

        cloned.t_ = this->t_.clone(pool);
        cloned.gamma_ = this->gamma_.clone(pool);
        cloned.m_tilde_value_ = this->m_tilde_value_;
        cloned.inv_q_last_mod_t_ = this->inv_q_last_mod_t_;
        cloned.q_last_mod_t_ = this->q_last_mod_t_;
        cloned.q_last_half_ = this->q_last_half_;

        cloned.device = this->device;

        return cloned;
    }

    template <typename T>
    static void optional_to_device_inplace(std::optional<T>& opt, MemoryPoolHandle pool) {
        if (opt.has_value()) {
            opt.value().to_device_inplace(pool);
        }
    }

    void RNSTool::to_device_inplace(MemoryPoolHandle pool) {
        if (this->on_device()) {
            return;
        }
        
        this->base_q_.to_device_inplace(pool);
        this->base_B_.to_device_inplace(pool);
        this->base_Bsk_.to_device_inplace(pool);
        this->base_Bsk_m_tilde_.to_device_inplace(pool);
        optional_to_device_inplace(this->base_t_gamma_, pool);

        this->base_q_to_Bsk_conv_.to_device_inplace(pool);
        this->base_q_to_m_tilde_conv_.to_device_inplace(pool);
        this->base_B_to_q_conv_.to_device_inplace(pool);
        this->base_B_to_m_sk_conv_.to_device_inplace(pool);
        optional_to_device_inplace(this->base_q_to_t_gamma_conv_, pool);
        optional_to_device_inplace(this->base_q_to_t_conv_, pool);

        this->inv_prod_q_mod_Bsk_.to_device_inplace(pool);
        this->prod_B_mod_q_.to_device_inplace(pool);
        this->inv_m_tilde_mod_Bsk_.to_device_inplace(pool);
        this->prod_q_mod_Bsk_.to_device_inplace(pool);
        optional_to_device_inplace(this->neg_inv_q_mod_t_gamma_, pool);
        optional_to_device_inplace(this->prod_t_gamma_mod_q_, pool);
        this->inv_q_last_mod_q_.to_device_inplace(pool);
        for (size_t i = 0; i < base_Bsk_ntt_tables_.size(); i++) {
            this->base_Bsk_ntt_tables_[i].to_device_inplace(pool);
        }
        this->base_Bsk_ntt_tables_.to_device_inplace(pool);

        this->m_tilde_.to_device_inplace(pool);
        this->m_sk_.to_device_inplace(pool);
        this->t_.to_device_inplace(pool);
        this->gamma_.to_device_inplace(pool);

        this->device = true;
    }

    static void host_divide_and_round_q_last_inplace(const RNSTool& self, Slice<uint64_t> input) {
        
        size_t base_q_size = self.base_q().size();
        size_t coeff_count = self.coeff_count();
        size_t last_input_offset = (base_q_size - 1) * coeff_count;
        size_t half = self.q_last_half();

        Slice<uint64_t> input_last = input.slice(last_input_offset, last_input_offset + coeff_count);
        Array<uint64_t> temp(coeff_count, false, nullptr);
        for (size_t i = 0; i < base_q_size - 1; i++) {
            ConstPointer<Modulus> modulus = self.base_q().base().at(i);
            Slice<uint64_t> input_i = input.slice(i * coeff_count, (i + 1) * coeff_count);
            // (ct mod qk) mod qi
            utils::modulo(input_last.as_const(), modulus, temp.reference());
            // Subtract rounding correction here; the negative sign will turn into a plus in the next subtraction
            uint64_t half_mod = modulus->reduce(half);
            utils::sub_scalar_inplace(temp.reference(), half_mod, modulus);
            // (ct mod qi) - (ct mod qk) mod qi
            utils::sub_inplace(input_i, temp.const_reference(), modulus);
            // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
            utils::multiply_uint64operand_inplace(input_i, self.inv_q_last_mod_q().at(i), modulus);
        }

    }

    __global__ static void kernel_divide_and_round_q_last_inplace(
        ConstSlice<Modulus> base_q, size_t coeff_count, size_t q_last_half,
        ConstSlice<MultiplyUint64Operand> inv_q_last_mod_q, 
        Slice<uint64_t> input
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_q_size = base_q.size();
        if (global_index >= coeff_count * (base_q_size - 1)) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;

        size_t last_input_offset = (base_q.size() - 1) * coeff_count;
        Slice<uint64_t> input_last = input.slice(last_input_offset, last_input_offset + coeff_count);
        Slice<uint64_t> input_i = input.slice(i * coeff_count, (i + 1) * coeff_count);
        uint64_t temp;
        const Modulus& modulus = *base_q.at(i);
        // (ct mod qk) mod qi
        temp = modulus.reduce(input_last[j]);
        // Subtract rounding correction here; the negative sign will turn into a plus in the next subtraction
        uint64_t half_mod = modulus.reduce(q_last_half);
        temp = utils::sub_uint64_mod(temp, half_mod, modulus);
        // (ct mod qi) - (ct mod qk) mod qi
        input_i[j] = utils::sub_uint64_mod(input_i[j], temp, modulus);
        // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
        input_i[j] = utils::multiply_uint64operand_mod(input_i[j], inv_q_last_mod_q[i], modulus);
    }

    void RNSTool::divide_and_round_q_last_inplace(Slice<uint64_t> input) const {
        bool device = this->on_device();
        if (!utils::device_compatible(*this, input)) {
            throw std::invalid_argument("[RNSTool::divide_and_round_q_last_inplace] RNSTool and input must be on the same device.");
        }
        size_t base_q_size = this->base_q().size();
        size_t coeff_count = this->coeff_count();
        Slice<uint64_t> input_last = input.slice((base_q_size - 1) * coeff_count, base_q_size * coeff_count);
        // Add (qi-1)/2 to change from flooring to rounding
        utils::add_scalar_inplace(input_last, this->q_last_half(), this->base_q().base().at(base_q_size - 1));
        if (device) {
            size_t block_count = utils::ceil_div(this->coeff_count() * (base_q_size - 1), utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_divide_and_round_q_last_inplace<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                this->base_q().base(),
                this->coeff_count(),
                this->q_last_half(),
                this->inv_q_last_mod_q(),
                input
            );
            utils::stream_sync();
        } else {
            host_divide_and_round_q_last_inplace(*this, input);
        }
    }

    static void host_divide_and_round_q_last_ntt_inplace_step1(const RNSTool& self, Slice<uint64_t> input, Slice<uint64_t> temp) {
        size_t base_q_size = self.base_q().size();
        size_t coeff_count = self.coeff_count();
        Slice<uint64_t> input_last = input.slice((base_q_size - 1) * coeff_count, base_q_size * coeff_count);
        ConstPointer<Modulus> last_modulus = self.base_q().base().at(base_q_size - 1);

        for (size_t i = 0; i < base_q_size - 1; i++) {
            ConstPointer<Modulus> modulus = self.base_q().base().at(i);
            Slice<uint64_t> temp_i = temp.slice(i * coeff_count, (i + 1) * coeff_count);
            if (modulus->value() < last_modulus->value()) {
                utils::modulo(input_last.as_const(), modulus, temp_i);
            } else {
                utils::set_uint(input_last.as_const(), coeff_count, temp_i);
            }
            uint64_t half_mod = modulus->reduce(self.q_last_half());
            utils::sub_scalar_inplace(temp_i, half_mod, modulus);
        }
    }

    __global__ static void kernel_divide_and_round_q_last_ntt_inplace_step1(
        ConstSlice<Modulus> base_q, size_t coeff_count, size_t q_last_half,
        Slice<uint64_t> input, Slice<uint64_t> temp
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_q_size = base_q.size();
        if (global_index >= coeff_count * (base_q_size - 1)) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;

        const Modulus& last_modulus = *base_q.at(base_q_size - 1);
        size_t last_input_offset = (base_q.size() - 1) * coeff_count;
        Slice<uint64_t> input_last = input.slice(last_input_offset, last_input_offset + coeff_count);
        Slice<uint64_t> temp_i = temp.slice(i * coeff_count, (i + 1) * coeff_count);
        uint64_t temp_value;
        const Modulus& modulus = *base_q.at(i);
        if (modulus.value() < last_modulus.value()) {
            temp_value = modulus.reduce(input_last[j]);
        } else {
            temp_value = input_last[j];
        }
        uint64_t half_mod = modulus.reduce(q_last_half);
        temp_value = utils::sub_uint64_mod(temp_value, half_mod, modulus);
        temp_i[j] = temp_value;
    }

    static void divide_and_round_q_last_ntt_inplace_step1(const RNSTool& self, Slice<uint64_t> input, Slice<uint64_t> temp) {
        bool device = self.on_device();
        size_t base_q_size = self.base_q().size();
        if (device) {
            size_t block_count = utils::ceil_div(self.coeff_count() * (base_q_size - 1), utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_divide_and_round_q_last_ntt_inplace_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                self.base_q().base(),
                self.coeff_count(),
                self.q_last_half(),
                input,
                temp
            );
            utils::stream_sync();
        } else {
            host_divide_and_round_q_last_ntt_inplace_step1(self, input, temp);
        }
    }
    
    static void host_divide_and_round_q_last_ntt_inplace_step2(const RNSTool& self, Slice<uint64_t> input, ConstSlice<uint64_t> temp) {
        
        size_t base_q_size = self.base_q().size();
        size_t coeff_count = self.coeff_count();

        for (size_t i = 0; i < base_q_size - 1; i++) {
            ConstPointer<Modulus> modulus = self.base_q().base().at(i);
            Slice<uint64_t> input_i = input.slice(i * coeff_count, (i + 1) * coeff_count);
            ConstSlice<uint64_t> temp_i = temp.const_slice(i * coeff_count, (i + 1) * coeff_count);
            uint64_t qi_lazy = modulus->value() << 2;
            utils::add_scalar_inplace(input_i, qi_lazy, modulus);
            utils::sub_inplace(input_i, temp_i, modulus);
            utils::multiply_uint64operand_inplace(input_i, self.inv_q_last_mod_q().at(i), modulus);
        }
    }

    __global__ static void kernel_divide_and_round_q_last_ntt_inplace_step2(
        ConstSlice<Modulus> base_q, size_t coeff_count,
        ConstSlice<MultiplyUint64Operand> inv_q_last_mod_q,
        Slice<uint64_t> input, ConstSlice<uint64_t> temp
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_q_size = base_q.size();
        if (global_index >= coeff_count * (base_q_size - 1)) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;

        Slice<uint64_t> input_i = input.slice(i * coeff_count, (i + 1) * coeff_count);
        ConstSlice<uint64_t> temp_i = temp.const_slice(i * coeff_count, (i + 1) * coeff_count);
        uint64_t temp_value;
        const Modulus& modulus = *base_q.at(i);
        uint64_t qi_lazy = modulus.value() << 2;
        input_i[j] = utils::add_uint64_mod(input_i[j], qi_lazy, modulus);
        temp_value = temp_i[j];
        input_i[j] = utils::sub_uint64_mod(input_i[j], temp_value, modulus);
        input_i[j] = utils::multiply_uint64operand_mod(input_i[j], inv_q_last_mod_q[i], modulus);
    }

    static void divide_and_round_q_last_ntt_inplace_step2(const RNSTool& self, Slice<uint64_t> input, ConstSlice<uint64_t> temp) {
        bool device = self.on_device();
        size_t base_q_size = self.base_q().size();
        if (device) {
            size_t block_count = utils::ceil_div(self.coeff_count() * (base_q_size - 1), utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_divide_and_round_q_last_ntt_inplace_step2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                self.base_q().base(),
                self.coeff_count(),
                self.inv_q_last_mod_q(),
                input,
                temp
            );
            utils::stream_sync();
        } else {
            host_divide_and_round_q_last_ntt_inplace_step2(self, input, temp);
        }
    }
    
    void RNSTool::divide_and_round_q_last_ntt_inplace(Slice<uint64_t> input, ConstSlice<NTTTables> rns_ntt_tables, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        if (!utils::device_compatible(*this, input, rns_ntt_tables)) {
            throw std::invalid_argument("[RNSTool::divide_and_round_q_last_ntt_inplace] RNSTool, input and ntt_tables must be on the same device.");
        }

        size_t base_q_size = this->base_q().size();
        size_t coeff_count = this->coeff_count();
        Slice<uint64_t> input_last = input.slice((base_q_size - 1) * coeff_count, base_q_size * coeff_count);
        
        utils::intt_inplace(input_last, coeff_count, rns_ntt_tables.at(base_q_size - 1));
        
        // Add (qi-1)/2 to change from flooring to rounding
        utils::add_scalar_inplace(input_last, this->q_last_half(), this->base_q().base().at(base_q_size - 1));

        Array<uint64_t> temp(coeff_count * (base_q_size - 1), device, pool);
        divide_and_round_q_last_ntt_inplace_step1(*this, input, temp.reference());
        
        utils::ntt_lazy_inplace_p(temp.reference(), coeff_count, rns_ntt_tables.const_slice(0, base_q_size - 1));
    
        divide_and_round_q_last_ntt_inplace_step2(*this, input, temp.const_reference());
    }

    static void host_fast_b_conv_sk_step1(const RNSTool& self, ConstSlice<uint64_t> input, Slice<uint64_t> destination, ConstSlice<uint64_t> temp) {
        size_t coeff_count = self.coeff_count();
        const Modulus& m_sk = *self.m_sk();
        uint64_t m_sk_value = m_sk.value();
        uint64_t m_sk_div_2 = m_sk_value >> 1;
        size_t base_B_size = self.base_B().size();
        for (size_t j = 0; j < coeff_count; j++) {
            uint64_t alpha_sk = multiply_uint64operand_mod(
                temp[j] + (m_sk_value - input[base_B_size * coeff_count + j]),
                self.inv_prod_B_mod_m_sk(),
                m_sk
            );
            for (size_t i = 0; i < self.base_q().size(); i++) {
                const Modulus& modulus = *self.base_q().base().at(i);
                MultiplyUint64Operand prod_B_mod_q_elt(self.prod_B_mod_q()[i], modulus);
                MultiplyUint64Operand neg_prod_B_mod_q_elt(modulus.value() - self.prod_B_mod_q()[i], modulus);
                uint64_t& dest = destination[i * coeff_count + j];
                if (alpha_sk > m_sk_div_2) {
                    dest = utils::multiply_uint64operand_add_uint64_mod(
                        utils::negate_uint64_mod(alpha_sk, m_sk), prod_B_mod_q_elt, dest, modulus
                    );
                } else {
                    dest = utils::multiply_uint64operand_add_uint64_mod(
                        alpha_sk, neg_prod_B_mod_q_elt, dest, modulus
                    );
                }
            }
        }
    }

    __global__ static void kernel_fast_b_conv_sk_step1(
        ConstSlice<Modulus> base_B,
        ConstSlice<Modulus> base_q,
        ConstPointer<Modulus> m_sk,
        MultiplyUint64Operand inv_prod_B_mod_m_sk,
        ConstSlice<uint64_t> prod_B_mod_q,
        size_t coeff_count,
        ConstSlice<uint64_t> input,
        Slice<uint64_t> destination,
        ConstSlice<uint64_t> temp
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_B_size = base_B.size();
        size_t base_q_size = base_q.size();
        if (global_index >= coeff_count * base_q_size) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;
        uint64_t m_sk_value = m_sk->value();
        uint64_t m_sk_div_2 = m_sk_value >> 1;
        uint64_t alpha_sk = multiply_uint64operand_mod(
            temp[j] + (m_sk_value - input[base_B_size * coeff_count + j]),
            inv_prod_B_mod_m_sk,
            *m_sk
        );
        const Modulus& modulus = *base_q.at(i);
        MultiplyUint64Operand prod_B_mod_q_elt(prod_B_mod_q[i], modulus);
        MultiplyUint64Operand neg_prod_B_mod_q_elt(modulus.value() - prod_B_mod_q[i], modulus);
        uint64_t& dest = destination[i * coeff_count + j];
        if (alpha_sk > m_sk_div_2) {
            dest = utils::multiply_uint64operand_add_uint64_mod(
                utils::negate_uint64_mod(alpha_sk, *m_sk), prod_B_mod_q_elt, dest, modulus
            );
        } else {
            dest = utils::multiply_uint64operand_add_uint64_mod(
                alpha_sk, neg_prod_B_mod_q_elt, dest, modulus
            );
        }
    }
    
    void RNSTool::fast_b_conv_sk(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        if (!utils::device_compatible(*this, input, destination)) {
            throw std::invalid_argument("[RNSTool::fast_b_conv_sk] RNSTool, input and destination must be on the same device.");
        }
        size_t coeff_count = this->coeff_count();
        const RNSBase& base_B = this->base_B();
        size_t base_B_size = base_B.size();

        // Fast convert B -> q; input is in Bsk but we only use B
        this->base_B_to_q_conv().fast_convert_array(input.const_slice(0, base_B_size * coeff_count), destination, pool);
        
        // Compute alpha_sk
        // Fast convert B -> {m_sk}; input is in Bsk but we only use B
        Array<uint64_t> temp(coeff_count, device, pool);
        this->base_B_to_m_sk_conv().fast_convert_array(input.const_slice(0, base_B_size * coeff_count), temp.reference(), pool);
        
        if (device) {
            size_t base_q_size = this->base_q().size();
            size_t block_count = utils::ceil_div(coeff_count * base_q_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_fast_b_conv_sk_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                base_B.base(),
                this->base_q().base(),
                this->m_sk(),
                this->inv_prod_B_mod_m_sk(),
                this->prod_B_mod_q(),
                coeff_count,
                input,
                destination,
                temp.const_reference()
            );
            utils::stream_sync();
        } else {
            host_fast_b_conv_sk_step1(*this, input, destination, temp.const_reference());
        }

    }

    static void host_sm_mrq(const RNSTool& self, ConstSlice<uint64_t> input, Slice<uint64_t> destination) {
        ConstSlice<Modulus> base_Bsk = self.base_Bsk().base();
        size_t base_Bsk_size = base_Bsk.size();
        size_t coeff_count = self.coeff_count();
        ConstSlice<uint64_t> input_m_tilde = input.const_slice(base_Bsk_size * coeff_count, (base_Bsk_size + 1) * coeff_count);
        uint64_t m_tilde_div_2 = self.m_tilde()->value() >> 1;
        Array<MultiplyUint64Operand> prod_q_mod_Bsk_elt(base_Bsk_size, false, nullptr);
        for (size_t i = 0; i < base_Bsk_size; i++) {
            const Modulus& modulus = *base_Bsk.at(i);
            prod_q_mod_Bsk_elt[i] = MultiplyUint64Operand(self.prod_q_mod_Bsk()[i], modulus);
        }
        for (size_t j = 0; j < coeff_count; j++) {
            uint64_t r_m_tilde = utils::multiply_uint64operand_mod(
                input_m_tilde[j], 
                self.neg_inv_prod_q_mod_m_tilde(),
                *self.m_tilde()
            );
            for (size_t i = 0; i < base_Bsk_size; i++) {
                const Modulus& modulus = *base_Bsk.at(i);
                uint64_t temp = r_m_tilde;
                if (temp >= m_tilde_div_2) {
                    temp += modulus.value() - self.m_tilde()->value();
                }
                destination[i * coeff_count + j] = utils::multiply_uint64operand_mod(
                    utils::multiply_uint64operand_add_uint64_mod(
                        temp,
                        prod_q_mod_Bsk_elt[i],
                        input[i * coeff_count + j],
                        modulus
                    ),
                    self.inv_m_tilde_mod_Bsk()[i],
                    modulus
                );
            }
        }
    }

    __global__ static void kernel_sm_mrq(
        ConstSlice<Modulus> base_Bsk,
        ConstPointer<Modulus> m_tilde,
        MultiplyUint64Operand neg_inv_prod_q_mod_m_tilde,
        ConstSlice<uint64_t> prod_q_mod_Bsk,
        ConstSlice<MultiplyUint64Operand> inv_m_tilde_mod_Bsk,
        size_t coeff_count,
        ConstSlice<uint64_t> input,
        Slice<uint64_t> destination
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_Bsk_size = base_Bsk.size();
        if (global_index >= coeff_count) return;
        size_t j = global_index % coeff_count;
        for (size_t i = 0; i < base_Bsk_size; i++) {
            const Modulus& modulus = *base_Bsk.at(i);
            uint64_t m_tilde_div_2 = m_tilde->value() >> 1;
            uint64_t r_m_tilde = utils::multiply_uint64operand_mod(
                input[base_Bsk_size * coeff_count + j], 
                neg_inv_prod_q_mod_m_tilde,
                *m_tilde
            );
            uint64_t temp = r_m_tilde;
            if (temp >= m_tilde_div_2) {
                temp += modulus.value() - m_tilde->value();
            }
            uint64_t& dest = destination[i * coeff_count + j];
            dest = utils::multiply_uint64operand_mod(
                utils::multiply_uint64operand_add_uint64_mod(
                    temp,
                    MultiplyUint64Operand(prod_q_mod_Bsk[i], modulus),
                    input[i * coeff_count + j],
                    modulus
                ),
                inv_m_tilde_mod_Bsk[i],
                modulus
            );
        }
    }
    
    void RNSTool::sm_mrq(ConstSlice<uint64_t> input, Slice<uint64_t> destination) const {
        bool device = this->on_device();
        if (!utils::device_compatible(*this, input, destination)) {
            throw std::invalid_argument("[RNSTool::sm_mrq] RNSTool, input and destination must be on the same device.");
        }
        size_t coeff_count = this->coeff_count();
        const RNSBase& base_Bsk = this->base_Bsk();
        if (device) {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_sm_mrq<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                base_Bsk.base(),
                this->m_tilde(),
                this->neg_inv_prod_q_mod_m_tilde(),
                this->prod_q_mod_Bsk(),
                this->inv_m_tilde_mod_Bsk(),
                coeff_count,
                input,
                destination
            );
            utils::stream_sync();
        } else {
            host_sm_mrq(*this, input, destination);
        }
    }

    static void host_fast_floor(const RNSTool& self, ConstSlice<uint64_t> input, Slice<uint64_t> destination) {
        size_t base_q_size = self.base_q().size();
        size_t base_Bsk_size = self.base_Bsk().size();
        size_t coeff_count = self.coeff_count();
        input = input.const_slice(base_q_size * coeff_count, input.size());
        for (size_t i = 0; i < base_Bsk_size; i++) {
            for (size_t j = 0; j < coeff_count; j++) {
                size_t index = i * coeff_count + j;
                destination[index] = utils::multiply_uint64operand_mod(
                    input[index] + self.base_Bsk().base()[i].value() - destination[index],
                    self.inv_prod_q_mod_Bsk()[i],
                    self.base_Bsk().base()[i]
                );
            }
        }
    }

    __global__ static void kernel_fast_floor(
        ConstSlice<Modulus> base_Bsk,
        ConstSlice<MultiplyUint64Operand> inv_prod_q_mod_Bsk,
        size_t coeff_count,
        size_t base_q_size,
        ConstSlice<uint64_t> input,
        Slice<uint64_t> destination
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_Bsk_size = base_Bsk.size();
        if (global_index >= coeff_count * base_Bsk_size) return;
        size_t i = global_index / coeff_count;
        uint64_t& dest = destination[global_index];
        dest = utils::multiply_uint64operand_mod(
            input[global_index + base_q_size * coeff_count] + base_Bsk[i].value() - dest,
            inv_prod_q_mod_Bsk[i],
            base_Bsk[i]
        );
    }

    void RNSTool::fast_floor(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool) const {
        size_t base_q_size = this->base_q().size();
        size_t base_Bsk_size = this->base_Bsk().size();
        size_t coeff_count = this->coeff_count();

        this->base_q_to_Bsk_conv().fast_convert_array(
            input.const_slice(0, base_q_size * coeff_count),
            destination, pool
        );

        if (this->on_device()) {
            size_t block_count = utils::ceil_div(coeff_count * base_Bsk_size, utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_fast_floor<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                this->base_Bsk().base(),
                this->inv_prod_q_mod_Bsk(),
                coeff_count,
                base_q_size,
                input,
                destination
            );
            utils::stream_sync();
        } else {
            host_fast_floor(*this, input, destination);
        }

    }
    
    void RNSTool::fast_floor_fast_b_conv_sk(ConstSlice<uint64_t> input_q, ConstSlice<uint64_t> input_Bsk, Slice<uint64_t> destination, MemoryPoolHandle pool) const {
        if (!utils::device_compatible(*this, input_q, input_Bsk, destination)) {
            throw std::invalid_argument("[RNSTool::fast_floor_fast_b_conv_sk] RNSTool, input_q, input_Bsk and destination must be on the same device.");
        }
        bool device = this->on_device();
        size_t base_q_size = this->base_q().size();
        size_t base_Bsk_size = this->base_Bsk().size();
        size_t coeff_count = this->coeff_count();
        ConstSlice<Modulus> base_q = this->base_q().base();
        ConstSlice<Modulus> base_Bsk = this->base_Bsk().base();
        size_t dest_size = input_q.size() / base_q_size / coeff_count;
        if (!device) {
            Buffer<uint64_t> temp_q_Bsk(base_q_size + base_Bsk_size, coeff_count, device, pool);
            Buffer<uint64_t> temp_Bsk(base_Bsk_size, coeff_count, device, pool);
            uint64_t plain_modulus_value = this->t()->value();
            for (size_t i = 0; i < dest_size; i++) {
                // Bring together the base q and base Bsk components into a single allocation
                // Step (6): multiply base q components by t (plain_modulus)
                utils::multiply_scalar_p(
                    input_q.const_slice(i*coeff_count*base_q_size, (i+1)*coeff_count*base_q_size),
                    plain_modulus_value,
                    coeff_count,
                    base_q,
                    temp_q_Bsk.components(0, base_q_size)
                );
                utils::multiply_scalar_p(
                    input_Bsk.const_slice(i*coeff_count*base_Bsk_size, (i+1)*coeff_count*base_Bsk_size),
                    plain_modulus_value,
                    coeff_count,
                    base_Bsk,
                    temp_q_Bsk.components(base_q_size, base_q_size + base_Bsk_size)
                );
                // Step (7): divide by q and floor, producing a result in base Bsk
                this->fast_floor(temp_q_Bsk.const_reference(), temp_Bsk.reference(), pool);
                // Step (8): use Shenoy-Kumaresan method to convert the result to base q and write to encrypted1
                this->fast_b_conv_sk(temp_Bsk.const_reference(), destination.slice(i*coeff_count*base_q_size, (i+1)*coeff_count*base_q_size), pool);
            }
        } else {
            fgk::rns_tool::fast_floor_fast_b_conv_sk(
                input_q, input_Bsk, *this, dest_size, destination, pool
            );
        }
        
    }
    
    void RNSTool::fast_b_conv_m_tilde(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        size_t base_q_size = this->base_q().size();
        size_t base_Bsk_size = this->base_Bsk().size();
        size_t coeff_count = this->coeff_count();
        Buffer<uint64_t> temp(base_q_size, coeff_count, device, pool);
        utils::multiply_scalar_p(input, this->m_tilde_value(), coeff_count, this->base_q().base(), temp.reference());
        this->base_q_to_Bsk_conv().fast_convert_array(temp.const_reference(),
            destination.slice(0, base_Bsk_size * coeff_count), pool);
        this->base_q_to_m_tilde_conv().fast_convert_array(temp.const_reference(),
            destination.slice(base_Bsk_size * coeff_count, (base_Bsk_size + 1) * coeff_count), pool);
    }

    void RNSTool::fast_b_conv_m_tilde_sm_mrq(ConstSlice<uint64_t> input, Slice<uint64_t> destination, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        size_t base_Bsk_size = this->base_Bsk().size();
        size_t coeff_count = this->coeff_count();
        if (!input.on_device()) {
            Buffer<uint64_t> temp(base_Bsk_size + 1, coeff_count, device, pool);
            this->fast_b_conv_m_tilde(input, temp.reference(), pool);
            this->sm_mrq(temp.const_reference(), destination);
        } else {
            fgk::rns_tool::fast_b_conv_m_tilde_sm_mrq(
                input, coeff_count, this->m_tilde_value(), this->base_q().base(),
                this->base_q_to_Bsk_conv(), 
                this->base_q_to_m_tilde_conv(),
                this->neg_inv_prod_q_mod_m_tilde(),
                this->prod_q_mod_Bsk(),
                this->inv_m_tilde_mod_Bsk(),
                destination,
                pool
            );
        }
    }

    static void host_decrypt_scale_and_round_step1(const RNSTool& self, Slice<uint64_t> destination, size_t coeff_count, ConstSlice<uint64_t> temp_t_gamma) {
        uint64_t gamma = self.gamma()->value();
        uint64_t gamma_div_2 = gamma >> 1;
        const Modulus& t = *self.t();
        for (size_t i = 0; i < coeff_count; i++) {
            if (temp_t_gamma[coeff_count + i] > gamma_div_2) {
                destination[i] = add_uint64_mod(
                    temp_t_gamma[i], t.reduce(gamma - temp_t_gamma[coeff_count + i]), t
                );
            } else {
                destination[i] = sub_uint64_mod(
                    temp_t_gamma[i], t.reduce(temp_t_gamma[coeff_count + i]), t
                );
            }
            if (destination[i] != 0) {
                destination[i] = multiply_uint64operand_mod(destination[i], self.inv_gamma_mod_t(), t);
            }
        }
    }

    __global__ static void kernel_host_decrypt_scale_and_round_step1(
        ConstPointer<Modulus> gamma,
        ConstPointer<Modulus> t,
        MultiplyUint64Operand inv_gamma_mod_t,
        size_t coeff_count,
        Slice<uint64_t> destination,
        ConstSlice<uint64_t> temp_t_gamma
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coeff_count) return;
        uint64_t gamma_value = gamma->value();
        uint64_t gamma_div_2 = gamma_value >> 1;
        uint64_t temp;
        uint64_t& dest = destination[i];
        if (temp_t_gamma[coeff_count + i] > gamma_div_2) {
            temp = add_uint64_mod(
                temp_t_gamma[i], t->reduce(gamma_value - temp_t_gamma[coeff_count + i]), *t
            );
        } else {
            temp = sub_uint64_mod(
                temp_t_gamma[i], t->reduce(temp_t_gamma[coeff_count + i]), *t
            );
        }
        if (temp != 0) {
            dest = multiply_uint64operand_mod(temp, inv_gamma_mod_t, *t);
        } else {
            dest = 0;
        }
    }

    static void decrypt_scale_and_round_step1(const RNSTool& self, Slice<uint64_t> destination, size_t coeff_count, ConstSlice<uint64_t> temp_t_gamma) {
        bool device = self.on_device();
        if (device) {
            size_t block_count = utils::ceil_div(coeff_count, utils::KERNEL_THREAD_COUNT);
            utils::set_device(destination.device_index());
            kernel_host_decrypt_scale_and_round_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                self.gamma(),
                self.t(),
                self.inv_gamma_mod_t(),
                coeff_count,
                destination,
                temp_t_gamma
            );
            utils::stream_sync();
        } else {
            host_decrypt_scale_and_round_step1(self, destination, coeff_count, temp_t_gamma);
        }
    }
    
    void RNSTool::decrypt_scale_and_round(ConstSlice<uint64_t> phase, size_t phase_coeff_count, Slice<uint64_t> destination, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        if (!utils::device_compatible(*this, phase, destination)) {
            throw std::invalid_argument("[RNSTool::decrypt_scale_and_round] RNSTool, phase and destination must be on the same device.");
        }
        size_t base_q_size = this->base_q().size();
        size_t base_t_gamma_size = this->base_t_gamma().size();

        // Compute |gamma * t|_qi * ct(s)
        Array<uint64_t> temp(phase_coeff_count * base_q_size, device, pool);
        utils::multiply_uint64operand_p(
            phase.const_slice(0, base_q_size * phase_coeff_count),
            this->prod_t_gamma_mod_q(),
            phase_coeff_count,
            this->base_q().base(),
            temp.reference()
        );

        // Make another temp destination to get the poly in mod {t, gamma}
        Array<uint64_t> temp_t_gamma(phase_coeff_count * base_t_gamma_size, device, pool);
        this->base_q_to_t_gamma_conv()
            .fast_convert_array(temp.const_reference(), temp_t_gamma.reference(), pool);
        
        // Multiply by -prod(q)^(-1) mod {t, gamma}
        utils::multiply_uint64operand_inplace_p(
            temp_t_gamma.reference(),
            this->neg_inv_q_mod_t_gamma(),
            phase_coeff_count,
            this->base_t_gamma().base()
        );

        // Need to correct values in temp_t_gamma (gamma component only) which are
        // larger than floor(gamma/2)
        decrypt_scale_and_round_step1(*this, destination, phase_coeff_count, temp_t_gamma.const_reference());
    }

    static void host_mod_t_and_divide_q_last_inplace_step1(const RNSTool& self, Slice<uint64_t> input, ConstSlice<uint64_t> neg_c_last_mod_t) {
        if (self.on_device()) {
            throw std::logic_error("[host_mod_t_and_divide_q_last_inplace_step1] Unreachable.");
        }
        size_t base_q_size = self.base_q().size();
        size_t coeff_count = self.coeff_count();
        Array<uint64_t> delta_mod_q_i(coeff_count, false, nullptr);
        uint64_t last_modulus_value = self.base_q().base().at(base_q_size - 1)->value();
        for (size_t i = 0; i < base_q_size - 1; i++) {

            // delta_mod_q_i = neg_c_last_mod_t (mod q_i)
            ConstPointer<Modulus> modulus = self.base_q().base().at(i);
            utils::modulo(neg_c_last_mod_t, modulus, delta_mod_q_i.reference());

            // delta_mod_q_i *= q_last (mod q_i)
            utils::multiply_scalar_inplace(
                delta_mod_q_i.reference(), last_modulus_value, modulus
            );

            // c_i = c_i - c_last - neg_c_last_mod_t * q_last (mod 2q_i)
            uint64_t two_times_q_i = modulus->value() << 1;
            for (size_t j = 0; j < coeff_count; j++) {
                input[i * coeff_count + j] += two_times_q_i - modulus->reduce(
                    input[(base_q_size - 1) * coeff_count + j]
                ) - delta_mod_q_i[j];
            }
            
            // c_i = c_i * inv_q_last_mod_q_i (mod q_i)
            utils::multiply_uint64operand_inplace(
                input.slice(i * coeff_count, (i + 1) * coeff_count),
                self.inv_q_last_mod_q().at(i),
                modulus
            );
        }
    }

    __global__ static void kernel_mod_t_and_divide_q_last_inplace_step1(
        ConstSlice<Modulus> base_q,
        size_t coeff_count,
        ConstSlice<uint64_t> neg_c_last_mod_t,
        ConstSlice<MultiplyUint64Operand> inv_q_last_mod_q,
        Slice<uint64_t> input
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_q_size = base_q.size();
        if (global_index >= coeff_count * (base_q_size - 1)) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;

        uint64_t& dest = input[i * coeff_count + j];
        const Modulus& modulus = *base_q.at(i);
        uint64_t two_times_q_i = modulus.value() << 1;
        uint64_t delta_mod_q_i;
        // delta_mod_q_i = neg_c_last_mod_t (mod q_i)
        delta_mod_q_i = modulus.reduce(neg_c_last_mod_t[j]);
        // delta_mod_q_i *= q_last (mod q_i)
        delta_mod_q_i = utils::multiply_uint64_mod(delta_mod_q_i, base_q[base_q_size - 1].value(), modulus);
        // c_i = c_i - c_last - neg_c_last_mod_t * q_last (mod 2q_i)
        dest += two_times_q_i - modulus.reduce(input[(base_q_size - 1) * coeff_count + j]) - delta_mod_q_i;
        // c_i = c_i * inv_q_last_mod_q_i (mod q_i)
        dest = utils::multiply_uint64operand_mod(dest, inv_q_last_mod_q[i], modulus);
    }

    static void mod_t_and_divide_q_last_inplace_step1(const RNSTool& self, Slice<uint64_t> input, ConstSlice<uint64_t> neg_c_last_mod_t) {
        bool device = self.on_device();
        size_t base_q_size = self.base_q().size();
        size_t coeff_count = self.coeff_count();
        if (device) {
            size_t block_count = utils::ceil_div(coeff_count * (base_q_size - 1), utils::KERNEL_THREAD_COUNT);
            utils::set_device(input.device_index());
            kernel_mod_t_and_divide_q_last_inplace_step1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                self.base_q().base(),
                coeff_count,
                neg_c_last_mod_t,
                self.inv_q_last_mod_q(),
                input
            );
            utils::stream_sync();
        } else {
            host_mod_t_and_divide_q_last_inplace_step1(self, input, neg_c_last_mod_t);
        }
    }

    void RNSTool::mod_t_and_divide_q_last_inplace(Slice<uint64_t> input, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        if (!utils::device_compatible(*this, input)) {
            throw std::invalid_argument("[RNSTool::mod_t_and_divide_q_last_inplace] RNSTool and input must be on the same device.");
        }
        size_t modulus_size = this->base_q().size();
        size_t coeff_count = this->coeff_count();

        // neg_c_last_mod_t = - c_last (mod t)
        Array<uint64_t> neg_c_last_mod_t(coeff_count, device, pool);
        utils::modulo(
            input.const_slice((modulus_size - 1) * coeff_count, modulus_size * coeff_count),
            this->t(),
            neg_c_last_mod_t.reference()
        );
        utils::negate_inplace(neg_c_last_mod_t.reference(), this->t());
        if (this->inv_q_last_mod_t() != 1) {
            // neg_c_last_mod_t *= q_last^(-1) (mod t)
            utils::multiply_scalar_inplace(neg_c_last_mod_t.reference(), this->inv_q_last_mod_t(), this->t());
        }

        mod_t_and_divide_q_last_inplace_step1(*this, input, neg_c_last_mod_t.const_reference());

    }

    static void host_mod_t_and_divide_q_last_ntt_inplace_step1(const RNSTool& self, Slice<uint64_t> input, ConstSlice<uint64_t> neg_c_last_mod_t, ConstSlice<NTTTables> rns_ntt_tables, MemoryPoolHandle pool) {
        bool device = self.on_device();
        size_t base_q_size = self.base_q().size();
        size_t coeff_count = self.coeff_count();
        Array<uint64_t> delta_mod_q_i(coeff_count, device, pool);
        uint64_t last_modulus_value = self.base_q().base().at(base_q_size - 1)->value();
        for (size_t i = 0; i < base_q_size - 1; i++) {

            // delta_mod_q_i = neg_c_last_mod_t (mod q_i)
            ConstPointer<Modulus> modulus = self.base_q().base().at(i);
            utils::modulo(neg_c_last_mod_t, modulus, delta_mod_q_i.reference());

            // delta_mod_q_i *= q_last (mod q_i)
            utils::multiply_scalar_inplace(
                delta_mod_q_i.reference(), last_modulus_value, modulus
            );

            // c_i = c_i - c_last - neg_c_last_mod_t * q_last (mod 2q_i)
            //   first all all those to be subtracted to delta_mod_q_i
            for (size_t j = 0; j < coeff_count; j++) {
                delta_mod_q_i[j] = add_uint64_mod(
                    delta_mod_q_i[j], 
                    modulus->reduce(input[(base_q_size - 1) * coeff_count + j]),
                    *modulus
                );
            }
            ntt_inplace(delta_mod_q_i.reference(), coeff_count, rns_ntt_tables.at(i));
            //   then subtract them all
            for (size_t j = 0; j < coeff_count; j++) {
                input[i * coeff_count + j] = sub_uint64_mod(
                    input[i * coeff_count + j], delta_mod_q_i[j], *modulus
                );
            }
            
            // c_i = c_i * inv_q_last_mod_q_i (mod q_i)
            utils::multiply_uint64operand_inplace(
                input.slice(i * coeff_count, (i + 1) * coeff_count),
                self.inv_q_last_mod_q().at(i),
                modulus
            );
        }
    }

    __global__ static void kernel_mod_t_and_divide_q_last_ntt_inplace_step1_inner1(
        ConstSlice<Modulus> base_q,
        size_t coeff_count,
        ConstSlice<uint64_t> neg_c_last_mod_t,
        ConstSlice<uint64_t> input,
        Slice<uint64_t> delta_mod_q_i
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_q_size = base_q.size();
        if (global_index >= coeff_count * (base_q_size - 1)) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;

        const Modulus& modulus = *base_q.at(i);
        uint64_t result;
        // delta_mod_q_i = neg_c_last_mod_t (mod q_i)
        result = modulus.reduce(neg_c_last_mod_t[j]);
        // delta_mod_q_i *= q_last (mod q_i)
        result = utils::multiply_uint64_mod(result, base_q[base_q_size - 1].value(), modulus);
        // c_i = c_i - c_last - neg_c_last_mod_t * q_last (mod 2q_i)
        result = utils::add_uint64_mod(result, modulus.reduce(input[(base_q_size - 1) * coeff_count + j]), modulus);
        delta_mod_q_i[global_index] = result;
    }

    __global__ static void kernel_mod_t_and_divide_q_last_ntt_inplace_step1_inner2(
        ConstSlice<Modulus> base_q,
        size_t coeff_count,
        ConstSlice<MultiplyUint64Operand> inv_q_last_mod_q,
        Slice<uint64_t> input,
        ConstSlice<uint64_t> delta_mod_q_i
    ) {
        size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
        size_t base_q_size = base_q.size();
        if (global_index >= coeff_count * (base_q_size - 1)) return;
        size_t i = global_index / coeff_count;
        size_t j = global_index % coeff_count;

        uint64_t& dest = input[i * coeff_count + j];
        const Modulus& modulus = *base_q.at(i);
        // subtract
        dest = utils::sub_uint64_mod(dest, delta_mod_q_i[global_index], modulus);
        // c_i = c_i * inv_q_last_mod_q_i (mod q_i)
        dest = utils::multiply_uint64operand_mod(dest, inv_q_last_mod_q[i], modulus);
    }

    static void mod_t_and_divide_q_last_ntt_inplace_step1(const RNSTool& self, Slice<uint64_t> input, ConstSlice<uint64_t> neg_c_last_mod_t, ConstSlice<NTTTables> rns_ntt_tables, MemoryPoolHandle pool) {
        bool device = self.on_device();
        size_t base_q_size = self.base_q().size();
        size_t coeff_count = self.coeff_count();
        if (device) {
            size_t block_count = utils::ceil_div(coeff_count * (base_q_size - 1), utils::KERNEL_THREAD_COUNT);
            Array<uint64_t> delta_mod_q_i(coeff_count * (base_q_size - 1), device, pool);
            utils::set_device(input.device_index());
            kernel_mod_t_and_divide_q_last_ntt_inplace_step1_inner1<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                self.base_q().base(),
                coeff_count,
                neg_c_last_mod_t,
                input.as_const(),
                delta_mod_q_i.reference()
            );
            utils::stream_sync();
            utils::ntt_inplace_p(delta_mod_q_i.reference(), coeff_count, rns_ntt_tables.const_slice(0, base_q_size - 1));
            utils::set_device(input.device_index());
            kernel_mod_t_and_divide_q_last_ntt_inplace_step1_inner2<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                self.base_q().base(),
                coeff_count,
                self.inv_q_last_mod_q(),
                input,
                delta_mod_q_i.const_reference()
            );
            utils::stream_sync();
        } else {
            host_mod_t_and_divide_q_last_ntt_inplace_step1(self, input, neg_c_last_mod_t, rns_ntt_tables, pool);
        }
    }

    void RNSTool::mod_t_and_divide_q_last_ntt_inplace(Slice<uint64_t> input, ConstSlice<NTTTables> rns_ntt_tables, MemoryPoolHandle pool) const {
        bool device = this->on_device();
        if (!utils::device_compatible(input, rns_ntt_tables, *this)) {
            throw std::invalid_argument("[RNSTool::mod_t_and_divide_q_last_ntt_inplace] RNSTool, input, rns_ntt_tables must be on the same device.");
        }
        
        size_t modulus_size = this->base_q().size();
        size_t coeff_count = this->coeff_count();

        Slice<uint64_t> c_last = input.slice((modulus_size - 1) * coeff_count, modulus_size * coeff_count);
        utils::intt_inplace(c_last, coeff_count, rns_ntt_tables.at(modulus_size - 1));
        
        // neg_c_last_mod_t = - c_last (mod t)
        Array<uint64_t> neg_c_last_mod_t(coeff_count, device, pool);
        utils::modulo(c_last.as_const(), this->t(), neg_c_last_mod_t.reference());
        utils::negate_inplace(neg_c_last_mod_t.reference(), this->t());
        if (this->inv_q_last_mod_t() != 1) {
            // neg_c_last_mod_t *= q_last^(-1) (mod t)
            utils::multiply_scalar_inplace(neg_c_last_mod_t.reference(), this->inv_q_last_mod_t(), this->t());
        }

        mod_t_and_divide_q_last_ntt_inplace_step1(*this, input, neg_c_last_mod_t.const_reference(), rns_ntt_tables, pool);

    }

}}