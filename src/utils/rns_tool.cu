#include "rns_tool.cuh"
namespace troy {namespace utils {

    
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
        
        // Populate the base arrays
        RNSBase base_q = q.clone();
        RNSBase base_B(ConstSlice(base_B_primes.data(), base_B_primes.size(), false));
        RNSBase base_Bsk = base_B.extend_modulus(m_sk);
        RNSBase base_Bsk_m_tilde = base_Bsk.extend_modulus(m_tilde);

        // Set up t-gamma base if t_ is non-zero (using BFV)
        std::optional<RNSBase> base_t_gamma = std::nullopt;
        std::optional<BaseConverter> base_q_to_t_conv = std::nullopt;
        if (!t.is_zero()) {
            Modulus t_gamma[2]{ t, gamma };
            base_t_gamma = std::optional(RNSBase(ConstSlice(t_gamma, 2, false)));
            base_q_to_t_conv = std::optional(BaseConverter(base_q, RNSBase(ConstSlice(&t, 1, false))));
        }
        
        // Generate the Bsk NTTTables; these are used for NTT after base extension to Bsk
        Array<NTTTables> base_Bsk_ntt_tables = NTTTables::create_ntt_tables(
            coeff_count_power,
            base_Bsk.base()
        );

        BaseConverter base_q_to_Bsk_conv = BaseConverter(base_q, base_Bsk);
        BaseConverter base_q_to_m_tilde_conv = BaseConverter(base_q, RNSBase(ConstSlice(&m_tilde, 1, false)));
        BaseConverter base_B_to_q_conv = BaseConverter(base_B, base_q);
        BaseConverter base_B_to_m_sk_conv = BaseConverter(base_B, RNSBase(ConstSlice(&m_sk, 1, false)));
        std::optional<BaseConverter> base_q_to_t_gamma_conv = std::nullopt;
        if (base_q_to_t_conv.has_value()) {
            base_q_to_t_gamma_conv = std::optional(BaseConverter(base_q, base_t_gamma.value()));
        }

        // Compute prod(B) mod q
        Array<uint64_t> prod_B_mod_q(base_q.size(), false);
        for (size_t i = 0; i < base_q.size(); i++) {
            prod_B_mod_q[i] = utils::modulo_uint(base_B.base_product(), base_q.base()[i]);
        }

        // Compute prod(q)^(-1) mod Bsk
        Array<MultiplyUint64Operand> inv_prod_q_mod_Bsk(base_Bsk.size(), false);
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
        Array<MultiplyUint64Operand> inv_m_tilde_mod_Bsk(base_Bsk.size(), false);
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

        Array<uint64_t> prod_q_mod_Bsk(base_Bsk.size(), false);
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
            prod_t_gamma_mod_q = std::optional(Array<MultiplyUint64Operand>(base_q.size(), false));
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
            neg_inv_q_mod_t_gamma = std::optional(Array<MultiplyUint64Operand>(2, false));
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
        Array<MultiplyUint64Operand> inv_q_last_mod_q(base_q.size() - 1, false);
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

        this->m_tilde_ = Box(std::move(m_tilde));
        this->m_sk_ = Box(std::move(m_sk));
        this->t_ = Box(Modulus(t));
        this->gamma_ = Box(std::move(gamma));

        this->inv_q_last_mod_t_ = inv_q_last_mod_t;
        this->q_last_mod_t_ = q_last_mod_t;
        this->q_last_half_ = last_q.value() >> 1;

        this->device = false;
        
    }

    template <typename T>
    static std::optional<T> optional_clone(const std::optional<T>& opt) {
        if (opt.has_value()) {
            return std::optional<T>(opt.value().clone());
        }
        return std::nullopt;
    }

    RNSTool RNSTool::clone() const {
        RNSTool cloned;

        cloned.coeff_count_ = this->coeff_count_;

        cloned.base_q_ = this->base_q_.clone();
        cloned.base_B_ = this->base_B_.clone();
        cloned.base_Bsk_ = this->base_Bsk_.clone();
        cloned.base_Bsk_m_tilde_ = this->base_Bsk_m_tilde_.clone();
        cloned.base_t_gamma_ = optional_clone(this->base_t_gamma_);

        cloned.base_q_to_Bsk_conv_ = this->base_q_to_Bsk_conv_.clone();
        cloned.base_q_to_m_tilde_conv_ = this->base_q_to_m_tilde_conv_.clone();
        cloned.base_B_to_q_conv_ = this->base_B_to_q_conv_.clone();
        cloned.base_B_to_m_sk_conv_ = this->base_B_to_m_sk_conv_.clone();
        cloned.base_q_to_t_gamma_conv_ = optional_clone(this->base_q_to_t_gamma_conv_);
        cloned.base_q_to_t_conv_ = optional_clone(this->base_q_to_t_conv_);

        cloned.inv_prod_q_mod_Bsk_ = this->inv_prod_q_mod_Bsk_.clone();
        cloned.neg_inv_prod_q_mod_m_tilde_ = this->neg_inv_prod_q_mod_m_tilde_;
        cloned.inv_prod_B_mod_m_sk_ = this->inv_prod_B_mod_m_sk_;
        cloned.inv_gamma_mod_t_ = this->inv_gamma_mod_t_;
        cloned.prod_B_mod_q_ = this->prod_B_mod_q_.clone();
        cloned.inv_m_tilde_mod_Bsk_ = this->inv_m_tilde_mod_Bsk_.clone();
        cloned.prod_q_mod_Bsk_ = this->prod_q_mod_Bsk_.clone();
        cloned.neg_inv_q_mod_t_gamma_ = optional_clone(this->neg_inv_q_mod_t_gamma_);
        cloned.prod_t_gamma_mod_q_ = optional_clone(this->prod_t_gamma_mod_q_);
        cloned.inv_q_last_mod_q_ = this->inv_q_last_mod_q_.clone();
        cloned.base_Bsk_ntt_tables_ = this->base_Bsk_ntt_tables_.clone();
        cloned.m_tilde_ = this->m_tilde_.clone();
        cloned.m_sk_ = this->m_sk_.clone();

        cloned.t_ = this->t_.clone();
        cloned.gamma_ = this->gamma_.clone();
        cloned.inv_q_last_mod_t_ = this->inv_q_last_mod_t_;
        cloned.q_last_mod_t_ = this->q_last_mod_t_;
        cloned.q_last_half_ = this->q_last_half_;

        cloned.device = this->device;

        return cloned;
    }

    template <typename T>
    static void optional_to_device_inplace(std::optional<T>& opt) {
        if (opt.has_value()) {
            opt.value().to_device_inplace();
        }
    }

    void RNSTool::to_device_inplace() {
        if (this->on_device()) {
            return;
        }
        
        this->base_q_.to_device_inplace();
        this->base_B_.to_device_inplace();
        this->base_Bsk_.to_device_inplace();
        this->base_Bsk_m_tilde_.to_device_inplace();
        optional_to_device_inplace(this->base_t_gamma_);

        this->base_q_to_Bsk_conv_.to_device_inplace();
        this->base_q_to_m_tilde_conv_.to_device_inplace();
        this->base_B_to_q_conv_.to_device_inplace();
        this->base_B_to_m_sk_conv_.to_device_inplace();
        optional_to_device_inplace(this->base_q_to_t_gamma_conv_);
        optional_to_device_inplace(this->base_q_to_t_conv_);

        this->inv_prod_q_mod_Bsk_.to_device_inplace();
        this->prod_B_mod_q_.to_device_inplace();
        this->inv_m_tilde_mod_Bsk_.to_device_inplace();
        this->prod_q_mod_Bsk_.to_device_inplace();
        optional_to_device_inplace(this->neg_inv_q_mod_t_gamma_);
        optional_to_device_inplace(this->prod_t_gamma_mod_q_);
        this->inv_q_last_mod_q_.to_device_inplace();
        this->base_Bsk_ntt_tables_.to_device_inplace();

        this->m_tilde_.to_device_inplace();
        this->m_sk_.to_device_inplace();
        this->t_.to_device_inplace();
        this->gamma_.to_device_inplace();

        this->device = true;
    }

    static void host_divide_and_round_q_last_inplace(const RNSTool& self, Slice<uint64_t> input) {
        
        size_t base_q_size = self.base_q().size();
        ConstPointer<Modulus> last_modulus = self.base_q().base().at(base_q_size - 1);
        size_t coeff_count = self.coeff_count();
        size_t last_input_offset = (base_q_size - 1) * coeff_count;
        size_t half = self.q_last_half();

        Slice<uint64_t> input_last = input.slice(last_input_offset, last_input_offset + coeff_count);
        // Add (qi-1)/2 to change from flooring to rounding
        utils::add_scalar_inplace(input_last, half, last_modulus);
        Array<uint64_t> temp(coeff_count, false);
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
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= coeff_count) return;
        size_t base_q_size = base_q.size();
        const Modulus& last_modulus = *base_q.at(base_q_size - 1);
        size_t last_input_offset = (base_q.size() - 1) * coeff_count;
        Slice<uint64_t> input_last = input.slice(last_input_offset, last_input_offset + coeff_count);
        // Add (qi-1)/2 to change from flooring to rounding
        input_last[j] = utils::add_uint64_mod(input_last[j], q_last_half, last_modulus);
        for (size_t i = 0; i < base_q_size; i++) {
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
    }

    void RNSTool::divide_and_round_q_last_inplace(Slice<uint64_t> input) const {
        bool device = this->on_device();
        if (device != input.on_device()) {
            throw std::invalid_argument("[RNSTool::divide_and_round_q_last_inplace] RNSTool and input must be on the same device.");
        }
        if (device) {
            size_t block_count = utils::ceil_div(this->coeff_count(), utils::KERNEL_THREAD_COUNT);
            kernel_divide_and_round_q_last_inplace<<<block_count, utils::KERNEL_THREAD_COUNT>>>(
                this->base_q().base(),
                this->coeff_count(),
                this->q_last_half(),
                this->inv_q_last_mod_q(),
                input
            );
        } else {
            host_divide_and_round_q_last_inplace(*this, input);
        }
    }

}}