#include "test_adv.h"

namespace tool {

    GeneralHeContext::GeneralHeContext(GeneralHeContextParameters args) {
        this->pool_ = args.pool;
        this->scheme_ = args.scheme;
        if (args.scheme == SchemeType::CKKS && (args.input_max == 0 || args.scale == 0)) {
            throw std::invalid_argument("input_max and scale must be set for CKKS");
        }
        // create enc params
        EncryptionParameters parms(args.scheme);
        parms.set_poly_modulus_degree(args.n);
        // create modulus
        if (args.ring2k_log_t != 0) args.simd_log_t = 20;
        if (args.scheme != SchemeType::CKKS) {
            if (args.simd_log_t == 0) args.simd_log_t = 20;
            args.log_qi.push_back(args.simd_log_t);
            auto moduli = CoeffModulus::create(args.n, args.log_qi);
            parms.set_plain_modulus(moduli[moduli.size() - 1]);
            parms.set_coeff_modulus(moduli.const_slice(0, moduli.size() - 1));
        } else {
            auto moduli = CoeffModulus::create(args.n, args.log_qi);
            parms.set_coeff_modulus(moduli);
        }
        parms.set_use_special_prime_for_encryption(args.use_special_prime_for_encryption);
        this->params_host_ = parms;
        // create gadgets
        bool ckks = args.scheme == SchemeType::CKKS;
        auto context = HeContext::create(parms, args.expand_mod_chain, SecurityLevel::Nil, args.seed);
        GeneralEncoder* encoder;
        if (ckks) encoder = new GeneralEncoder(CKKSEncoder(context));
        else if (args.ring2k_log_t != 0) {
            if (args.ring2k_log_t > 64) encoder = new GeneralEncoder(PolynomialEncoderRing2k<uint128_t>(context, args.ring2k_log_t));
            else if (args.ring2k_log_t > 32) encoder = new GeneralEncoder(PolynomialEncoderRing2k<uint64_t>(context, args.ring2k_log_t));
            else encoder = new GeneralEncoder(PolynomialEncoderRing2k<uint32_t>(context, args.ring2k_log_t));
        } else {
            encoder = new GeneralEncoder(BatchEncoder(context));
        }

        if (args.device && !args.to_device_after_keygeneration) { 
            context->to_device_inplace(args.pool);
            encoder->to_device_inplace(args.pool);
        }

        auto key_generator = new KeyGenerator(context, args.pool);
        auto public_key = key_generator->create_public_key(false, args.pool);
        auto encryptor = new Encryptor(context);
        encryptor->set_public_key(public_key, args.pool);
        encryptor->set_secret_key(key_generator->secret_key(), args.pool);
        auto decryptor = new Decryptor(context, key_generator->secret_key(), args.pool);
        auto evaluator = new Evaluator(context);
        uint64_t t = ckks ? 0 : parms.plain_modulus()->value();
        this->ring_mask_ = (args.ring2k_log_t == 128) ? (static_cast<uint128_t>(-1)) : ((static_cast<uint128_t>(1) << args.ring2k_log_t) - 1);
        
        if (args.device && args.to_device_after_keygeneration) { 
            context->to_device_inplace(args.pool);
            encoder->to_device_inplace(args.pool);
            key_generator->to_device_inplace(args.pool);
            encryptor->to_device_inplace(args.pool);
            decryptor->to_device_inplace(args.pool);
        }

        this->he_context_ = context;
        this->encoder_ = encoder;
        this->key_generator_ = key_generator;
        this->encryptor_ = encryptor;
        this->decryptor_ = decryptor;
        this->evaluator_ = evaluator;
        this->t_ = t;
        this->input_max_ = args.input_max;
        this->scale_ = args.scale;
        this->tolerance_ = args.tolerance;
    }

    GeneralHeContext::~GeneralHeContext() {
        if (this->encoder_) {
            delete this->encoder_;
            delete this->key_generator_;
            delete this->encryptor_;
            delete this->decryptor_;
            delete this->evaluator_;
        }
    }

    GeneralHeContext::GeneralHeContext(GeneralHeContext&& from) {
        this->pool_ = from.pool_;
        this->scheme_ = from.scheme_;
        this->params_host_ = from.params_host_;
        this->he_context_ = from.he_context_;
        this->encoder_ = from.encoder_;
        this->key_generator_ = from.key_generator_;
        this->encryptor_ = from.encryptor_;
        this->decryptor_ = from.decryptor_;
        this->evaluator_ = from.evaluator_;
        this->t_ = from.t_;
        this->input_max_ = from.input_max_;
        this->scale_ = from.scale_;
        this->tolerance_ = from.tolerance_;

        from.pool_ = nullptr;
        from.he_context_ = nullptr;
        from.encoder_ = nullptr;
        from.key_generator_ = nullptr;
        from.encryptor_ = nullptr;
        from.decryptor_ = nullptr;
        from.evaluator_ = nullptr;
    }

}