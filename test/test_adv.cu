#include "test_adv.cuh"

namespace tool {

    GeneralHeContext::GeneralHeContext(bool device, SchemeType scheme, size_t n, size_t log_t, vector<size_t> log_qi, 
        bool expand_mod_chain, uint64_t seed, double input_max, double scale, double tolerance,
        bool to_device_after_keygeneration, bool use_special_prime_for_encryption, MemoryPoolHandle pool)
    {
        this->pool_ = pool;
        this->scheme_ = scheme;
        if (scheme == SchemeType::CKKS && (input_max == 0 || scale == 0)) {
            throw std::invalid_argument("input_max and scale must be set for CKKS");
        }
        // create enc params
        EncryptionParameters parms(scheme);
        parms.set_poly_modulus_degree(n);
        // create modulus
        if (scheme != SchemeType::CKKS) {
            log_qi.push_back(log_t);
            auto moduli = CoeffModulus::create(n, log_qi);
            parms.set_plain_modulus(moduli[moduli.size() - 1]);
            parms.set_coeff_modulus(moduli.const_slice(0, moduli.size() - 1));
        } else {
            auto moduli = CoeffModulus::create(n, log_qi);
            parms.set_coeff_modulus(moduli);
        }
        parms.set_use_special_prime_for_encryption(use_special_prime_for_encryption);
        this->params_host_ = parms;
        // create gadgets
        bool ckks = scheme == SchemeType::CKKS;
        auto context = HeContext::create(parms, expand_mod_chain, SecurityLevel::Nil, seed);
        auto encoder = ckks ? new GeneralEncoder(CKKSEncoder(context)) : new GeneralEncoder(BatchEncoder(context));
        if (device && !to_device_after_keygeneration) { 
            context->to_device_inplace(pool);
            encoder->to_device_inplace(pool);
        }

        auto key_generator = new KeyGenerator(context, pool);
        auto public_key = key_generator->create_public_key(false, pool);
        auto encryptor = new Encryptor(context);
        encryptor->set_public_key(public_key);
        encryptor->set_secret_key(key_generator->secret_key());
        auto decryptor = new Decryptor(context, key_generator->secret_key(), pool);
        auto evaluator = new Evaluator(context);
        uint64_t t = ckks ? 0 : parms.plain_modulus()->value();
        
        if (device && to_device_after_keygeneration) { 
            context->to_device_inplace(pool);
            encoder->to_device_inplace(pool);
            key_generator->to_device_inplace(pool);
            encryptor->to_device_inplace(pool);
            decryptor->to_device_inplace(pool);
        }

        this->he_context_ = context;
        this->encoder_ = encoder;
        this->key_generator_ = key_generator;
        this->encryptor_ = encryptor;
        this->decryptor_ = decryptor;
        this->evaluator_ = evaluator;
        this->t_ = t;
        this->input_max_ = input_max;
        this->scale_ = scale;
        this->tolerance_ = tolerance;
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