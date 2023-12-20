#include "../test_adv.cuh"
#include "timer.cuh"

namespace bench {

    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;
    
    void test_encode_simd(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_encode = timer.register_timer(name + ".EncodeSimd");
        size_t timer_decode = timer.register_timer(name + ".DecodeSimd");
        for (size_t i = 0; i < repeat; i++) {
            GeneralVector message = context.random_simd_full();
            timer.tick(timer_encode);
            Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
            timer.tock(timer_encode);
            timer.tick(timer_decode);
            GeneralVector decoded = context.encoder().decode_simd(plain);
            timer.tock(timer_decode);
        }
        timer.print_divided(repeat);
    }

    void test_encode_polynomial(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_encode = timer.register_timer(name + ".EncodePoly");
        size_t timer_decode = timer.register_timer(name + ".DecodePoly");
        for (size_t i = 0; i < repeat; i++) {
            GeneralVector message = context.random_polynomial_full();
            timer.tick(timer_encode);
            Plaintext plain = context.encoder().encode_polynomial(message, std::nullopt, scale);
            timer.tock(timer_encode);
            timer.tick(timer_decode);
            GeneralVector decoded = context.encoder().decode_polynomial(plain);
            timer.tock(timer_decode);
        }
        timer.print_divided(repeat);
    }

    void test_encrypt(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_enc_asym = timer.register_timer(name + ".EncryptAsym");
        size_t timer_enc_sym = timer.register_timer(name + ".EncryptSym");
        size_t timer_dec = timer.register_timer(name + ".Decrypt");
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_enc_asym);
            Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
            timer.tock(timer_enc_asym);
            timer.tick(timer_enc_sym);
            Ciphertext cipher_sym = context.encryptor().encrypt_symmetric_new(plain, true);
            timer.tock(timer_enc_sym);
            timer.tick(timer_dec);
            Plaintext plain_dec = context.decryptor().decrypt_new(cipher);
            timer.tock(timer_dec);
        }
        timer.print_divided(repeat);
    }

    void test_negate(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_neg = timer.register_timer(name + ".Negate");
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        Ciphertext result; 
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_neg);
            Ciphertext cipher_neg = context.evaluator().negate_new(cipher);
            // context.evaluator().add(cipher, cipher, result);
            timer.tock(timer_neg);
        }
        timer.print_divided(repeat);
    }

    void test_add(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_add = timer.register_timer(name + ".Add");
        size_t timer_sub = timer.register_timer(name + ".Sub");
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        Ciphertext result; 
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_add);
            Ciphertext cipher_add = context.evaluator().add_new(cipher, cipher);
            // context.evaluator().add(cipher, cipher, result);
            timer.tock(timer_add);
            timer.tick(timer_sub);
            Ciphertext cipher_sub = context.evaluator().sub_new(cipher, cipher);
            timer.tock(timer_sub);
        }
        timer.print_divided(repeat);
    }

    void test_add_plain(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_add = timer.register_timer(name + ".AddPlain");
        size_t timer_sub = timer.register_timer(name + ".SubPlain");
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_add);
            Ciphertext cipher_add = context.evaluator().add_plain_new(cipher, plain);
            timer.tock(timer_add);
            timer.tick(timer_sub);
            Ciphertext cipher_sub = context.evaluator().sub_plain_new(cipher, plain);
            timer.tock(timer_sub);
        }
        timer.print_divided(repeat);
    }

    void test_multiply_relinearize(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_mul = timer.register_timer(name + ".Multiply");
        size_t timer_square = timer.register_timer(name + ".Square");
        size_t timer_relin = timer.register_timer(name + ".Relinearize");
        RelinKeys relin_keys = context.key_generator().create_relin_keys(false);
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_mul);
            Ciphertext cipher_mul = context.evaluator().multiply_new(cipher, cipher);
            timer.tock(timer_mul);
            timer.tick(timer_square);
            Ciphertext cipher_square = context.evaluator().square_new(cipher);
            timer.tock(timer_square);
            timer.tick(timer_relin);
            Ciphertext cipher_relin = context.evaluator().relinearize_new(cipher_mul, relin_keys);
            timer.tock(timer_relin);
        }
        timer.print_divided(repeat);
    }

    void test_multiply_plain(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_mul = timer.register_timer(name + ".MultiplyPlain");
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_mul);
            Ciphertext cipher_mul = context.evaluator().multiply_plain_new(cipher, plain);
            timer.tock(timer_mul);
        }
        timer.print_divided(repeat);
    }

    void test_mod_switch_to_next(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        double scale = context.scale();
        Timer timer;
        size_t timer_mod = timer.register_timer(name + ".ModSwitchToNext");
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        if (!context.context()->get_context_data(cipher.parms_id()).value()->next_context_data().has_value()) {
            return; // no next context so cannot test
        }
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_mod);
            Ciphertext cipher_mod = context.evaluator().mod_switch_to_next_new(cipher);
            timer.tock(timer_mod);
        }
        timer.print_divided(repeat);
    }

    void test_rescale_to_next(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        if (!is_ckks) return;
        double scale = context.scale();
        Timer timer;
        size_t timer_rescale = timer.register_timer(name + ".RescaleToNext");
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        if (!context.context()->get_context_data(cipher.parms_id()).value()->next_context_data().has_value()) {
            return; // no next context so cannot test
        }
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_rescale);
            Ciphertext cipher_rescale = context.evaluator().rescale_to_next_new(cipher);
            timer.tock(timer_rescale);
        }
        timer.print_divided(repeat);
    }

    void test_rotate_rows(const string& name, const GeneralHeContext& context, size_t rotate_count, size_t repeat = 100) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        if (is_ckks) return;
        double scale = context.scale();
        Timer timer;
        size_t timer_rotate = timer.register_timer(name + ".RotateRows(" + std::to_string(rotate_count) + ")");
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_rotate);
            Ciphertext cipher_rotate = context.evaluator().rotate_rows_new(cipher, rotate_count, glk);
            timer.tock(timer_rotate);
        }
        timer.print_divided(repeat);
    }

    void test_rotate_columns(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        if (is_ckks) return;
        double scale = context.scale();
        Timer timer;
        size_t timer_rotate = timer.register_timer(name + ".RotateColumns");
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_rotate);
            Ciphertext cipher_rotate = context.evaluator().rotate_columns_new(cipher, glk);
            timer.tock(timer_rotate);
        }
        timer.print_divided(repeat);
    }

    void test_rotate_vector(const string& name, const GeneralHeContext& context, size_t rotate_count, size_t repeat = 100) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        if (!is_ckks) return;
        double scale = context.scale();
        Timer timer;
        size_t timer_rotate = timer.register_timer(name + ".RotateVector(" + std::to_string(rotate_count) + ")");
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_rotate);
            Ciphertext cipher_rotate = context.evaluator().rotate_vector_new(cipher, rotate_count, glk);
            timer.tock(timer_rotate);
        }
        timer.print_divided(repeat);
    }

    void test_complex_conjugate(const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        if (!is_ckks) return;
        double scale = context.scale();
        Timer timer;
        size_t timer_conj = timer.register_timer(name + ".Conjugate");
        GaloisKeys glk = context.key_generator().create_galois_keys(false);
        GeneralVector message = context.random_simd_full();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_conj);
            Ciphertext cipher_conj = context.evaluator().complex_conjugate_new(cipher, glk);
            timer.tock(timer_conj);
        }
        timer.print_divided(repeat);
    }

    void test_suite(const string& name, const GeneralHeContext& context, size_t repeat_count) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        test_encode_simd(name, context, repeat_count);
        test_encode_polynomial(name, context, repeat_count);
        test_encrypt(name, context, repeat_count);
        test_negate(name, context, repeat_count);
        test_add(name, context, repeat_count);
        test_add_plain(name, context, repeat_count);
        test_multiply_relinearize(name, context, repeat_count);
        test_multiply_plain(name, context, repeat_count);
        test_mod_switch_to_next(name, context, repeat_count);
        if (is_ckks)
            test_rescale_to_next(name, context, repeat_count);
        if (!is_ckks) {
            test_rotate_rows(name, context, 1, repeat_count);
            test_rotate_rows(name, context, 7, repeat_count);
            test_rotate_columns(name, context, repeat_count);
        } else {
            test_rotate_vector(name, context, 1, repeat_count);
            test_rotate_vector(name, context, 7, repeat_count);
            test_complex_conjugate(name, context, repeat_count);
        }
    }

    void run_all(bool device) {
        size_t repeat = device ? 1000 : 100;

        std::cout << "[BFV]" << std::endl;
        GeneralHeContext bfv(device, SchemeType::BFV, 8192, 40, { 60, 40, 40, 60 }, true, 0x123);
        test_suite("BFV", bfv, repeat);

        std::cout << "[CKKS]" << std::endl;
        GeneralHeContext ckks(device, SchemeType::CKKS, 8192, 0, { 60, 40, 40, 60 }, true, 0x123, 10, 1ull<<20, 1e-2);
        test_suite("CKKS", ckks, repeat);

        std::cout << "[BGV]" << std::endl;
        GeneralHeContext bgv(device, SchemeType::BGV, 8192, 40, { 60, 40, 40, 60 }, true, 0x123);
        test_suite("BGV", bgv, repeat);
    }

}

int main(int argc, char** argv) {
    using std::cout;
    using std::endl;
    using std::string;
    bool host = false;
    bool device = false;
    // if no argument, both set to true
    // if one argument with "host" or "device", set that one to true
    // if one argument but not "host" or "device", print usage
    if (argc == 1) {
        host = true;
        device = true;
    } else if (argc == 2) {
        string arg = argv[1];
        if (arg == "host") {
            host = true;
        } else if (arg == "device") {
            device = true;
        } else {
            cout << "Usage: " << argv[0] << " [host|device]" << endl;
            return 1;
        }
    } else {
        cout << "Usage: " << argv[0] << " [host|device]" << endl;
        return 1;
    }
    if (host) {
        cout << "========= HOST =========" << endl;
        bench::run_all(false);
    }
    if (device) {
        cout << "======== DEVICE ========" << endl;
        bench::run_all(true);
        troy::utils::MemoryPool::Destroy();
    }
    return 0;
}