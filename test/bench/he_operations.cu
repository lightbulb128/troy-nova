#include "../test_adv.cuh"
#include "../argparse.h"
#include "timer.cuh"

namespace bench {

    struct Arguments {
        bool help = false;
        bool host = false;
        bool device = false;
        bool scheme_bfv = false;
        bool scheme_ckks = false;
        bool scheme_bgv = false;
        size_t poly_modulus_degree = 8192;
        size_t log_t = 40;
        std::vector<size_t> log_q = { 60, 40, 40, 60 };
        uint64_t seed = 0x123;
        double input_max = 5;
        double scale = 1 << 30;
        double tolerance = 0.1;
        size_t repeat = 200;

        bool no_test_correct = false;
        bool no_bench_encode = false;
        bool no_bench_encrypt = false;
        bool no_bench_negate = false;
        bool no_bench_translate = false;
        bool no_bench_translate_plain = false;
        bool no_bench_multiply_relinearize = false;
        bool no_bench_multiply_plain = false;
        bool no_bench_mod_switch_to_next = false;
        bool no_bench_rescale_to_next = false;
        bool no_bench_rotate_rows = false;
        bool no_bench_rotate_columns = false;
        bool no_bench_rotate_vector = false;
        bool no_bench_complex_conjugate = false;

        Arguments(int argc, char** argv) {
            ArgumentParser parser(argc, argv);

            help = parser.get_bool_store_true("-h").value_or(parser.get_bool_store_true("--help").value_or(false));

            host = parser.get_bool_store_true("-H").value_or(parser.get_bool_store_true("--host").value_or(false));
            device = parser.get_bool_store_true("-D").value_or(parser.get_bool_store_true("--device").value_or(false));
            if (not host and not device) {
                host = true;
                device = true;
            }

            scheme_bfv = parser.get_bool_store_true("--bfv").value_or(false);
            scheme_ckks = parser.get_bool_store_true("--ckks").value_or(false);
            scheme_bgv = parser.get_bool_store_true("--bgv").value_or(false);
            if (not scheme_bfv and not scheme_ckks and not scheme_bgv) {
                scheme_bfv = true;
                scheme_ckks = true;
                scheme_bgv = true;
            }

            poly_modulus_degree = parser.get_uint<size_t>("-N").value_or(parser.get_uint<size_t>("--poly-modulus-degree").value_or(poly_modulus_degree));
            log_t = parser.get_uint<size_t>("-t").value_or(parser.get_uint<size_t>("--log-t").value_or(log_t));
            log_q = parser.get_uint_list<size_t>("-q").value_or(parser.get_uint_list<size_t>("--log-q").value_or(log_q));
            seed = parser.get_uint<uint64_t>("-S").value_or(parser.get_uint<uint64_t>("--seed").value_or(seed));
            input_max = parser.get_float<double>("-i").value_or(parser.get_float<double>("--input-max").value_or(input_max));
            scale = parser.get_float<double>("-s").value_or(parser.get_float<double>("--scale").value_or(scale));
            tolerance = parser.get_float<double>("-T").value_or(parser.get_float<double>("--tolerance").value_or(tolerance));
            repeat = parser.get_uint<size_t>("-R").value_or(parser.get_uint<size_t>("--repeat").value_or(repeat));

            no_test_correct = parser.get_bool_store_true("--no-test-correct").value_or(false);
            no_bench_encode = parser.get_bool_store_true("--no-bench-encode").value_or(false);
            no_bench_encrypt = parser.get_bool_store_true("--no-bench-encrypt").value_or(false);
            no_bench_negate = parser.get_bool_store_true("--no-bench-negate").value_or(false);
            no_bench_translate = parser.get_bool_store_true("--no-bench-translate").value_or(false);
            no_bench_translate_plain = parser.get_bool_store_true("--no-bench-translate-plain").value_or(false);
            no_bench_multiply_relinearize = parser.get_bool_store_true("--no-bench-multiply-relinearize").value_or(false);
            no_bench_multiply_plain = parser.get_bool_store_true("--no-bench-multiply-plain").value_or(false);
            no_bench_mod_switch_to_next = parser.get_bool_store_true("--no-bench-mod-switch-to-next").value_or(false);
            no_bench_rescale_to_next = parser.get_bool_store_true("--no-bench-rescale-to-next").value_or(false);
            no_bench_rotate_rows = parser.get_bool_store_true("--no-bench-rotate-rows").value_or(false);
            no_bench_rotate_columns = parser.get_bool_store_true("--no-bench-rotate-columns").value_or(false);
            no_bench_rotate_vector = parser.get_bool_store_true("--no-bench-rotate-vector").value_or(false);
            no_bench_complex_conjugate = parser.get_bool_store_true("--no-bench-complex-conjugate").value_or(false);

        }

        static void print_help() {
            std::cout << "Usage: troybench [options]" << std::endl;
            std::cout << "Run benchmark for HE operations" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << std::endl;
            std::cout << "  -H, --host                  Run on host" << std::endl;
            std::cout << "  -D, --device                Run on device" << std::endl;
            std::cout << "      If neither of -H and -D are set, run on both host and device" << std::endl;
            std::cout << std::endl;
            std::cout << "  --bfv                       Run BFV scheme" << std::endl;
            std::cout << "  --ckks                      Run CKKS scheme" << std::endl;
            std::cout << "  --bgv                       Run BGV scheme" << std::endl;
            std::cout << "      If none of --bfv, --ckks, and --bgv are set, run all schemes" << std::endl;
            std::cout << std::endl;
            std::cout << "  -N, --poly-modulus-degree   Poly modulus degree (default: 8192)" << std::endl;
            std::cout << "  -t, --log-t                 Log t (default: 40)" << std::endl;
            std::cout << "  -q, --log-q                 Log q (default: 60,40,40,60)" << std::endl;
            std::cout << "  -S, --seed                  Seed (default: 0x123)" << std::endl;
            std::cout << "  -i, --input-max             Input max (default: 10)" << std::endl;
            std::cout << "  -s, --scale                 Scale (default: 1 << 20)" << std::endl;
            std::cout << "  -T, --tolerance             Tolerance (default: 1e-2)" << std::endl;
            std::cout << "  -R, --repeat                Repeat count (default: 200)" << std::endl;

            std::cout << std::endl;
            std::cout << "  --no-test-correct           Skip correctness test" << std::endl;
            // no explain
            std::cout << "  --no-bench-encode" << std::endl;
            std::cout << "  --no-bench-encrypt" << std::endl;
            std::cout << "  --no-bench-negate" << std::endl;
            std::cout << "  --no-bench-translate" << std::endl;
            std::cout << "  --no-bench-translate-plain" << std::endl;
            std::cout << "  --no-bench-multiply-relinearize" << std::endl;
            std::cout << "  --no-bench-multiply-plain" << std::endl;
            std::cout << "  --no-bench-mod-switch-to-next" << std::endl;
            std::cout << "  --no-bench-rescale-to-next" << std::endl;
            std::cout << "  --no-bench-rotate-rows" << std::endl;
            std::cout << "  --no-bench-rotate-columns" << std::endl;
            std::cout << "  --no-bench-rotate-vector" << std::endl;
            std::cout << "  --no-bench-complex-conjugate" << std::endl;

        }
    };

    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;

    void assert_true(bool condition, const string& message) {
        if (!condition) {
            std::cerr << message << std::endl;
            exit(1);
        }
    }
    
    void test_encode_simd(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            assert_true(context.near_equal(message, decoded), "test_encode_simd failed.");
        }
        timer.print_divided(repeat);
    }

    void test_encode_polynomial(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                assert_true(context.near_equal(message, decoded), "test_encode_polynomial failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_encrypt(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            
            if (i == 0 && args.no_test_correct) {
                auto decoded = context.encoder().decode_simd(plain_dec);
                assert_true(context.near_equal(message, decoded), "test_encrypt/asymmetric failed.");
                cipher_sym.expand_seed(context.context());
                decoded = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_sym));
                assert_true(context.near_equal(message, decoded), "test_encrypt/symmetric failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_negate(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_neg));
                assert_true(context.near_equal(context.negate(message), decrypted), "test_negate failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_translate(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_add));
                assert_true(context.near_equal(context.add(message, message), decrypted), "test_translate/add failed.");
                decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_sub));
                assert_true(context.near_equal(context.sub(message, message), decrypted), "test_translate/sub failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_translate_plain(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_add));
                assert_true(context.near_equal(context.add(message, message), decrypted), "test_translate_plain/add failed.");
                decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_sub));
                assert_true(context.near_equal(context.sub(message, message), decrypted), "test_translate_plain/sub failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_multiply_relinearize(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto truth = context.mul(message, message);
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_mul));
                assert_true(context.near_equal(truth, decrypted), "test_multiply_relinearize/multiply failed.");
                decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_relin));
                assert_true(context.near_equal(truth, decrypted), "test_multiply_relinearize/relin failed.");
                decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_square));
                assert_true(context.near_equal(truth, decrypted), "test_multiply_relinearize/square failed.");
            }

        }
        timer.print_divided(repeat);
    }

    void test_multiply_plain(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto truth = context.mul(message, message);
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_mul));
                assert_true(context.near_equal(truth, decrypted), "test_multiply_plain failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_mod_switch_to_next(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_mod));
                assert_true(context.near_equal(message, decrypted), "test_mod_switch_to_next failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_rescale_to_next(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        if (!is_ckks) return;
        double scale = context.scale();
        Timer timer;
        size_t timer_rescale = timer.register_timer(name + ".RescaleToNext");
        GeneralVector message = context.random_simd_full();
        const EncryptionParameters& parms = context.params_host();
        auto coeff_modulus = parms.coeff_modulus();
        double expanded_scale = scale * coeff_modulus[coeff_modulus.size() - 2].value();
        Plaintext plain = context.encoder().encode_simd(message, std::nullopt, expanded_scale);
        Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain);
        if (!context.context()->get_context_data(cipher.parms_id()).value()->next_context_data().has_value()) {
            return; // no next context so cannot test
        }
        for (size_t i = 0; i < repeat; i++) {
            timer.tick(timer_rescale);
            Ciphertext cipher_rescale = context.evaluator().rescale_to_next_new(cipher);
            timer.tock(timer_rescale);
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_rescale));
                assert_true(context.near_equal(message, decrypted), "test_rescale_to_next failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_rotate_rows(const Arguments& args, const string& name, const GeneralHeContext& context, size_t rotate_count, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_rotate));
                assert_true(context.near_equal(message.rotate(rotate_count), decrypted), "test_rotate_rows failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_rotate_columns(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_rotate));
                assert_true(context.near_equal(message.conjugate(), decrypted), "test_rotate_columns failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_rotate_vector(const Arguments& args, const string& name, const GeneralHeContext& context, size_t rotate_count, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_rotate));
                assert_true(context.near_equal(message.rotate(rotate_count), decrypted), "test_rotate_vector failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_complex_conjugate(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat = 100) {
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
            if (i == 0 && args.no_test_correct) {
                auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_conj));
                assert_true(context.near_equal(message.conjugate(), decrypted), "test_complex_conjugate failed.");
            }
        }
        timer.print_divided(repeat);
    }

    void test_suite(const Arguments& args, const string& name, const GeneralHeContext& context, size_t repeat_count) {
        bool is_ckks = context.params_host().scheme() == SchemeType::CKKS;
        if (!args.no_bench_encode) test_encode_simd(args, name, context, repeat_count);
        if (!args.no_bench_encode) test_encode_polynomial(args, name, context, repeat_count);
        if (!args.no_bench_encrypt) test_encrypt(args, name, context, repeat_count);
        if (!args.no_bench_negate) test_negate(args, name, context, repeat_count);
        if (!args.no_bench_translate) test_translate(args, name, context, repeat_count);
        if (!args.no_bench_translate_plain) test_translate_plain(args, name, context, repeat_count);
        if (!args.no_bench_multiply_relinearize) test_multiply_relinearize(args, name, context, repeat_count);
        if (!args.no_bench_multiply_plain) test_multiply_plain(args, name, context, repeat_count);
        if (!args.no_bench_mod_switch_to_next) test_mod_switch_to_next(args, name, context, repeat_count);
        if (is_ckks)
            if (!args.no_bench_rescale_to_next) test_rescale_to_next(args, name, context, repeat_count);
        if (!is_ckks) {
            if (!args.no_bench_rotate_rows) test_rotate_rows(args, name, context, 1, repeat_count);
            if (!args.no_bench_rotate_rows) test_rotate_rows(args, name, context, 7, repeat_count);
            if (!args.no_bench_rotate_columns) test_rotate_columns(args, name, context, repeat_count);
        } else {
            if (!args.no_bench_rotate_vector) test_rotate_vector(args, name, context, 1, repeat_count);
            if (!args.no_bench_rotate_vector) test_rotate_vector(args, name, context, 7, repeat_count);
            if (!args.no_bench_complex_conjugate) test_complex_conjugate(args, name, context, repeat_count);
        }
    }

    void run_all(bool device, const Arguments& args) {
        size_t repeat = args.repeat;

        if (args.scheme_bfv) {
            std::cout << "[BFV]" << std::endl;
            GeneralHeContext bfv(device, SchemeType::BFV, args.poly_modulus_degree, args.log_t, args.log_q, true, args.seed);
            test_suite(args, "BFV", bfv, repeat);
        }

        if (args.scheme_ckks) {
            std::cout << "[CKKS]" << std::endl;
            GeneralHeContext ckks(device, SchemeType::CKKS, args.poly_modulus_degree, args.log_t, args.log_q, true, args.seed, args.input_max, args.scale, args.tolerance);
            test_suite(args, "CKKS", ckks, repeat);
        }

        if (args.scheme_bgv) {
            std::cout << "[BGV]" << std::endl;
            GeneralHeContext bgv(device, SchemeType::BGV, args.poly_modulus_degree, args.log_t, args.log_q, true, args.seed);
            test_suite(args, "BGV", bgv, repeat);
        }
    }

}

int main(int argc, char** argv) {
    using std::cout;
    using std::endl;
    using std::string;
    using bench::Arguments;
    Arguments args(argc, argv);
    if (args.help) {
        Arguments::print_help();
        return 0;
    } 

    if (args.host) {
        cout << "========= HOST =========" << endl;
        bench::run_all(false, args);
    }
    if (args.device) {
        cout << "======== DEVICE ========" << endl;
        bench::run_all(true, args);
        troy::utils::MemoryPool::Destroy();
    }
    return 0;
}