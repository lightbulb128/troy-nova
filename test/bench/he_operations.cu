#include "../test_multithread.h"
#include "../argparse.h"
#include "../../src/utils/timer.h"
#include "../../src/batch_utils.h"
#include "argument_helper.h"
#include <iostream>
#include <thread>
#include <future>

namespace bench::he_operations {

    using namespace troy;
    using namespace tool;

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
        size_t warm_up_repeat = 10;
        size_t batch_size = 1;
        size_t threads = 1;
        bool multiple_pools = false;
        bool multiple_devices = false;

        bool no_test_correct = false;
        bool no_bench_encode_simd = false;
        bool no_bench_encode_polynomial = false;
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

        Arguments(const Arguments&) = default;

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
            warm_up_repeat = parser.get_uint<size_t>("-W").value_or(parser.get_uint<size_t>("--warm-up-repeat").value_or(warm_up_repeat));
            batch_size = parser.get_uint<size_t>("-B").value_or(parser.get_uint<size_t>("--batch-size").value_or(batch_size));
            threads = parser.get_uint<size_t>("-c").value_or(parser.get_uint<size_t>("--threads").value_or(threads));

            multiple_pools = parser.get_bool_store_true("-mp").value_or(parser.get_bool_store_true("--multiple-pools").value_or(false));
            multiple_devices = parser.get_bool_store_true("-md").value_or(parser.get_bool_store_true("--multiple-devices").value_or(false));

            no_test_correct = parser.get_bool_store_true("--no-test-correct").value_or(false);
            no_bench_encode_simd = parser.get_bool_store_true("--no-bench-encode-simd").value_or(false);
            no_bench_encode_polynomial = parser.get_bool_store_true("--no-bench-encode-polynomial").value_or(false);
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

            if (batch_size > 1) {
                bool flag = false;
                std::stringstream ss;
                if (!no_bench_encode_polynomial) {
                    no_bench_encode_polynomial = true;
                    ss << "  --no-bench-encode-polynomial\n";
                    flag = true;
                }
                if (!no_bench_translate_plain) {
                    no_bench_translate_plain = true;
                    ss << "  --no-bench-translate-plain\n";
                    flag = true;
                }
                if (!no_bench_multiply_relinearize) {
                    no_bench_multiply_relinearize = true;
                    ss << "  --no-bench-multiply-relinearize\n";
                    flag = true;
                }
                if (!no_bench_rescale_to_next) {
                    no_bench_rescale_to_next = true;
                    ss << "  --no-bench-rescale-to-next\n";
                    flag = true;
                }
                if (flag) {
                    std::cout << "Batch size is greater than 1. Some tests are disabled because those batch operations are not implemented yet:\n";
                    std::cout << ss.str();
                }
            }

            if (threads < 1) {
                throw std::invalid_argument("threads must be at least 1");
            }
            if (threads == 1) {
                if (multiple_pools || multiple_devices) {
                    std::cout << "Warning: multiple-pools and multiple-devices require more than 1 thread. Setting both to false." << std::endl;
                    multiple_pools = false;
                    multiple_devices = false;
                }
            }
            if (!multiple_pools && multiple_devices) {
                std::cout << "Warning: multiple-devices requires multiple-pools. Setting multiple-pools to true." << std::endl;
                multiple_pools = true;
            }
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
            std::cout << "  -W, --warm-up-repeat        Warm-up repeat count (default: 10)" << std::endl;
            std::cout << "  -B, --batch-size            Batch size (default: 1)" << std::endl;

            std::cout << std::endl;
            std::cout << "  -c, --threads               Number of threads (default: 1)" << std::endl;
            std::cout << "  -mp, --multiple-pools       Use multiple memory pools. Only meaningful for device test." << std::endl;
            std::cout << "  -md, --multiple-devices     Use multiple devices.  Only meaningful for device test." << std::endl;

            std::cout << std::endl;
            std::cout << "  --no-test-correct           Skip correctness test" << std::endl;
            // no explain
            std::cout << "  --no-bench-encode-simd" << std::endl;
            std::cout << "  --no-bench-encode-polynomial" << std::endl;
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

        void print_arguments() {
            std::cout << "[Arguments]" << std::endl;
            
            std::vector<std::string> run_on; 
            if (host) run_on.push_back("host");
            if (device) run_on.push_back("device");
            std::cout << "  run-on              = " << concat_by_comma(run_on) << std::endl;

            std::cout << "  threads             = " << threads << std::endl;
        
            if (device) {
                std::cout << "  multiple-pools      = " << bool_to_string(multiple_pools) << std::endl;
                std::cout << "  multiple-devices    = " << bool_to_string(multiple_devices) << std::endl;
                if (multiple_devices) {
                    int device_count = 0;
                    cudaError_t success = cudaGetDeviceCount(&device_count);
                    if (success != cudaSuccess) {
                        throw std::runtime_error("cudaGetDeviceCount failed");
                    }
                    std::cout << "  device-count        = " << device_count << std::endl;
                }
            }

            std::vector<std::string> schemes;
            if (scheme_bfv) schemes.push_back("BFV");
            if (scheme_ckks) schemes.push_back("CKKS");
            if (scheme_bgv) schemes.push_back("BGV");
            std::cout << "  schemes             = " << concat_by_comma(schemes) << std::endl;

            std::cout << "  poly-modulus-degree = " << poly_modulus_degree << std::endl;
            std::cout << "  log-t               = " << log_t << std::endl;
            std::cout << "  log-q               = " << list_usize_to_string(log_q) << std::endl;
            std::cout << "  seed                = " << "0x" << std::hex << seed << std::dec << std::endl;
            if (scheme_ckks) {
                std::cout << "  input-max           = " << input_max << std::endl;
                std::cout << "  scale               = " << scale << std::endl;
                std::cout << "  tolerance           = " << tolerance << std::endl;
            }
            std::cout << "  repeat              = " << repeat << std::endl;
            std::cout << "  warm-up-repeat      = " << warm_up_repeat << std::endl;
            std::cout << "  batch-size          = " << batch_size << std::endl;
            std::cout << "  no-test-correct     = " << bool_to_string(no_test_correct) << std::endl;
        }
    };

    using namespace troy;
    using tool::GeneralEncoder;
    using tool::GeneralVector;
    using tool::GeneralHeContext;
    using troy::bench::Timer;
    using troy::bench::TimerThreaded;
    using std::string;

    void assert_true(bool condition, const string& message) {
        if (!condition) {
            std::cerr << message << std::endl;
            exit(1);
        }
    }

    class Benchmark {

        private:
            string name;
            size_t repeat;
            size_t warm_up_repeat;
            size_t batch_size;
            Arguments args;
            SchemeType scheme;
            bool device;
            MultithreadHeContext environment;

        public:

            Benchmark(bool device, const string& name, SchemeType scheme, const Arguments& args):
                name(name), repeat(args.repeat), warm_up_repeat(args.warm_up_repeat), batch_size(args.batch_size), args(args), scheme(scheme), device(device)
            {
                GeneralHeContextParameters ghep(
                    device, scheme, args.poly_modulus_degree, args.log_t, args.log_q, true, args.seed, args.input_max,
                    args.scale, args.tolerance, false, false
                );
                environment = MultithreadHeContext(args.threads, args.multiple_pools, args.multiple_devices, ghep);
            }

            MemoryPoolHandle get_pool(size_t thread_id) const {
                return environment.get_pool(thread_id);
            }

            const GeneralHeContext& get_context(size_t thread_id) const {
                return environment.get_context(thread_id);
            }

            size_t get_repeat(size_t thread_id) const {
                return environment.get_divided(repeat, thread_id);
            }

            vector<GeneralVector> batch_random_simd_full(const GeneralHeContext& context) {
                vector<GeneralVector> batch;
                for (size_t i = 0; i < batch_size; i++) {
                    batch.push_back(context.random_simd_full());
                }
                return batch;
            }

            vector<GeneralVector> batch_random_polynomial_full(const GeneralHeContext& context) {
                vector<GeneralVector> batch;
                for (size_t i = 0; i < batch_size; i++) {
                    batch.push_back(context.random_polynomial_full());
                }
                return batch;
            }

            vector<Plaintext> batch_encode_simd(const GeneralHeContext& context, const vector<GeneralVector>& batch, double scale, MemoryPoolHandle pool) {
                vector<Plaintext> encoded;
                for (size_t i = 0; i < batch_size; i++) {
                    encoded.push_back(context.encoder().encode_simd(batch[i], std::nullopt, scale, pool));
                }
                return encoded;
            }

            vector<Plaintext> batch_encode_polynomial(const GeneralHeContext& context, const vector<GeneralVector>& batch, double scale, MemoryPoolHandle pool) {
                vector<Plaintext> encoded;
                for (size_t i = 0; i < batch_size; i++) {
                    encoded.push_back(context.encoder().encode_polynomial(batch[i], std::nullopt, scale, pool));
                }
                return encoded;
            }

            vector<Ciphertext> batch_encrypt_asymmetric(const GeneralHeContext& context, const vector<Plaintext>& batch, MemoryPoolHandle pool) {
                vector<Ciphertext> encrypted;
                for (size_t i = 0; i < batch_size; i++) {
                    encrypted.push_back(context.encryptor().encrypt_asymmetric_new(batch[i], nullptr, pool));
                }
                return encrypted;
            }

            vector<Plaintext> batch_decrypt(const GeneralHeContext& context, const vector<Ciphertext>& batch, MemoryPoolHandle pool) {
                vector<Plaintext> decrypted;
                for (size_t i = 0; i < batch_size; i++) {
                    decrypted.push_back(context.decryptor().decrypt_new(batch[i], pool));
                }
                return decrypted;
            }

            vector<GeneralVector> batch_decode_simd(const GeneralHeContext& context, const vector<Plaintext>& batch, MemoryPoolHandle pool) {
                vector<GeneralVector> decoded;
                for (size_t i = 0; i < batch_size; i++) {
                    decoded.push_back(context.encoder().decode_simd(batch[i], pool));
                }
                return decoded;
            }

            vector<GeneralVector> batch_decode_polynomial(const GeneralHeContext& context, const vector<Plaintext>& batch, MemoryPoolHandle pool) {
                vector<GeneralVector> decoded;
                for (size_t i = 0; i < batch_size; i++) {
                    decoded.push_back(context.encoder().decode_polynomial(batch[i], pool));
                }
                return decoded;
            }



            template <typename L>
            void run_threads_and_print_times(const L& lambda) {
                if (args.threads == 1) {
                    Timer timer_single = lambda(0);
                    timer_single.print_divided(repeat * batch_size);
                } else {
                    std::vector<std::future<Timer>> futures;
                    for (size_t i = 0; i < args.threads; i++) {
                        futures.push_back(std::async(lambda, i));
                    }
                    std::vector<Timer> timers;
                    for (size_t i = 0; i < args.threads; i++) {
                        timers.push_back(futures[i].get());
                    }
                    TimerThreaded::PrintDivided(timers, repeat * batch_size);
                }
            }

            void test_encode_simd() {

                // Batch op for simd encoding is only implemented for BFV now.
                bool has_batch_op = scheme == SchemeType::BFV;
                if (batch_size > 1 && !has_batch_op) return;

                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_encode = timer.register_timer(name + ".EncodeSimd");
                    size_t timer_decode = timer.register_timer(name + ".DecodeSimd");
                    auto message = context.batch_random_simd_full(batch_size);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {

                        timer.tick(timer_encode);
                        vector<Plaintext> plain;
                        if (batch_size == 1) {
                            plain.resize(1);
                            plain[0] = context.encoder().encode_simd(message[0], std::nullopt, scale, pool);
                        } else {
                            if (context.encoder().is_batch()) {
                                const BatchEncoder& encoder = context.encoder().batch(); 
                                size_t n = encoder.slot_count();
                                // Collect all message integers into one continuous array
                                utils::Array<uint64_t> message_total = utils::Array<uint64_t>::create_uninitialized(n * batch_size, false);
                                for (size_t j = 0; j < batch_size; j++) {
                                    message_total.slice(j * n, (j + 1) * n).copy_from_slice(
                                        utils::ConstSlice<uint64_t>(message[j].integers().data(), n, false, nullptr)
                                    );
                                }
                                // Convey to device
                                if (device) message_total.to_device_inplace(pool);
                                // Batched encode
                                plain.resize(batch_size);
                                vector<utils::ConstSlice<uint64_t>> slices; slices.reserve(batch_size);
                                vector<Plaintext*> plain_ptrs; plain_ptrs.reserve(batch_size);
                                for (size_t j = 0; j < batch_size; j++) {
                                    slices.push_back(message_total.slice(j * n, (j + 1) * n));
                                    plain_ptrs.push_back(&plain[j]);
                                }
                                encoder.encode_slice_batched(slices, plain_ptrs, pool);
                            } else {
                                throw std::runtime_error("Batch op for simd encoding is only implemented for BFV now.");
                            }
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_encode);

                        timer.tick(timer_decode);
                        vector<GeneralVector> decoded;
                        if (batch_size == 1) {
                            decoded.resize(1);
                            decoded[0] = context.encoder().decode_simd(plain[0], pool);
                        } else {
                            if (context.encoder().is_batch()) {
                                const BatchEncoder& encoder = context.encoder().batch(); 
                                size_t n = encoder.slot_count();
                                decoded.resize(batch_size);
                                utils::Array<uint64_t> decoded_total = utils::Array<uint64_t>::create_uninitialized(
                                    n * batch_size, device, pool
                                );
                                vector<utils::Slice<uint64_t>> slices; slices.reserve(batch_size);
                                for (size_t j = 0; j < batch_size; j++) {
                                    slices.push_back(decoded_total.slice(j * n, (j + 1) * n));
                                }
                                encoder.decode_slice_batched(batch_utils::collect_const_pointer(plain), slices, pool);
                                decoded_total.to_host_inplace();
                                for (size_t j = 0; j < batch_size; j++) {
                                    std::vector<uint64_t> decoded_vec; decoded_vec.reserve(n);
                                    for (size_t k = 0; k < n; k++) {
                                        decoded_vec.push_back(decoded_total[j * n + k]);
                                    }
                                    decoded[j] = GeneralVector(std::move(decoded_vec), false);
                                }
                            } else {
                                throw std::runtime_error("Batch op for simd encoding is only implemented for BFV now.");
                            }
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_decode);
                        if (i == 0 && !args.no_test_correct) {
                            assert_true(context.batch_near_equal(message, decoded), "test_encode_simd failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_encode_polynomial() {
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_encode = timer.register_timer(name + ".EncodePoly");
                    size_t timer_decode = timer.register_timer(name + ".DecodePoly");
                    GeneralVector message = context.random_polynomial_full();
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_encode);
                        Plaintext plain = context.encoder().encode_polynomial(message, std::nullopt, scale, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_encode);
                        timer.tick(timer_decode);
                        GeneralVector decoded = context.encoder().decode_polynomial(plain, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_decode);
                        if (i == 0 && !args.no_test_correct) {
                            assert_true(context.near_equal(message, decoded), "test_encode_polynomial failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_encrypt() {
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_enc_asym = timer.register_timer(name + ".EncryptAsym");
                    size_t timer_enc_sym = timer.register_timer(name + ".EncryptSym");
                    size_t timer_dec = timer.register_timer(name + ".Decrypt");
                    auto message = context.batch_random_simd_full(batch_size);
                    auto plain = context.encoder().batch_encode_simd(message, std::nullopt, scale, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {

                        timer.tick(timer_enc_asym);
                        std::vector<Ciphertext> cipher;
                        if (batch_size == 1) {
                            cipher.resize(1);
                            cipher[0] = context.encryptor().encrypt_asymmetric_new(plain[0], nullptr, pool);
                        } else {
                            cipher.resize(batch_size);
                            auto plain_ptrs = batch_utils::collect_const_pointer(plain);
                            auto cipher_ptrs = batch_utils::collect_pointer(cipher);
                            context.encryptor().encrypt_asymmetric_batched(plain_ptrs, cipher_ptrs, nullptr, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_enc_asym);

                        timer.tick(timer_enc_sym);
                        std::vector<Ciphertext> cipher_sym;
                        if (batch_size == 1) {
                            cipher_sym.resize(1);
                            cipher_sym[0] = context.encryptor().encrypt_symmetric_new(plain[0], true, nullptr, pool);
                        } else {
                            cipher_sym.resize(batch_size);
                            auto cipher_ptrs = batch_utils::collect_pointer(cipher_sym);
                            auto plain_ptrs = batch_utils::collect_const_pointer(plain);
                            context.encryptor().encrypt_symmetric_batched(plain_ptrs, true, cipher_ptrs, nullptr, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_enc_sym);

                        timer.tick(timer_dec);
                        std::vector<Plaintext> plain_dec;
                        if (batch_size == 1) {
                            plain_dec.resize(1);
                            plain_dec[0] = context.decryptor().decrypt_new(cipher[0], pool);
                        } else {
                            plain_dec.resize(batch_size);
                            auto plain_dec_ptrs = batch_utils::collect_pointer(plain_dec);
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            context.decryptor().decrypt_batched(cipher_ptrs, plain_dec_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_dec);
                        
                        if (i == 0 && !args.no_test_correct) {
                            auto decoded = context.encoder().batch_decode_simd(plain_dec, pool);
                            assert_true(context.batch_near_equal(message, decoded), "test_encrypt/asymmetric failed.");
                            for (auto& each: cipher_sym) each.expand_seed(context.context());
                            decoded = context.encoder().batch_decode_simd(context.batch_decrypt(cipher_sym), pool);
                            assert_true(context.batch_near_equal(message, decoded), "test_encrypt/symmetric failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_negate() {
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_neg = timer.register_timer(name + ".Negate");
                    auto message = this->batch_random_simd_full(context);
                    auto plain = this->batch_encode_simd(context, message, scale, pool);
                    auto cipher = this->batch_encrypt_asymmetric(context, plain, pool);
                    Ciphertext result; 
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_neg);
                        vector<Ciphertext> cipher_neg;
                        if (batch_size == 1) {
                            cipher_neg.resize(1);
                            cipher_neg[0] = context.evaluator().negate_new(cipher[0], pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_neg.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_neg);
                            context.evaluator().negate_batched(cipher_ptrs, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_neg);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = this->batch_decrypt(context, cipher_neg, pool);
                            auto decoded = this->batch_decode_simd(context, decrypted, pool);
                            auto truth = context.batch_negate(message);
                            assert_true(context.batch_near_equal(decoded, truth), "test_negate failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_translate() {
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_add = timer.register_timer(name + ".Add");
                    size_t timer_sub = timer.register_timer(name + ".Sub");
                    auto message = this->batch_random_simd_full(context);
                    auto plain = this->batch_encode_simd(context, message, scale, pool);
                    auto cipher = this->batch_encrypt_asymmetric(context, plain, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_add);
                        vector<Ciphertext> cipher_add;
                        if (batch_size == 1) {
                            cipher_add.resize(1);
                            cipher_add[0] = context.evaluator().add_new(cipher[0], cipher[0], pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_add.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_add);
                            context.evaluator().add_batched(cipher_ptrs, cipher_ptrs, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_add);
                        timer.tick(timer_sub);
                        vector<Ciphertext> cipher_sub;
                        if (batch_size == 1) {
                            cipher_sub.resize(1);
                            cipher_sub[0] = context.evaluator().sub_new(cipher[0], cipher[0], pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_sub.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_sub);
                            context.evaluator().sub_batched(cipher_ptrs, cipher_ptrs, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_sub);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = this->batch_decrypt(context, cipher_add, pool);
                            auto decoded = this->batch_decode_simd(context, decrypted, pool);
                            assert_true(context.batch_near_equal(context.batch_add(message, message), decoded), "test_translate/add failed.");
                            decrypted = this->batch_decrypt(context, cipher_sub, pool);
                            decoded = this->batch_decode_simd(context, decrypted, pool);
                            assert_true(context.batch_near_equal(context.batch_sub(message, message), decoded), "test_translate/sub failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }
            

            void test_translate_plain() {
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_add = timer.register_timer(name + ".AddPlain");
                    size_t timer_sub = timer.register_timer(name + ".SubPlain");
                    GeneralVector message = context.random_simd_full();
                    Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale, pool);
                    Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain, nullptr, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_add);
                        Ciphertext cipher_add = context.evaluator().add_plain_new(cipher, plain, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_add);
                        timer.tick(timer_sub);
                        Ciphertext cipher_sub = context.evaluator().sub_plain_new(cipher, plain, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_sub);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_add, pool), pool);
                            assert_true(context.near_equal(context.add(message, message), decrypted), "test_translate_plain/add failed.");
                            decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_sub, pool), pool);
                            assert_true(context.near_equal(context.sub(message, message), decrypted), "test_translate_plain/sub failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_multiply_relinearize() {
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_mul = timer.register_timer(name + ".Multiply");
                    size_t timer_square = timer.register_timer(name + ".Square");
                    size_t timer_relin = timer.register_timer(name + ".Relinearize");
                    RelinKeys relin_keys = context.key_generator().create_relin_keys(false, 2, pool);
                    GeneralVector message = context.random_simd_full();
                    Plaintext plain = context.encoder().encode_simd(message, std::nullopt, scale, pool);
                    Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain, nullptr, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_mul);
                        Ciphertext cipher_mul = context.evaluator().multiply_new(cipher, cipher, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_mul);
                        timer.tick(timer_square);
                        Ciphertext cipher_square = context.evaluator().square_new(cipher, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_square);
                        timer.tick(timer_relin);
                        Ciphertext cipher_relin = context.evaluator().relinearize_new(cipher_mul, relin_keys, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_relin);
                        if (i == 0 && !args.no_test_correct) {
                            auto truth = context.mul(message, message);
                            auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_mul, pool), pool);
                            assert_true(context.near_equal(truth, decrypted), "test_multiply_relinearize/multiply failed.");
                            decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_relin, pool), pool);
                            assert_true(context.near_equal(truth, decrypted), "test_multiply_relinearize/relin failed.");
                            decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_square, pool), pool);
                            assert_true(context.near_equal(truth, decrypted), "test_multiply_relinearize/square failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_multiply_plain() {
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_mul = timer.register_timer(name + ".MultiplyPlain");
                    auto message = this->batch_random_simd_full(context);
                    auto plain = this->batch_encode_simd(context, message, scale, pool);
                    auto cipher = this->batch_encrypt_asymmetric(context, plain, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_mul);
                        vector<Ciphertext> cipher_mul;
                        if (batch_size == 1) {
                            cipher_mul.resize(1);
                            cipher_mul[0] = context.evaluator().multiply_plain_new(cipher[0], plain[0], pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            auto plain_ptrs = batch_utils::collect_const_pointer(plain);
                            cipher_mul.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_mul);
                            context.evaluator().multiply_plain_batched(cipher_ptrs, plain_ptrs, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_mul);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = this->batch_decrypt(context, cipher_mul, pool);
                            auto decoded = this->batch_decode_simd(context, decrypted, pool);
                            assert_true(context.batch_near_equal(context.batch_mul(message, message), decoded), "test_multiply_plain failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_mod_switch_to_next() {
                bool can_run = get_context(0).context()->first_context_data().value()->next_context_data().has_value();
                if (!can_run) return;
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_mod = timer.register_timer(name + ".ModSwitchToNext");
                    auto message = context.batch_random_simd_full(batch_size);
                    auto plain = context.encoder().batch_encode_simd(message, std::nullopt, scale, pool);
                    auto cipher = context.batch_encrypt_asymmetric(plain);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_mod);
                        vector<Ciphertext> cipher_mod;
                        if (batch_size == 1) {
                            cipher_mod.resize(1);
                            cipher_mod[0] = context.evaluator().mod_switch_to_next_new(cipher[0], pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_mod.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_mod);
                            context.evaluator().mod_switch_to_next_batched(cipher_ptrs, result_ptrs, pool);
                        };
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_mod);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = context.encoder().batch_decode_simd(context.batch_decrypt(cipher_mod), pool);
                            assert_true(context.batch_near_equal(message, decrypted), "test_mod_switch_to_next failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_rescale_to_next() {
                bool is_ckks = get_context(0).params_host().scheme() == SchemeType::CKKS;
                if (!is_ckks) return;
                bool can_run = get_context(0).context()->first_context_data().value()->next_context_data().has_value();
                if (!can_run) return;
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_rescale = timer.register_timer(name + ".RescaleToNext");
                    GeneralVector message = context.random_simd_full();
                    const EncryptionParameters& parms = context.params_host();
                    auto coeff_modulus = parms.coeff_modulus();
                    double expanded_scale = scale * coeff_modulus[coeff_modulus.size() - 2].value();
                    Plaintext plain = context.encoder().encode_simd(message, std::nullopt, expanded_scale, pool);
                    Ciphertext cipher = context.encryptor().encrypt_asymmetric_new(plain, nullptr, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_rescale);
                        Ciphertext cipher_rescale = context.evaluator().rescale_to_next_new(cipher, pool);
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_rescale);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = context.encoder().decode_simd(context.decryptor().decrypt_new(cipher_rescale, pool), pool);
                            assert_true(context.near_equal(message, decrypted), "test_rescale_to_next failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_rotate_rows(size_t rotate_count) {
                bool is_ckks = get_context(0).params_host().scheme() == SchemeType::CKKS;
                if (is_ckks) return;
                auto thread_lambda = [this, rotate_count](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_rotate = timer.register_timer(name + ".RotateRows(" + std::to_string(rotate_count) + ")");
                    GaloisKeys glk = context.key_generator().create_galois_keys(false, pool);
                    auto message = this->batch_random_simd_full(context);
                    auto plain = this->batch_encode_simd(context, message, scale, pool);
                    auto cipher = this->batch_encrypt_asymmetric(context, plain, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_rotate);
                        vector<Ciphertext> cipher_result;
                        if (batch_size == 1) {
                            cipher_result.resize(1);
                            cipher_result[0] = context.evaluator().rotate_rows_new(cipher[0], rotate_count, glk, pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_result.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_result);
                            context.evaluator().rotate_rows_batched(cipher_ptrs, rotate_count, glk, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_rotate);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = this->batch_decrypt(context, cipher_result, pool);
                            auto decoded = this->batch_decode_simd(context, decrypted, pool);
                            auto truth = context.batch_rotate(message, rotate_count);
                            assert_true(context.batch_near_equal(decoded, truth), "test_rotate_rows failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_rotate_columns() {
                bool is_ckks = get_context(0).params_host().scheme() == SchemeType::CKKS;
                if (is_ckks) return;
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_rotate = timer.register_timer(name + ".RotateColumns");
                    GaloisKeys glk = context.key_generator().create_galois_keys(false, pool);
                    auto message = this->batch_random_simd_full(context);
                    auto plain = this->batch_encode_simd(context, message, scale, pool);
                    auto cipher = this->batch_encrypt_asymmetric(context, plain, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_rotate);
                        vector<Ciphertext> cipher_result;
                        if (batch_size == 1) {
                            cipher_result.resize(1);
                            cipher_result[0] = context.evaluator().rotate_columns_new(cipher[0], glk, pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_result.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_result);
                            context.evaluator().rotate_columns_batched(cipher_ptrs, glk, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_rotate);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = this->batch_decrypt(context, cipher_result, pool);
                            auto decoded = this->batch_decode_simd(context, decrypted, pool);
                            auto truth = context.batch_conjugate(message);
                            assert_true(context.batch_near_equal(decoded, truth), "test_rotate_columns failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_rotate_vector(size_t rotate_count) {
                bool is_ckks = get_context(0).params_host().scheme() == SchemeType::CKKS;
                if (!is_ckks) return;
                auto thread_lambda = [this, rotate_count](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_rotate = timer.register_timer(name + ".RotateVector(" + std::to_string(rotate_count) + ")");
                    GaloisKeys glk = context.key_generator().create_galois_keys(false, pool);
                    auto message = this->batch_random_simd_full(context);
                    auto plain = this->batch_encode_simd(context, message, scale, pool);
                    auto cipher = this->batch_encrypt_asymmetric(context, plain, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_rotate);
                        vector<Ciphertext> cipher_result;
                        if (batch_size == 1) {
                            cipher_result.resize(1);
                            cipher_result[0] = context.evaluator().rotate_vector_new(cipher[0], rotate_count, glk, pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_result.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_result);
                            context.evaluator().rotate_vector_batched(cipher_ptrs, rotate_count, glk, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_rotate);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = this->batch_decrypt(context, cipher_result, pool);
                            auto decoded = this->batch_decode_simd(context, decrypted, pool);
                            auto truth = context.batch_rotate(message, rotate_count);
                            assert_true(context.batch_near_equal(decoded, truth), "test_rotate_vector failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_complex_conjugate() {
                bool is_ckks = get_context(0).params_host().scheme() == SchemeType::CKKS;
                if (!is_ckks) return;
                auto thread_lambda = [this](size_t thread_index) {
                    double scale = args.scale;
                    const GeneralHeContext& context = get_context(thread_index);
                    MemoryPoolHandle pool = get_pool(thread_index);
                    size_t repeat = get_repeat(thread_index);
                    Timer timer;
                    size_t timer_rotate = timer.register_timer(name + ".Conjugate");
                    GaloisKeys glk = context.key_generator().create_galois_keys(false, pool);
                    auto message = this->batch_random_simd_full(context);
                    auto plain = this->batch_encode_simd(context, message, scale, pool);
                    auto cipher = this->batch_encrypt_asymmetric(context, plain, pool);
                    for (size_t i = 0; i < repeat + warm_up_repeat; i++) {
                        timer.tick(timer_rotate);
                        vector<Ciphertext> cipher_result;
                        if (batch_size == 1) {
                            cipher_result.resize(1);
                            cipher_result[0] = context.evaluator().complex_conjugate_new(cipher[0], glk, pool);
                        } else {
                            auto cipher_ptrs = batch_utils::collect_const_pointer(cipher);
                            cipher_result.resize(batch_size);
                            auto result_ptrs = batch_utils::collect_pointer(cipher_result);
                            context.evaluator().complex_conjugate_batched(cipher_ptrs, glk, result_ptrs, pool);
                        }
                        if (device) cudaStreamSynchronize(0);
                        if (i >= warm_up_repeat) timer.tock(timer_rotate);
                        if (i == 0 && !args.no_test_correct) {
                            auto decrypted = this->batch_decrypt(context, cipher_result, pool);
                            auto decoded = this->batch_decode_simd(context, decrypted, pool);
                            auto truth = context.batch_conjugate(message);
                            assert_true(context.batch_near_equal(decoded, truth), "test_complex_conjugate failed.");
                        }
                    }
                    return timer;
                };
                run_threads_and_print_times(thread_lambda);
            }

            void test_suite() {
                bool is_ckks = scheme == SchemeType::CKKS;
                if (!args.no_bench_encode_simd) test_encode_simd();
                if (!args.no_bench_encode_polynomial) test_encode_polynomial();
                if (!args.no_bench_encrypt) test_encrypt();
                if (!args.no_bench_negate) test_negate();
                if (!args.no_bench_translate) test_translate();
                if (!args.no_bench_translate_plain) test_translate_plain();
                if (!args.no_bench_multiply_relinearize) test_multiply_relinearize();
                if (!args.no_bench_multiply_plain) test_multiply_plain();
                if (!args.no_bench_mod_switch_to_next) test_mod_switch_to_next();
                if (is_ckks)
                    if (!args.no_bench_rescale_to_next) test_rescale_to_next();
                if (!is_ckks) {
                    if (!args.no_bench_rotate_rows) test_rotate_rows(1);
                    if (!args.no_bench_rotate_rows) test_rotate_rows(7);
                    if (!args.no_bench_rotate_columns) test_rotate_columns();
                } else {
                    if (!args.no_bench_rotate_vector) test_rotate_vector(1);
                    if (!args.no_bench_rotate_vector) test_rotate_vector(7);
                    if (!args.no_bench_complex_conjugate) test_complex_conjugate();
                }
            }

        
    };
    
    void run_all(bool device, const Arguments& args) {

        if (args.scheme_bfv) {
            std::cout << "[BFV]" << std::endl;
            Benchmark bench_bfv(device, "BFV", SchemeType::BFV, args);
            bench_bfv.test_suite();
        }

        if (args.scheme_ckks) {
            std::cout << "[CKKS]" << std::endl;
            Benchmark bench_ckks(device, "CKKS", SchemeType::CKKS, args);
            bench_ckks.test_suite();
        }

        if (args.scheme_bgv) {
            std::cout << "[BGV]" << std::endl;
            Benchmark bench_bgv(device, "BGV", SchemeType::BGV, args);
            bench_bgv.test_suite();
        }
    }

}

int main(int argc, char** argv) {
    using std::cout;
    using std::endl;
    using std::string;
    using bench::he_operations::Arguments;
    Arguments args(argc, argv);
    if (args.help) {
        Arguments::print_help();
        return 0;
    } 

    args.print_arguments();

    if (args.host) {
        cout << endl << "========= HOST =========" << endl;
        bench::he_operations::run_all(false, args);
    }
    if (args.device) {
        cout << endl << "======== DEVICE ========" << endl;
        bench::he_operations::run_all(true, args);
        troy::utils::MemoryPool::Destroy();
    }
    return 0;
}