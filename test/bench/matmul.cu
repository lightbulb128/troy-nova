#include "../../src/app/matmul.h"
#include "../test_multithread.h"
#include "../../src/utils/timer.h"
#include "argument_helper.h"
#include "../argparse.h"
#include <cassert>
#include <thread>
#include <future>

namespace bench::matmul {

    using namespace troy;
    using namespace tool;
    using namespace troy::linear;
    using tool::GeneralEncoder;
    using tool::GeneralHeContext;
    using tool::GeneralVector;
    using std::stringstream;
    using std::vector;
    using troy::bench::Timer;
    using troy::bench::TimerThreaded;
    using troy::bench::TimerOnce;

    struct Arguments {
        bool help = false;
        bool host = false;
        bool device = false;
        bool scheme_bfv = false;
        bool scheme_ckks = false;
        bool scheme_bgv = false;

        size_t repeat;

        size_t m;
        size_t n;
        size_t r;

        size_t poly_modulus_degree = 8192;
        size_t simd_log_t = 40;
        size_t ring2k_log_t = 0;
        std::vector<size_t> log_q = { 60, 40, 40, 60 };
        uint64_t seed = 0x123;
        double input_max = 5;
        double scale = 1 << 30;
        double tolerance = 0.1;
        size_t threads = 1;
        bool multiple_pools = false;
        bool multiple_devices = false;

        bool use_special_prime_for_encryption = false;
        size_t mod_switch_down_levels = 0;
        bool pack_lwes = true;
        bool batched_mul = true;
        bool use_zstd = false;

        bool no_check_correctness = false;
        
        Arguments(int argc, char** argv) {
            ArgumentParser parser(argc, argv);

            help = parser.get_bool_store_true("-h").value_or(parser.get_bool_store_true("--help").value_or(false));

            host = parser.get_bool_store_true("-H").value_or(parser.get_bool_store_true("--host").value_or(false));
            device = parser.get_bool_store_true("-D").value_or(parser.get_bool_store_true("--device").value_or(false));
            if (not host and not device) {
                host = true;
            } else if (host) {
                device = false;
            } // only one of host and device can be true

            scheme_bfv = parser.get_bool_store_true("--bfv").value_or(false);
            scheme_ckks = parser.get_bool_store_true("--ckks").value_or(false);
            scheme_bgv = parser.get_bool_store_true("--bgv").value_or(false);
            if (not scheme_bfv and not scheme_ckks and not scheme_bgv) {
                scheme_bfv = true;
            } else if (scheme_bfv) {
                scheme_ckks = false;
                scheme_bgv = false;
            } else if (scheme_ckks) {
                scheme_bgv = false;
            } // only one scheme can be true

            repeat = parser.get_uint<size_t>("-R").value_or(parser.get_uint<size_t>("--repeat").value_or(1));

            m = parser.get_uint<size_t>("-m").value_or(parser.get_uint<size_t>("--m").value_or(10));
            n = parser.get_uint<size_t>("-n").value_or(parser.get_uint<size_t>("--n").value_or(10));
            r = parser.get_uint<size_t>("-r").value_or(parser.get_uint<size_t>("--r").value_or(10));

            poly_modulus_degree = parser.get_uint<size_t>("-N").value_or(parser.get_uint<size_t>("--poly-modulus-degree").value_or(poly_modulus_degree));
            simd_log_t = parser.get_uint<size_t>("-st").value_or(parser.get_uint<size_t>("--simd-log-t").value_or(simd_log_t));
            ring2k_log_t = parser.get_uint<size_t>("-rt").value_or(parser.get_uint<size_t>("--ring2k-log-t").value_or(ring2k_log_t));
            log_q = parser.get_uint_list<size_t>("-q").value_or(parser.get_uint_list<size_t>("--log-q").value_or(log_q));
            seed = parser.get_uint<uint64_t>("-S").value_or(parser.get_uint<uint64_t>("--seed").value_or(seed));
            input_max = parser.get_float<double>("-i").value_or(parser.get_float<double>("--input-max").value_or(input_max));
            scale = parser.get_float<double>("-s").value_or(parser.get_float<double>("--scale").value_or(scale));
            tolerance = parser.get_float<double>("-T").value_or(parser.get_float<double>("--tolerance").value_or(tolerance));
            threads = parser.get_uint<size_t>("-c").value_or(parser.get_uint<size_t>("--threads").value_or(threads));

            multiple_pools = parser.get_bool_store_true("-mp").value_or(parser.get_bool_store_true("--multiple-pools").value_or(false));
            multiple_devices = parser.get_bool_store_true("-md").value_or(parser.get_bool_store_true("--multiple-devices").value_or(false));

            use_special_prime_for_encryption = parser.get_bool_store_true("-usp").value_or(parser.get_bool_store_true("--use-special-prime-for-encryption").value_or(false));
            mod_switch_down_levels = parser.get_uint<size_t>("-msd").value_or(parser.get_uint<size_t>("--mod-switch-down-levels").value_or(0));
            bool no_pack_lwes = parser.get_bool_store_true("-np").value_or(parser.get_bool_store_true("--no-pack-lwes").value_or(false));
            pack_lwes = !no_pack_lwes;
            bool no_batched_mul = parser.get_bool_store_true("-nbm").value_or(parser.get_bool_store_true("--no-batched-mul").value_or(false));
            batched_mul = !no_batched_mul;
            use_zstd = parser.get_bool_store_true("--use-zstd").value_or(false);
            no_check_correctness = parser.get_bool_store_true("--no-check-correctness").value_or(false);

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
            if (ring2k_log_t > 64 && log_q == vector<size_t>{ 60, 40, 40, 60 }) {
                std::cout << "Warning: using uint128_t but default log_q. This might lead to wrong results. Setting to log_q = {60*6}" << std::endl;
                log_q = vector<size_t>{ 60, 60, 60, 60, 60, 60 };
            }
            if (threads > m) {
                throw std::invalid_argument("m must be equal to or greater than threads");
            }
            if (multiple_devices || multiple_pools) {
                if (!device) {
                    std::cout << "Warning: multiple-pools and multiple-devices require device. Setting device to true." << std::endl;
                    host = false;
                    device = true;
                }
            }
            if (!multiple_pools && multiple_devices) {
                std::cout << "Warning: multiple-devices requires multiple-pools. Setting multiple-pools to true." << std::endl;
                multiple_pools = true;
            }
            if (ring2k_log_t != 0 and simd_log_t != 0) {
                simd_log_t = 0;
                std::cout << "Warning: simd_log_t and ring2k_log_t cannot be both set. Setting simd_log_t to 0." << std::endl;
            }
            if (ring2k_log_t == 0 && simd_log_t == 0) {
                throw std::invalid_argument("Either simd_log_t or ring2k_log_t must be set");
            }
            if (ring2k_log_t && !scheme_bfv) {
                std::cout << "Warning: ring2k_log_t is only used in BFV scheme. Setting scheme to BFV." << std::endl;
                scheme_bfv = true;
                scheme_ckks = false;
                scheme_bgv = false;
            }
        }

        
        static void print_help() {
            std::cout << "Usage: bench_matmul [options]" << std::endl;
            std::cout << "Run benchmark for HE matmul" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << std::endl;
            std::cout << "  -H, --host                  Run on host" << std::endl;
            std::cout << "  -D, --device                Run on device" << std::endl;
            std::cout << "      Exactly one of -H or -D should be set." << std::endl;
            std::cout << std::endl;
            std::cout << "  --bfv                       Run BFV scheme" << std::endl;
            std::cout << "  --ckks                      Run CKKS scheme" << std::endl;
            std::cout << "  --bgv                       Run BGV scheme" << std::endl;
            std::cout << "      Exactly one of --bfv, --ckks, or --bgv should be set." << std::endl;
            std::cout << std::endl;
            std::cout << "  -m, --m                     LHS rows (default: 10)" << std::endl;
            std::cout << "  -r, --r                     LHS columns = RHS rows (default: 10)" << std::endl;
            std::cout << "  -n, --n                     RHS columns (default: 10)" << std::endl;
            std::cout << std::endl;
            std::cout << "  -R, --repeat                Repeat count (default: 1)" << std::endl;
            std::cout << "  -N, --poly-modulus-degree   Poly modulus degree (default: 8192)" << std::endl;
            std::cout << "  -st, --simd-log-t           SIMD log t (default: 40)" << std::endl;
            std::cout << "  -rt, --ring2k-log-t         Ring2k log t (default: 0)" << std::endl;
            std::cout << "      Either simd-log-t or ring2k-log-t must be set (the other set to 0)." << std::endl;
            std::cout << "  -q, --log-q                 Log q (default: 60,40,40,60)" << std::endl;
            std::cout << "  -S, --seed                  Seed (default: 0x123)" << std::endl;
            std::cout << "  -i, --input-max             Input max (default: 10)" << std::endl;
            std::cout << "  -s, --scale                 Scale (default: 1 << 20)" << std::endl;
            std::cout << "  -T, --tolerance             Tolerance (default: 1e-2)" << std::endl;
            std::cout << "  -c, --threads               Number of threads (default: 1)" << std::endl;
            std::cout << "  -mp, --multiple-pools       Use multiple pools" << std::endl;
            std::cout << "  -md, --multiple-devices     Use multiple devices" << std::endl;
            std::cout << "  -usp, --use-special-prime-for-encryption" << std::endl;
            std::cout << "                              Use special prime for encryption" << std::endl;
            std::cout << "  -msd, --mod-switch-down-levels" << std::endl;
            std::cout << "  -np, --no-pack-lwes         Do not pack lwes" << std::endl;
            std::cout << "  -nbm, --no-batched-mul      Do not use batched mul" << std::endl;
            std::cout << "  --use-zstd                  Use Zstd for compressing serialized ciphers" << std::endl;
            std::cout << "  --no-check-correctness      Do not check correctness" << std::endl;
            std::cout << std::endl;
        }

        void print_arguments() {
            std::cout << "[Arguments]" << std::endl;
            if (host) {
                std::cout << "  run-on              = host" << std::endl;
            } else {
                std::cout << "  run-on              = device" << std::endl;
            }
            std::cout << "  threads             = " << threads << std::endl;
            std::cout << "  dimensions          = " << m << " x " << r << " x " << n << std::endl;
            
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

            if (scheme_bfv) {
                std::cout << "  scheme              = BFV" << std::endl;
            } else if (scheme_ckks) {
                std::cout << "  scheme              = CKKS" << std::endl;
            } else if (scheme_bgv) {
                std::cout << "  scheme              = BGV" << std::endl;
            }

            std::cout << "  repeat              = " << repeat << std::endl;
            std::cout << "  poly-modulus-degree = " << poly_modulus_degree << std::endl;
            if (simd_log_t != 0) {
                std::cout << "  simd-log-t          = " << simd_log_t << std::endl;
            } else {
                std::cout << "  ring2k-log-t        = " << ring2k_log_t;
                if (ring2k_log_t > 64) std::cout << " (uint128_t)";
                else if (ring2k_log_t > 32) std::cout << " (uint64_t)";
                else std::cout << " (uint32_t)";
                std::cout << std::endl;
            }
            std::cout << "  log-q               = " << list_usize_to_string(log_q) << std::endl;

            std::cout << "  use-special-prime   = " << bool_to_string(use_special_prime_for_encryption) << std::endl;
            std::cout << "  mod-down-levels     = " << mod_switch_down_levels << std::endl;
            std::cout << "  pack-lwes           = " << bool_to_string(pack_lwes) << std::endl;
            std::cout << "  batched-mul         = " << bool_to_string(batched_mul) << std::endl;
            std::cout << "  use-zstd            = " << bool_to_string(use_zstd) << std::endl;
            std::cout << "  no-check-correct    = " << bool_to_string(no_check_correctness) << std::endl;

            std::cout << "  seed                = " << "0x" << std::hex << seed << std::dec << std::endl;
            if (scheme_ckks) {
                std::cout << "  input-max           = " << input_max << std::endl;
                std::cout << "  scale               = " << scale << std::endl;
                std::cout << "  tolerance           = " << tolerance << std::endl;
            }

        }

    };

    class BenchmarkMatmul {

        Arguments args;
        MultithreadHeContext environment;
        SchemeType scheme;
        
    public:

        GeneralVector matmul_plaintext(const GeneralVector& lhs, const GeneralVector& rhs, const GeneralVector& bias, size_t threads) const {
            size_t m = args.m;
            size_t n = args.n;
            size_t r = args.r;
            size_t m_divided = (m + threads - 1) / threads;
            GeneralVector output = GeneralVector::zeros_like(lhs, m * n);
            
            assert(lhs.size() == m * r);
            assert(rhs.size() == r * n);
            assert(bias.size() == m * n);

            uint64_t simd_t = environment.simd_t();
            uint128_t ring_t_mask = environment.ring_t_mask();
            
            auto thread_lambda = [m_divided, m, n, r, &output, &lhs, &rhs, &bias, simd_t, ring_t_mask](size_t thread_id) {
                size_t m_start = m_divided * thread_id;
                size_t m_end = std::min(m_divided * (thread_id + 1), m);
                for (size_t i = m_start; i < m_end; i++) {
                    for (size_t j = 0; j < n; j++) {
                        for (size_t k = 0; k < r; k++) {
                            if (lhs.is_complexes()) {
                                output.complexes()[i * n + j] += lhs.complexes()[i * r + k] * rhs.complexes()[k * n + j];
                            } else if (lhs.is_doubles()) {
                                output.doubles()[i * n + j] += lhs.doubles()[i * r + k] * rhs.doubles()[k * n + j];
                            } else if (lhs.is_integers()) {
                                uint128_t mult = static_cast<uint128_t>(lhs.integers()[i * r + k]) * static_cast<uint128_t>(rhs.integers()[k * n + j]);
                                mult = mult % static_cast<uint128_t>(simd_t);
                                output.integers()[i * n + j] = (output.integers()[i * n + j] + static_cast<uint64_t>(mult)) % simd_t;
                            } else if (lhs.is_uint32s()) {
                                output.uint32s()[i * n + j] += lhs.uint32s()[i * r + k] * rhs.uint32s()[k * n + j];
                                output.uint32s()[i * n + j] &= ring_t_mask;
                            } else if (lhs.is_uint64s()) {
                                output.uint64s()[i * n + j] += lhs.uint64s()[i * r + k] * rhs.uint64s()[k * n + j];
                                output.uint64s()[i * n + j] &= ring_t_mask;
                            } else if (lhs.is_uint128s()) {
                                output.uint128s()[i * n + j] += lhs.uint128s()[i * r + k] * rhs.uint128s()[k * n + j];
                                output.uint128s()[i * n + j] &= ring_t_mask;
                            } else {
                                throw std::runtime_error("Unsupported data type");
                            }
                        }
                        if (lhs.is_complexes()) {
                            output.complexes()[i * n + j] += bias.complexes()[i * n + j];
                        } else if (lhs.is_doubles()) {
                            output.doubles()[i * n + j] += bias.doubles()[i * n + j];
                        } else if (lhs.is_integers()) {
                            output.integers()[i * n + j] = (output.integers()[i * n + j] + bias.integers()[i * n + j]) % simd_t;
                        } else if (lhs.is_uint32s()) {
                            output.uint32s()[i * n + j] += bias.uint32s()[i * n + j];
                            output.uint32s()[i * n + j] &= ring_t_mask;
                        } else if (lhs.is_uint64s()) {
                            output.uint64s()[i * n + j] += bias.uint64s()[i * n + j];
                            output.uint64s()[i * n + j] &= ring_t_mask;
                        } else if (lhs.is_uint128s()) {
                            output.uint128s()[i * n + j] += bias.uint128s()[i * n + j];
                            output.uint128s()[i * n + j] &= ring_t_mask;
                        } else {
                            throw std::runtime_error("Unsupported data type");
                        }
                    }
                }
            };

            if (threads == 1) {
                thread_lambda(0);
            } else {
                std::vector<std::thread> thread_instances;
                for (size_t i = 0; i < threads; i++) {
                    thread_instances.push_back(std::thread(thread_lambda, i));
                }
                for (size_t i = 0; i < thread_instances.size(); i++) {
                    thread_instances[i].join();
                }
            }

            return output;
        }
        
        BenchmarkMatmul(Arguments args): args(args) {
            scheme = SchemeType::Nil;
            if (args.scheme_bfv) {
                scheme = SchemeType::BFV;
            } else if (args.scheme_ckks) {
                scheme = SchemeType::CKKS;
            } else if (args.scheme_bgv) {
                scheme = SchemeType::BGV;
            }
            bool device = args.device;
            GeneralHeContextParameters ghep(
                device, scheme, args.poly_modulus_degree, args.simd_log_t, args.log_q, true, args.seed, args.input_max,
                args.scale, args.tolerance, false, args.use_special_prime_for_encryption, MemoryPool::GlobalPool(), args.ring2k_log_t
            );
            
            environment = MultithreadHeContext(args.threads, args.multiple_pools, args.multiple_devices, ghep);
        }

        bool test_matmul() const {

            size_t m = args.m;
            size_t n = args.n;
            size_t r = args.r;
            size_t threads = args.threads;
            size_t m_divided = (m + threads - 1) / threads;

            // generate input data
            const GeneralHeContext& context = environment.get_context(0);
            GeneralVector x = context.random_polynomial(m * r);
            // x = GeneralVector(vector<uint64_t>{1, 2, 3, 4, 5, 6});
            x.resize(m_divided * threads * r); // pad to multiple of threads
            GeneralVector w = context.random_polynomial(r * n);
            // w = GeneralVector(vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
            GeneralVector s = context.random_polynomial(m * n);
            // s = GeneralVector(vector<uint64_t>(0, m * n));
            s.resize(m_divided * threads * n); // pad to multiple of threads

            // create helpers. one helper for each thread
            MatmulHelper helper(m_divided, r, n, context.params_host().poly_modulus_degree(), MatmulObjective::EncryptLeft, args.pack_lwes);
            helper.batched_mul = args.batched_mul;
            std::vector<MatmulHelper> thread_helpers = {};
            for (size_t i = 0; i < args.threads; i++) {
                thread_helpers.push_back(helper);
                thread_helpers[i].set_pool(environment.get_pool(i));
            }

            // create automorphism keys. one for each device.
            std::vector<GaloisKeys> context_automorphism_keys = {};
            context_automorphism_keys.push_back(context.key_generator().create_automorphism_keys(false));
            if (args.multiple_devices) {
                for (size_t i = 1; i < environment.get_context_count(); i++) {
                    context_automorphism_keys.push_back(context_automorphism_keys[0].clone(environment.pool_at(i)));
                }
            }

            // encode w_encoded. one for each device
            std::vector<Plain2d> context_w_encoded = {[&helper, &context, &w]() {
                const GeneralEncoder& encoder = context.encoder();
                Timer timer;
                size_t timer_handle = timer.register_timer("Ecd w (1 thread)");
                timer.tick(timer_handle);
                Plain2d w_encoded;
                if (encoder.is_batch()) {
                    w_encoded = helper.encode_weights_uint64s(encoder.batch(), w.integers().data());
                } else if (encoder.is_ckks()) {
                    w_encoded = helper.encode_weights_doubles(encoder.ckks(), w.doubles().data(), std::nullopt, context.scale());
                } else if (encoder.is_ring32()) {
                    w_encoded = helper.encode_weights_ring2k<uint32_t>(encoder.poly32(), w.uint32s().data(), std::nullopt);
                } else if (encoder.is_ring64()) {
                    w_encoded = helper.encode_weights_ring2k<uint64_t>(encoder.poly64(), w.uint64s().data(), std::nullopt);
                } else if (encoder.is_ring128()) {
                    w_encoded = helper.encode_weights_ring2k<uint128_t>(encoder.poly128(), w.uint128s().data(), std::nullopt);
                } else {
                    throw std::runtime_error("Unsupported encoder");
                }
                timer.tock(timer_handle);
                timer.print();
                return w_encoded;
            }()};
            for (size_t i = 1; i < environment.get_context_count(); i++) {
                context_w_encoded.push_back(context_w_encoded[0].clone(environment.pool_at(i)));
            }

            bool pack_lwes = args.pack_lwes;
            size_t mod_switch_down_levels = args.mod_switch_down_levels;

            ParmsID s_parms_id;
            if (context.encoder().is_ring32() || context.encoder().is_ring64() || context.encoder().is_ring128()) {
                ContextDataPointer pointer = context.context()->first_context_data_pointer();
                for (size_t i = 0; i < mod_switch_down_levels; i++) {
                    pointer = pointer->next_context_data_pointer();
                }
                s_parms_id = pointer->parms_id();
            }

            bool success = false;

            Timer total_timer;
            size_t total_timer_handle = total_timer.register_timer("Time cost");

            for (size_t rep = 0; rep < args.repeat; rep++) {
                bool last_rep = rep == args.repeat - 1;
                total_timer.tick(total_timer_handle);

                TimerOnce thread_timer;
                Timer total_timer_once; 
                size_t total_timer_once_handle = 0;
                if (args.repeat > 1) {
                    total_timer_once_handle = total_timer_once.register_timer(std::string("Time cost #") + std::to_string(rep + 1));
                    total_timer_once.tick(total_timer_once_handle);
                }
                std::vector<Timer> timers;
                timers.clear();

                // encode and encrypt inputs
                auto lambda_encrypt_inputs = [this, &x, m_divided, r](size_t thread_id, const MatmulHelper& helper) {
                    const GeneralHeContext& context = environment.get_context(thread_id);
                    HeContextPointer he = context.context();
                    const GeneralEncoder& encoder = context.encoder();
                    const Encryptor& encryptor = context.encryptor();
                    size_t m_lower = thread_id * m_divided;

                    Timer timer; timer.tab(1);

                    size_t timer_single_handle = timer.register_timer("Ecd/enc x");
                    timer.tick(timer_single_handle);
                    size_t offset = m_lower * r;
                    Cipher2d x_encrypted;
                    if (encoder.is_batch()) {
                        x_encrypted = helper.encrypt_inputs_uint64s(encryptor, encoder.batch(), x.integers().data() + offset);
                    } else if (encoder.is_ckks()) {
                        x_encrypted = helper.encrypt_inputs_doubles(encryptor, encoder.ckks(), x.doubles().data() + offset, std::nullopt, context.scale());
                    } else if (encoder.is_ring32()) {
                        x_encrypted = helper.encrypt_inputs_ring2k<uint32_t>(encryptor, encoder.poly32(), x.uint32s().data() + offset, std::nullopt);
                    } else if (encoder.is_ring64()) {
                        x_encrypted = helper.encrypt_inputs_ring2k<uint64_t>(encryptor, encoder.poly64(), x.uint64s().data() + offset, std::nullopt);
                    } else if (encoder.is_ring128()) {
                        x_encrypted = helper.encrypt_inputs_ring2k<uint128_t>(encryptor, encoder.poly128(), x.uint128s().data() + offset, std::nullopt);
                    } else {
                        throw std::runtime_error("Unsupported encoder");
                    }
                    timer.tock(timer_single_handle);

                    stringstream x_serialized;
                    timer_single_handle = timer.register_timer("Ser [x]");
                    timer.tick(timer_single_handle);
                    x_encrypted.save(x_serialized, he, this->args.use_zstd ? CompressionMode::Zstd : CompressionMode::Nil);
                    timer.tock(timer_single_handle);
                    
                    return std::make_pair(x_serialized.str(), std::move(timer));
                };

                // run threads
                std::vector<std::string> x_serialized;
                if (threads == 1) {
                    auto [x_serialized_single, timer_single] = lambda_encrypt_inputs(0, thread_helpers[0]);
                    x_serialized.push_back(std::move(x_serialized_single));
                    timers.push_back(timer_single);
                } else {
                    std::vector<std::future<std::pair<std::string, Timer>>> futures;
                    for (size_t i = 0; i < threads; i++) {
                        futures.push_back(std::async(std::launch::async, lambda_encrypt_inputs, i, thread_helpers[i]));
                    }
                    for (size_t i = 0; i < threads; i++) {
                        auto [x_serialized_single, timer_single] = futures[i].get();
                        x_serialized.push_back(x_serialized_single);
                        timers.push_back(timer_single);
                    }
                }
                if (last_rep) {
                    thread_timer.finish("Client-Enc[x]");
                    if (timers.size() == 1) timers[0].print();
                    else TimerThreaded::Print(timers);
                }
                timers.clear();

                size_t x_serialized_size = 0;
                for (size_t i = 0; i < threads; i++) {
                    x_serialized_size += x_serialized[i].size();
                }

                thread_timer.restart();

                // calculate matmul and serialize ys
                auto lambda_matmul = [this, m_divided, n, &s, pack_lwes, mod_switch_down_levels, &s_parms_id](
                    size_t thread_id, std::string x_serialized, const MatmulHelper& helper, const GaloisKeys& automorphism_key, const Plain2d& w_encoded
                ) {
                    const GeneralHeContext& context = environment.get_context(thread_id);
                    HeContextPointer he = context.context();
                    const GeneralEncoder& encoder = context.encoder();
                    const Evaluator& evaluator = context.evaluator();
                    MemoryPoolHandle pool = environment.get_pool(thread_id);

                    Timer timer; timer.tab(1);
                    
                    size_t timer_single_handle = timer.register_timer("Ecd s");
                    timer.tick(timer_single_handle);
                    Plain2d s_encoded;
                    size_t offset = thread_id * m_divided * n;
                    if (encoder.is_batch()) {
                        s_encoded = helper.encode_outputs_uint64s(encoder.batch(), s.integers().data() + offset);
                    } else if (encoder.is_ckks()) {
                        s_encoded = helper.encode_outputs_doubles(encoder.ckks(), s.doubles().data() + offset, std::nullopt, context.scale() * context.scale());
                    } else if (encoder.is_ring32()) {
                        s_encoded = helper.encode_outputs_ring2k<uint32_t>(encoder.poly32(), s.uint32s().data() + offset, s_parms_id);
                    } else if (encoder.is_ring64()) {
                        s_encoded = helper.encode_outputs_ring2k<uint64_t>(encoder.poly64(), s.uint64s().data() + offset, s_parms_id);
                    } else if (encoder.is_ring128()) {
                        s_encoded = helper.encode_outputs_ring2k<uint128_t>(encoder.poly128(), s.uint128s().data() + offset, s_parms_id);
                    } else {
                        throw std::runtime_error("Unsupported encoder");
                    }
                    timer.tock(timer_single_handle);

                    timer_single_handle = timer.register_timer("Deser [x]");
                    timer.tick(timer_single_handle);
                    stringstream x_serialized_stream(x_serialized);
                    Cipher2d x_encrypted = Cipher2d::load_new(x_serialized_stream, he, pool);
                    timer.tock(timer_single_handle);
                    
                    timer_single_handle = timer.register_timer("Matmul");
                    timer.tick(timer_single_handle);
                    Cipher2d y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded);
                    timer.tock(timer_single_handle);
                    
                    if (mod_switch_down_levels > 0) {
                        timer_single_handle = timer.register_timer("Mod switch");
                        timer.tick(timer_single_handle);
                        for (size_t i = 0; i < mod_switch_down_levels; i++) {
                            y_encrypted.mod_switch_to_next_inplace(evaluator, pool);
                        }
                        timer.tock(timer_single_handle);
                    }

                    if (pack_lwes) {
                        timer_single_handle = timer.register_timer("Pack LWEs");
                        timer.tick(timer_single_handle);
                        y_encrypted = helper.pack_outputs(evaluator, automorphism_key, y_encrypted);
                        timer.tock(timer_single_handle);
                    }

                    timer_single_handle = timer.register_timer("Add s");
                    timer.tick(timer_single_handle);
                    y_encrypted.add_plain_inplace(evaluator, s_encoded, pool);
                    timer.tock(timer_single_handle);

                    stringstream y_serialized;
                    timer_single_handle = timer.register_timer("Ser [y]");
                    timer.tick(timer_single_handle);
                    helper.serialize_outputs(evaluator, y_encrypted, y_serialized, this->args.use_zstd ? CompressionMode::Zstd : CompressionMode::Nil);
                    timer.tock(timer_single_handle);;

                    return std::make_pair(y_serialized.str(), std::move(timer));

                };

                // run threads
                std::vector<std::string> y_serialized;
                if (threads == 1) {
                    auto [y_serialized_single, timer_single] = lambda_matmul(0, x_serialized[0], thread_helpers[0], context_automorphism_keys[0], context_w_encoded[0]);
                    y_serialized.push_back(y_serialized_single);
                    timers.push_back(timer_single);
                } else {
                    std::vector<std::future<std::pair<std::string, Timer>>> futures;
                    for (size_t i = 0; i < threads; i++) {
                        size_t context_id = environment.get_context_index(i);
                        futures.push_back(std::async(std::launch::async, lambda_matmul, i, 
                            x_serialized[i], thread_helpers[i], context_automorphism_keys[context_id], 
                            context_w_encoded[context_id]
                        ));
                    }
                    for (size_t i = 0; i < threads; i++) {
                        auto [y_serialized_single, timer_single] = futures[i].get();
                        y_serialized.push_back(y_serialized_single);
                        timers.push_back(timer_single);
                    }
                }
                if (last_rep) {
                    thread_timer.finish("Server-Matmul");
                    if (timers.size() == 1) timers[0].print();
                    else TimerThreaded::Print(timers);
                }
                timers.clear();

                size_t y_serialized_size = 0;
                for (size_t i = 0; i < threads; i++) {
                    y_serialized_size += y_serialized[i].size();
                }

                thread_timer.restart();

                // decrypt and decode outputs
                GeneralVector y_decoded = GeneralVector::zeros_like(s, m * n);
                auto lambda_decrypt_decode = [this, m_divided, m, n, &y_decoded](
                    size_t thread_id, std::string y_serialized, const MatmulHelper& helper
                ) {
                    const GeneralHeContext& context = environment.get_context(thread_id);
                    const GeneralEncoder& encoder = context.encoder();
                    const Decryptor& decryptor = context.decryptor();
                    size_t m_lower = thread_id * m_divided;
                    size_t m_upper = std::min((thread_id + 1) * m_divided, m);

                    Timer timer; timer.tab(1);

                    size_t timer_single_handle = timer.register_timer("Deser [y]");
                    timer.tick(timer_single_handle);
                    stringstream y_serialized_stream(y_serialized);
                    Cipher2d y_encrypted = helper.deserialize_outputs(context.evaluator(), y_serialized_stream);
                    timer.tock(timer_single_handle);

                    timer_single_handle = timer.register_timer("Dec [y]");
                    timer.tick(timer_single_handle);
                    GeneralVector y_decoded_single(vector<double>({}));
                    if (encoder.is_batch()) {
                        y_decoded_single = GeneralVector(helper.decrypt_outputs_uint64s(encoder.batch(), decryptor, y_encrypted), false);
                    } else if (encoder.is_ckks()) {
                        y_decoded_single = helper.decrypt_outputs_doubles(encoder.ckks(), decryptor, y_encrypted);
                    } else if (encoder.is_ring32()) {
                        y_decoded_single = helper.decrypt_outputs_ring2k<uint32_t>(encoder.poly32(), decryptor, y_encrypted);
                    } else if (encoder.is_ring64()) {
                        y_decoded_single = GeneralVector(helper.decrypt_outputs_ring2k<uint64_t>(encoder.poly64(), decryptor, y_encrypted), true);
                    } else if (encoder.is_ring128()) {
                        y_decoded_single = helper.decrypt_outputs_ring2k<uint128_t>(encoder.poly128(), decryptor, y_encrypted);
                    } else {
                        throw std::runtime_error("Unsupported encoder");
                    }
                    // put to y_decoded
                    for (size_t i = m_lower; i < m_upper; i++) {
                        for (size_t j = 0; j < n; j++) {
                            if (encoder.is_batch()) {
                                y_decoded.integers()[i * n + j] = y_decoded_single.integers()[(i - m_lower) * n + j];
                            } else if (encoder.is_ckks()) {
                                y_decoded.doubles()[i * n + j] = y_decoded_single.doubles()[(i - m_lower) * n + j];
                            } else if (encoder.is_ring32()) {
                                y_decoded.uint32s()[i * n + j] = y_decoded_single.uint32s()[(i - m_lower) * n + j];
                            } else if (encoder.is_ring64()) {
                                y_decoded.uint64s()[i * n + j] = y_decoded_single.uint64s()[(i - m_lower) * n + j];
                            } else if (encoder.is_ring128()) {
                                y_decoded.uint128s()[i * n + j] = y_decoded_single.uint128s()[(i - m_lower) * n + j];
                            } else {
                                throw std::runtime_error("Unsupported encoder");
                            }
                        }
                    }
                    timer.tock(timer_single_handle);

                    return timer;
                };

                // run threads
                if (threads == 1) {
                    timers.push_back(lambda_decrypt_decode(0, y_serialized[0], thread_helpers[0]));
                } else {
                    std::vector<std::future<Timer>> futures;
                    for (size_t i = 0; i < threads; i++) {
                        futures.push_back(std::async(std::launch::async, lambda_decrypt_decode, i, y_serialized[i], thread_helpers[i]));
                    }
                    for (size_t i = 0; i < threads; i++) {
                        timers.push_back(futures[i].get());
                    }
                }
                if (last_rep) {
                    thread_timer.finish("Client-Dec[y]");
                    if (timers.size() == 1) timers[0].print();
                    else TimerThreaded::Print(timers);
                }
                timers.clear();

                total_timer.tock(total_timer_handle);
                if (args.repeat > 1) {
                    total_timer_once.tock(total_timer_once_handle);
                    total_timer_once.print();
                }

                if (rep == args.repeat - 1) {
                    std::cout << "Communication cost:\n";
                    std::cout << "  [x] = " << x_serialized_size << " bytes" << std::endl;
                    std::cout << "  [y] = " << y_serialized_size << " bytes" << std::endl;

                    if (args.no_check_correctness) {
                        success = true;
                    } else {
                        GeneralVector y_truth = matmul_plaintext(x, w, s, threads);
                        success = y_decoded.near_equal(y_truth, context.tolerance());
                    }
                }
            }

            total_timer.print_divided(args.repeat);

            if (!success) {
                std::cout << "Output incorrect!" << std::endl;
            }
            return success;
        }

    };

}

int main(int argc, char** argv) {
    
    using std::cout;
    using std::endl;
    using std::string;
    using bench::matmul::Arguments;
    Arguments args(argc, argv);
    if (args.help) {
        Arguments::print_help();
        return 0;
    } 

    args.print_arguments();

    bench::matmul::BenchmarkMatmul benchmark(args);
    bool success = benchmark.test_matmul();

    if (args.device) {
        troy::utils::MemoryPool::Destroy();
    }

    if (success) {
        return 0;
    } else {
        return 1;
    }
}