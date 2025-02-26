#include "../../src/app/conv2d.h"
#include "../test_adv.h"
#include "../../src/utils/timer.h"
#include "argument_helper.h"
#include "../argparse.h"
#include <cassert>

namespace bench::conv2d {

    using namespace troy;
    using namespace tool;
    using namespace troy::linear;
    using tool::GeneralEncoder;
    using tool::GeneralHeContext;
    using tool::GeneralVector;
    using std::stringstream;
    using std::vector;
    using troy::bench::Timer;
    using troy::bench::TimerOnce;

    struct Arguments {
        bool help = false;
        bool host = false;
        bool device = false;
        bool scheme_bfv = false;
        bool scheme_ckks = false;
        bool scheme_bgv = false;

        size_t repeat;

        size_t batch_size, input_channels, output_channels, image_height, image_width, kernel_height, kernel_width;

        size_t poly_modulus_degree = 8192;
        size_t simd_log_t = 40;
        size_t ring2k_log_t = 0;
        std::vector<size_t> log_q = { 60, 40, 40, 60 };
        uint64_t seed = 0x123;
        double input_max = 5;
        double scale = 1 << 30;
        double tolerance = 0.1;

        bool use_special_prime_for_encryption = false;
        size_t mod_switch_down_levels = 0;
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

            batch_size = parser.get_uint<size_t>("-bs").value_or(parser.get_uint<size_t>("--batch-size").value_or(4));
            input_channels = parser.get_uint<size_t>("-ic").value_or(parser.get_uint<size_t>("--input-channels").value_or(3));
            output_channels = parser.get_uint<size_t>("-oc").value_or(parser.get_uint<size_t>("--output-channels").value_or(16));
            image_height = parser.get_uint<size_t>("-ih").value_or(parser.get_uint<size_t>("--image-height").value_or(32));
            image_width = parser.get_uint<size_t>("-iw").value_or(parser.get_uint<size_t>("--image-width").value_or(32));
            kernel_height = parser.get_uint<size_t>("-kh").value_or(parser.get_uint<size_t>("--kernel-height").value_or(3));
            kernel_width = parser.get_uint<size_t>("-kw").value_or(parser.get_uint<size_t>("--kernel-width").value_or(3));

            poly_modulus_degree = parser.get_uint<size_t>("-N").value_or(parser.get_uint<size_t>("--poly-modulus-degree").value_or(poly_modulus_degree));
            simd_log_t = parser.get_uint<size_t>("-st").value_or(parser.get_uint<size_t>("--simd-log-t").value_or(simd_log_t));
            ring2k_log_t = parser.get_uint<size_t>("-rt").value_or(parser.get_uint<size_t>("--ring2k-log-t").value_or(ring2k_log_t));
            log_q = parser.get_uint_list<size_t>("-q").value_or(parser.get_uint_list<size_t>("--log-q").value_or(log_q));
            seed = parser.get_uint<uint64_t>("-S").value_or(parser.get_uint<uint64_t>("--seed").value_or(seed));
            input_max = parser.get_float<double>("-i").value_or(parser.get_float<double>("--input-max").value_or(input_max));
            scale = parser.get_float<double>("-s").value_or(parser.get_float<double>("--scale").value_or(scale));
            tolerance = parser.get_float<double>("-T").value_or(parser.get_float<double>("--tolerance").value_or(tolerance));

            use_special_prime_for_encryption = parser.get_bool_store_true("-usp").value_or(parser.get_bool_store_true("--use-special-prime-for-encryption").value_or(false));
            mod_switch_down_levels = parser.get_uint<size_t>("-msd").value_or(parser.get_uint<size_t>("--mod-switch-down-levels").value_or(0));
            use_zstd = parser.get_bool_store_true("--use-zstd").value_or(false);
            no_check_correctness = parser.get_bool_store_true("--no-check-correctness").value_or(false);

            if (ring2k_log_t > 64 && log_q == vector<size_t>{ 60, 40, 40, 60 }) {
                std::cout << "Warning: using uint128_t but default log_q. This might lead to wrong results. Setting to log_q = {60*6}" << std::endl;
                log_q = vector<size_t>{ 60, 60, 60, 60, 60, 60 };
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
            std::cout << "  -bs, --batch-size           Batch size (default: 4)" << std::endl;
            std::cout << "  -ic, --input-channels       Input channels (default: 3)" << std::endl;
            std::cout << "  -oc, --output-channels      Output channels (default: 16)" << std::endl;
            std::cout << "  -ih, --image-height         Image height (default: 32)" << std::endl;
            std::cout << "  -iw, --image-width          Image width (default: 32)" << std::endl;
            std::cout << "  -kh, --kernel-height        Kernel height (default: 3)" << std::endl;
            std::cout << "  -kw, --kernel-width         Kernel width (default: 3)" << std::endl;
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
            std::cout << "  -usp, --use-special-prime-for-encryption" << std::endl;
            std::cout << "                              Use special prime for encryption" << std::endl;
            std::cout << "  -msd, --mod-switch-down-levels" << std::endl;
            std::cout << "  --use-zstd                  Use Zstd for compressing serialized ciphers" << std::endl;
            std::cout << "  --no-check-correctness       Do not check correctness" << std::endl;
            std::cout << std::endl;
        }

        void print_arguments() {
            std::cout << "[Arguments]" << std::endl;
            if (host) {
                std::cout << "  run-on              = host" << std::endl;
            } else {
                std::cout << "  run-on              = device" << std::endl;
            }
            std::cout << "  dimensions" << std::endl;
            std::cout << "    batch-size        = " << batch_size << std::endl;
            std::cout << "    input-channels    = " << input_channels << std::endl;
            std::cout << "    output-channels   = " << output_channels << std::endl;
            std::cout << "    image-height      = " << image_height << std::endl;
            std::cout << "    image-width       = " << image_width << std::endl;
            std::cout << "    kernel-height     = " << kernel_height << std::endl;
            std::cout << "    kernel-width      = " << kernel_width << std::endl;
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

    class BenchmarkConv2d {

        Arguments args;
        std::shared_ptr<GeneralHeContext> environment;
        SchemeType scheme;
        
    public:

        GeneralVector conv2d_plaintext(const GeneralVector& lhs, const GeneralVector& rhs, const GeneralVector& bias) const {

            size_t bs = args.batch_size;
            size_t ic = args.input_channels;
            size_t oc = args.output_channels;
            size_t ih = args.image_height;
            size_t iw = args.image_width;
            size_t kh = args.kernel_height;
            size_t kw = args.kernel_width;
            size_t oh = ih - kh + 1;
            size_t ow = iw - kw + 1;

            GeneralVector output = GeneralVector::zeros_like(lhs, bs * oc * oh * ow);
            
            assert(lhs.size() == bs * ic * ih * iw);
            assert(rhs.size() == oc * ic * kh * kw);
            assert(bias.size() == bs * oc * oh * ow);

            uint64_t simd_t = environment->simd_t();
            uint128_t ring_t_mask = environment->ring_t_mask();

            vector<uint64_t> y_truth(bs * oc * oh * ow, 0);
            for (size_t b = 0; b < bs; b++) {
                for (size_t o = 0; o < oc; o++) {
                    for (size_t i = 0; i < oh; i++) {
                        for (size_t j = 0; j < ow; j++) {
                            size_t y_id = b * oc * oh * ow + o * oh * ow + i * ow + j;
                            for (size_t c = 0; c < ic; c++) {
                                for (size_t p = 0; p < kh; p++) {
                                    for (size_t q = 0; q < kw; q++) {
                                        size_t x_id = b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q);
                                        size_t w_id = o * ic * kh * kw + c * kh * kw + p * kw + q;
                                        if (lhs.is_complexes()) {
                                            output.complexes()[y_id] += lhs.complexes()[x_id] * rhs.complexes()[w_id];
                                        } else if (lhs.is_doubles()) {
                                            output.doubles()[y_id] += lhs.doubles()[x_id] * rhs.doubles()[w_id];
                                        } else if (lhs.is_integers()) {
                                            uint128_t mult = static_cast<uint128_t>(lhs.integers()[x_id]) * static_cast<uint128_t>(rhs.integers()[w_id]);
                                            mult = mult % static_cast<uint128_t>(simd_t);
                                            output.integers()[y_id] = (output.integers()[y_id] + static_cast<uint64_t>(mult)) % simd_t;
                                        } else if (lhs.is_uint32s()) {
                                            output.uint32s()[y_id] += lhs.uint32s()[x_id] * rhs.uint32s()[w_id];
                                            output.uint32s()[y_id] &= ring_t_mask;
                                        } else if (lhs.is_uint64s()) {
                                            output.uint64s()[y_id] += lhs.uint64s()[x_id] * rhs.uint64s()[w_id];
                                            output.uint64s()[y_id] &= ring_t_mask;
                                        } else if (lhs.is_uint128s()) {
                                            output.uint128s()[y_id] += lhs.uint128s()[x_id] * rhs.uint128s()[w_id];
                                            output.uint128s()[y_id] &= ring_t_mask;
                                        } else {
                                            throw std::runtime_error("Unsupported data type");
                                        }

                                    }
                                }
                            }
                            if (lhs.is_complexes()) {
                                output.complexes()[y_id] += bias.complexes()[y_id];
                            } else if (lhs.is_doubles()) {
                                output.doubles()[y_id] += bias.doubles()[y_id];
                            } else if (lhs.is_integers()) {
                                output.integers()[y_id] = (output.integers()[y_id] + bias.integers()[y_id]) % simd_t;
                            } else if (lhs.is_uint32s()) {
                                output.uint32s()[y_id] += bias.uint32s()[y_id];
                                output.uint32s()[y_id] &= ring_t_mask;
                            } else if (lhs.is_uint64s()) {
                                output.uint64s()[y_id] += bias.uint64s()[y_id];
                                output.uint64s()[y_id] &= ring_t_mask;
                            } else if (lhs.is_uint128s()) {
                                output.uint128s()[y_id] += bias.uint128s()[y_id];
                                output.uint128s()[y_id] &= ring_t_mask;
                            } else {
                                throw std::runtime_error("Unsupported data type");
                            }
                        }
                    }
                }
            }

            return output;
        }
        
        BenchmarkConv2d(Arguments args): args(args) {
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
            environment = std::make_shared<GeneralHeContext>(ghep);
        }

        bool test_conv2d() const {

            size_t bs = args.batch_size;
            size_t ic = args.input_channels;
            size_t oc = args.output_channels;
            size_t ih = args.image_height;
            size_t iw = args.image_width;
            size_t kh = args.kernel_height;
            size_t kw = args.kernel_width;
            size_t oh = ih - kh + 1;
            size_t ow = iw - kw + 1;

            // generate input data
            const GeneralHeContext& context = *environment;
            GeneralVector x = context.random_polynomial(bs * ic * ih * iw);
            GeneralVector w = context.random_polynomial(oc * ic * kh * kw);
            GeneralVector s = context.random_polynomial(bs * oc * oh * ow);

            // create helper
            Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw, context.params_host().poly_modulus_degree(), MatmulObjective::EncryptLeft);

            // encode w_encoded
            const GeneralEncoder& encoder = context.encoder();
            const Encryptor& encryptor = context.encryptor();
            const Decryptor& decryptor = context.decryptor();
            const Evaluator& evaluator = context.evaluator();
            TimerOnce timer;
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
            timer.finish("Ecd w");

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

                Timer total_timer_once; 
                size_t total_timer_once_handle = 0;
                if (args.repeat > 1) {
                    total_timer_once_handle = total_timer_once.register_timer(std::string("Time cost #") + std::to_string(rep + 1));
                    total_timer_once.tick(total_timer_once_handle);
                }

                Timer timer; timer.tab(1);
                TimerOnce block_timer;

                size_t timer_single_handle = timer.register_timer("Ecd/enc x");
                timer.tick(timer_single_handle);
                Cipher2d x_encrypted;
                if (encoder.is_batch()) {
                    x_encrypted = helper.encrypt_inputs_uint64s(encryptor, encoder.batch(), x.integers().data());
                } else if (encoder.is_ckks()) {
                    x_encrypted = helper.encrypt_inputs_doubles(encryptor, encoder.ckks(), x.doubles().data(), std::nullopt, context.scale());
                } else if (encoder.is_ring32()) {
                    x_encrypted = helper.encrypt_inputs_ring2k<uint32_t>(encryptor, encoder.poly32(), x.uint32s().data(), std::nullopt);
                } else if (encoder.is_ring64()) {
                    x_encrypted = helper.encrypt_inputs_ring2k<uint64_t>(encryptor, encoder.poly64(), x.uint64s().data(), std::nullopt);
                } else if (encoder.is_ring128()) {
                    x_encrypted = helper.encrypt_inputs_ring2k<uint128_t>(encryptor, encoder.poly128(), x.uint128s().data(), std::nullopt);
                } else {
                    throw std::runtime_error("Unsupported encoder");
                }
                timer.tock(timer_single_handle);

                std::string x_serialized;
                {
                    stringstream x_serialized_stream;
                    timer_single_handle = timer.register_timer("Ser [x]");
                    timer.tick(timer_single_handle);
                    x_encrypted.save(x_serialized_stream, context.context(), this->args.use_zstd ? CompressionMode::Zstd : CompressionMode::Nil);
                    timer.tock(timer_single_handle);
                    x_serialized = x_serialized_stream.str();
                }

                if (last_rep) {
                    block_timer.finish("Client-Enc[x]");
                    timer.print();
                }
                timer.clear();

                size_t x_serialized_size = x_serialized.size();


                block_timer.restart();

                timer = Timer(); timer.tab(1);
                    
                timer_single_handle = timer.register_timer("Ecd s");
                timer.tick(timer_single_handle);
                Plain2d s_encoded;
                if (encoder.is_batch()) {
                    s_encoded = helper.encode_outputs_uint64s(encoder.batch(), s.integers().data());
                } else if (encoder.is_ckks()) {
                    s_encoded = helper.encode_outputs_doubles(encoder.ckks(), s.doubles().data(), std::nullopt, context.scale() * context.scale());
                } else if (encoder.is_ring32()) {
                    s_encoded = helper.encode_outputs_ring2k<uint32_t>(encoder.poly32(), s.uint32s().data(), s_parms_id);
                } else if (encoder.is_ring64()) {
                    s_encoded = helper.encode_outputs_ring2k<uint64_t>(encoder.poly64(), s.uint64s().data(), s_parms_id);
                } else if (encoder.is_ring128()) {
                    s_encoded = helper.encode_outputs_ring2k<uint128_t>(encoder.poly128(), s.uint128s().data(), s_parms_id);
                } else {
                    throw std::runtime_error("Unsupported encoder");
                }
                timer.tock(timer_single_handle);

                timer_single_handle = timer.register_timer("Deser [x]");
                timer.tick(timer_single_handle);
                stringstream x_serialized_stream(x_serialized);
                x_encrypted = Cipher2d::load_new(x_serialized_stream, context.context());
                timer.tock(timer_single_handle);
                
                timer_single_handle = timer.register_timer("Matmul");
                timer.tick(timer_single_handle);
                Cipher2d y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded);
                timer.tock(timer_single_handle);
                
                if (mod_switch_down_levels > 0) {
                    timer_single_handle = timer.register_timer("Mod switch");
                    timer.tick(timer_single_handle);
                    for (size_t i = 0; i < mod_switch_down_levels; i++) {
                        y_encrypted.mod_switch_to_next_inplace(evaluator);
                    }
                    timer.tock(timer_single_handle);
                }

                timer_single_handle = timer.register_timer("Add s");
                timer.tick(timer_single_handle);
                y_encrypted.add_plain_inplace(evaluator, s_encoded);
                timer.tock(timer_single_handle);

                std::string y_serialized;
                {
                    stringstream y_serialized_stream;
                    timer_single_handle = timer.register_timer("Ser [y]");
                    timer.tick(timer_single_handle);
                    helper.serialize_outputs(evaluator, y_encrypted, y_serialized_stream, this->args.use_zstd ? CompressionMode::Zstd : CompressionMode::Nil);
                    timer.tock(timer_single_handle);
                    y_serialized = y_serialized_stream.str();
                }

                if (last_rep) {
                    block_timer.finish("Server-Matmul");
                    timer.print();
                }
                timer.clear();

                size_t y_serialized_size = y_serialized_size = y_serialized.size();

                block_timer.restart();

                // decrypt and decode outputs
                GeneralVector y_decoded = GeneralVector::zeros_like(s, bs * oc * oh * ow);

                timer = Timer(); timer.tab(1);

                timer_single_handle = timer.register_timer("Deser [y]");
                timer.tick(timer_single_handle);
                stringstream y_serialized_stream(y_serialized);
                y_encrypted = helper.deserialize_outputs(context.evaluator(), y_serialized_stream);
                timer.tock(timer_single_handle);

                timer_single_handle = timer.register_timer("Dec [y]");
                timer.tick(timer_single_handle);
                GeneralVector y_decoded_single(vector<double>({}));
                if (encoder.is_batch()) {
                    y_decoded = GeneralVector(helper.decrypt_outputs_uint64s(encoder.batch(), decryptor, y_encrypted), false);
                } else if (encoder.is_ckks()) {
                    y_decoded = helper.decrypt_outputs_doubles(encoder.ckks(), decryptor, y_encrypted);
                } else if (encoder.is_ring32()) {
                    y_decoded = helper.decrypt_outputs_ring2k<uint32_t>(encoder.poly32(), decryptor, y_encrypted);
                } else if (encoder.is_ring64()) {
                    y_decoded = GeneralVector(helper.decrypt_outputs_ring2k<uint64_t>(encoder.poly64(), decryptor, y_encrypted), true);
                } else if (encoder.is_ring128()) {
                    y_decoded = helper.decrypt_outputs_ring2k<uint128_t>(encoder.poly128(), decryptor, y_encrypted);
                } else {
                    throw std::runtime_error("Unsupported encoder");
                }
                timer.tock(timer_single_handle);

                if (last_rep) {
                    block_timer.finish("Client-Dec[y]");
                    timer.print();
                }
                timer.clear();

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
                        GeneralVector y_truth = conv2d_plaintext(x, w, s);
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
    using bench::conv2d::Arguments;
    Arguments args(argc, argv);
    if (args.help) {
        Arguments::print_help();
        return 0;
    } 

    args.print_arguments();

    bench::conv2d::BenchmarkConv2d benchmark(args);
    bool success = benchmark.test_conv2d();

    if (args.device) {
        troy::utils::MemoryPool::Destroy();
    }

    if (success) {
        return 0;
    } else {
        return 1;
    }
}