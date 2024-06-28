#include "../../src/app/matmul.cuh"
#include "../test_adv.cuh"
#include "../../src/utils/timer.h"

using namespace troy;
using namespace troy::linear;
using tool::GeneralEncoder;
using tool::GeneralHeContext;
using tool::GeneralVector;
using std::stringstream;
using std::vector;
using troy::bench::TimerOnce;

void test_matmul(const GeneralHeContext& context, size_t m, size_t r, size_t n, bool pack_lwe, bool mod_switch_to_next) {
    
    SchemeType scheme = context.params_host().scheme();
    if (scheme != SchemeType::BFV && scheme != SchemeType::BGV) {
        throw std::runtime_error("[test_matmul] Unsupported scheme");
    }
    uint64_t t = context.t();
    
    GeneralVector x = context.random_polynomial(m * r);
    GeneralVector w = context.random_polynomial(r * n);
    GeneralVector s = context.random_polynomial(m * n);
    MatmulHelper helper(m, r, n, context.params_host().poly_modulus_degree(), MatmulObjective::EncryptLeft, pack_lwe);

    HeContextPointer he = context.context();
    const BatchEncoder& encoder = context.encoder().batch();
    const Encryptor& encryptor = context.encryptor();
    const Evaluator& evaluator = context.evaluator();
    const Decryptor& decryptor = context.decryptor();
    GaloisKeys automorphism_key;
    if (pack_lwe) {
        automorphism_key = context.key_generator().create_automorphism_keys(false);
    }

    TimerOnce timer;
    TimerOnce timer_total;
    
    timer_total.restart();
    timer.restart();
    Plain2d x_encoded = helper.encode_inputs_uint64s(encoder, x.integers().data());
    timer.finish("Encode inputs");

    timer.restart();
    Plain2d w_encoded = helper.encode_weights_uint64s(encoder, w.integers().data());
    timer.finish("Encode weights");

    timer.restart();
    Plain2d s_encoded = helper.encode_outputs_uint64s(encoder, s.integers().data());
    timer.finish("Encode bias");

    timer.restart();
    Cipher2d x_encrypted = x_encoded.encrypt_symmetric(encryptor);
    timer.finish("Encrypt inputs");

    stringstream x_serialized;
    timer.restart();
    x_encrypted.save(x_serialized, he);
    timer.finish("Serialize inputs");
    
    timer.restart();
    x_encrypted = Cipher2d::load_new(x_serialized, he);
    timer.finish("Deserialize inputs");

    timer.restart();
    Cipher2d y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded);
    timer.finish("Matmul");
    if (mod_switch_to_next) {
        timer.restart();
        y_encrypted.mod_switch_to_next_inplace(evaluator);
        timer.finish("Mod switch");
    }
    if (pack_lwe) {
        timer.restart();
        y_encrypted = helper.pack_outputs(evaluator, automorphism_key, y_encrypted);
        timer.finish("Pack outputs");
    }

    timer.restart();
    y_encrypted.add_plain_inplace(evaluator, s_encoded);
    timer.finish("Add bias");

    stringstream y_serialized;
    timer.restart();
    helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
    timer.finish("Serialize outputs");

    timer.restart();
    y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);
    timer.finish("Deserialize outputs");

    timer.restart();
    auto y_decrypted = helper.decrypt_outputs_uint64s(encoder, decryptor, y_encrypted);
    timer.finish("Decrypt outputs");

    timer_total.finish("Total");
}

std::string bool_to_string(bool b) {
    return b ? "true" : "false";
}

int main(int argc, char** argv) {
    // require m, r, n arguments at least
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " m r n [pack_lwe] [mod_switch_to_next] [poly_modulus_degree]" << std::endl;
        return 1;
    }
    size_t m = std::stoi(argv[1]);
    size_t r = std::stoi(argv[2]);
    size_t n = std::stoi(argv[3]);
    bool pack_lwe = false;
    bool mod_switch_to_next = false;
    if (argc >= 5) {
        pack_lwe = std::stoi(argv[4]);
    }
    if (argc >= 6) {
        mod_switch_to_next = std::stoi(argv[5]);
    }
    size_t poly_modulus_degree = 8192;
    if (argc >= 7) {
        poly_modulus_degree = std::stoi(argv[6]);
    }
    std::cout << "[Arguments]" << std::endl;
    std::cout << "  m = " << m << std::endl;
    std::cout << "  r = " << r << std::endl;
    std::cout << "  n = " << n << std::endl;
    std::cout << "  pack_lwe = " << bool_to_string(pack_lwe) << std::endl;
    std::cout << "  mod_switch_to_next = " << bool_to_string(mod_switch_to_next) << std::endl;
    
    GeneralHeContext ghe(true, SchemeType::BFV, poly_modulus_degree, 40, { 60, 60, 60 }, true, 0x123, 0);
    test_matmul(ghe, m, r, n, pack_lwe, mod_switch_to_next);

    utils::MemoryPool::Destroy();

    return 0;
}