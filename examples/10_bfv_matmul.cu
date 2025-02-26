#include "examples.h"

using namespace std;
using namespace troy;
using namespace troy::linear;

static inline uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t t) {
    __uint128_t c = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(c % static_cast<__uint128_t>(t));
}

static inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t t) {
    if (a + b >= t) {
        return a + b - t;
    } else {
        return a + b;
    }
}

static inline void add_mod_inplace(uint64_t& a, uint64_t b, uint64_t t) {
    a = add_mod(a, b, t);
}

static std::vector<uint64_t> random_polynomial(size_t size, uint64_t max_value = 10) {
    std::vector<uint64_t> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = rand() % max_value;
    }
    return result;
}

static bool vector_equal(const vector<uint64_t>& a, const vector<uint64_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static void test_bfv_matmul(
    bool is_bgv,
    size_t poly_modulus_degree,
    const Modulus& plain_modulus,
    const vector<Modulus>& coeff_modulus,
    size_t m, size_t r, size_t n, bool pack_lwe, bool mod_switch_to_next
) {

    // We only support BFV and BGV for integer matmul.
    SchemeType scheme = is_bgv ? SchemeType::BGV : SchemeType::BFV;
    
    // Create the context.
    EncryptionParameters parms(scheme);
    parms.set_coeff_modulus(coeff_modulus);
    parms.set_plain_modulus(plain_modulus);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);

    // Create encoder and convey to GPU memory
    BatchEncoder encoder(he);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    }

    // Create other util classes
    KeyGenerator keygen(he);
    Encryptor encryptor(he); encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());
    GaloisKeys automorphism_key;
    // automorphism key is only used when pack lwe is set to true
    if (pack_lwe) automorphism_key = keygen.create_automorphism_keys(false);

    // Obtain the plain modulus. 
    uint64_t t = parms.plain_modulus_host().value();
    
    // Create matmul values. We aim to calculate [y] = [x] * w + s
    // In practice, usually `x` is held by one party Alice with the secret key,
    // and `w` and `s` are held by another party Bob which conducts the HE computation.
    // Therefore, `x` is encrypted and sent to Bob, where Bob calculates `[y]` and
    // sends it back to Alice for decryption.
    // Here we omit the simulation of two parties and only shows the interfaces for HE computation.
    vector<uint64_t> x = random_polynomial(m * r);
    vector<uint64_t> w = random_polynomial(r * n);
    vector<uint64_t> s = random_polynomial(m * n);
    MatmulHelper helper(m, r, n, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, pack_lwe);
    
    // Encode into plaintexts
    Plain2d w_encoded = helper.encode_weights_uint64s(encoder, w.data());
    Plain2d s_encoded = helper.encode_outputs_uint64s(encoder, s.data());

    // Alice encrypts the input `x`. Since we only set the
    // secret key for the encryptor, we can only use symmetric encryption.
    Cipher2d x_encrypted = helper.encrypt_inputs_uint64s(encryptor, encoder, x.data());

    // Alice serializes the ciphertexts.
    stringstream x_serialized;
    x_encrypted.save(x_serialized, he);
    std::cout << "x serialized size = " << x_serialized.str().size() << " bytes" << std::endl;
    // Bob loads the ciphertexts.
    x_encrypted = Cipher2d::load_new(x_serialized, he);

    // Matrix multiplication
    Cipher2d y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded);
    if (mod_switch_to_next) {
        y_encrypted.mod_switch_to_next_inplace(evaluator);
    }
    if (pack_lwe) {
        y_encrypted = helper.pack_outputs(evaluator, automorphism_key, y_encrypted);
    }

    // Adding bias `s` to the matrix multiplication result.
    y_encrypted.add_plain_inplace(evaluator, s_encoded);

    // Bob serializes the output ciphertexts.
    stringstream y_serialized;
    helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
    std::cout << "y serialized size = " << y_serialized.str().size() << " bytes" << std::endl;
    // Alice deserializes the output ciphertexts.
    y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

    // Alice decrypts the output ciphertexts.
    vector<uint64_t> y_decrypted = helper.decrypt_outputs_uint64s(encoder, decryptor, y_encrypted);   

    // A plaintext computation, to verify the correctness of the HE computation.
    vector<uint64_t> y_truth(m * n, 0);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < r; k++) {
                add_mod_inplace(y_truth[i * n + j], 
                    multiply_mod(
                        x[i * r + k], 
                        w[k * n + j],
                    t), t);
            }
            add_mod_inplace(y_truth[i * n + j], s[i * n + j], t);
        }
    }
    
    bool success = vector_equal(y_decrypted, y_truth);
    if (success) {
        std::cout << "Matmul test passed!" << std::endl;
    } else {
        std::cout << "Matmul test failed!" << std::endl;
    }
    std::cout << std::endl;
}

void example_bfv_matmul()
{

    print_example_banner("BFV/BGV matrix multiplication");

    // Test the matmul function with BFV scheme.
    // the plaintext modulus can be a non-prime here.

    // We first test without packing LWEs
    size_t poly_modulus_degree = 8192;
    test_bfv_matmul(
        false, poly_modulus_degree, 
        Modulus(1<<21), CoeffModulus::create(poly_modulus_degree, { 60, 40, 40, 60 }).to_vector(), 
        25, 30, 35, false, true
    );

    // We then test with packing LWEs
    // We expect the output `y` to haves a smaller serialized size, but
    // the computation time will be longer.
    test_bfv_matmul(
        false, poly_modulus_degree, 
        Modulus(1<<21), CoeffModulus::create(poly_modulus_degree, { 60, 40, 40, 60 }).to_vector(), 
        25, 30, 35, true, true
    );

}