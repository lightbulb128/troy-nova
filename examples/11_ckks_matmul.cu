#include "examples.h"

using namespace std;
using namespace troy;
using namespace troy::linear;

std::vector<double> random_polynomial(size_t size, double bound = 5) {
    std::vector<double> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = (rand() / (double)RAND_MAX * bound * 2) - bound;
    }
    return result;
}

bool vector_equal(const vector<double>& a, const vector<double>& b, double tolerance = 1e-3) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (abs(a[i] - b[i]) > tolerance) return false;
    }
    return true;
}

void test_ckks_matmul(
    size_t poly_modulus_degree,
    const vector<Modulus>& coeff_modulus,
    double scale,
    size_t m, size_t r, size_t n, bool pack_lwe, bool mod_switch_to_next
) {

    // Create the context.
    EncryptionParameters parms(SchemeType::CKKS);
    parms.set_coeff_modulus(coeff_modulus);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);

    // Create encoder and convey to GPU memory
    CKKSEncoder encoder(he);
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
    
    // Create matmul values. We aim to calculate [y] = [x] * w + s
    // In practice, usually `x` is held by one party Alice with the secret key,
    // and `w` and `s` are held by another party Bob which conducts the HE computation.
    // Therefore, `x` is encrypted and sent to Bob, where Bob calculates `[y]` and
    // sends it back to Alice for decryption.
    // Here we omit the simulation of two parties and only shows the interfaces for HE computation.
    vector<double> x = random_polynomial(m * r);
    vector<double> w = random_polynomial(r * n);
    vector<double> s = random_polynomial(m * n);
    MatmulHelper helper(m, r, n, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, pack_lwe);
    
    // Encode into plaintexts
    Plain2d w_encoded = helper.encode_weights_doubles(encoder, w.data(), std::nullopt, scale);

    // As w*x will have a doubled scale, we directly encode `s` on scale^2.
    Plain2d s_encoded = helper.encode_outputs_doubles(encoder, s.data(), std::nullopt, scale * scale);

    // Alice encrypts the input `x`. Since we only set the
    // secret key for the encryptor, we can only use symmetric encryption.
    Cipher2d x_encrypted = helper.encrypt_inputs_doubles(encryptor, encoder, x.data(), std::nullopt, scale);

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
    vector<double> y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);   

    // A plaintext computation, to verify the correctness of the HE computation.
    vector<double> y_truth(m * n, 0);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < r; k++) {
                y_truth[i * n + j] += x[i * r + k] * w[k * n + j];
            }
            y_truth[i * n + j] += s[i * n + j];
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

void example_ckks_matmul()
{

    print_example_banner("CKKS matrix multiplication");

    // Test the matmul function with BFV scheme.
    // the plaintext modulus can be a non-prime here.

    // We first test without packing LWEs
    size_t poly_modulus_degree = 8192;
    double scale = static_cast<double>(1<<20);
    test_ckks_matmul(
        poly_modulus_degree, 
        CoeffModulus::create(poly_modulus_degree, { 60, 40, 40, 60 }).to_vector(), 
        scale,
        25, 30, 35, false, true
    );

    // We then test with packing LWEs
    // We expect the output `y` to haves a smaller serialized size, but
    // the computation time will be longer.
    test_ckks_matmul(
        poly_modulus_degree, 
        CoeffModulus::create(poly_modulus_degree, { 60, 40, 40, 60 }).to_vector(), 
        scale,
        25, 30, 35, true, true
    );

}