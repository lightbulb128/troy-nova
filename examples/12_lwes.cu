#include "examples.h"

using namespace std;
using namespace troy;

void example_lwes() {

    print_example_banner("LWEs");

    // Note: The APIs demonstrated below are subject to change in future versions.

    // Reference
    //   Chen et al., Efficient Homomorphic Conversion Between (Ring) LWE Ciphertext
    //   https://eprint.iacr.org/2020/015.pdf

    // The BFV, CKKS and BGV schemes are based on Ring-Learning with Error problems.
    // Omitting all the details, this means that all the ciphertexts consist of
    // (at least) two polynomials, encrypting N (polynomial degree) coefficients of
    // a plaintext polynomial.

    // Actually, one can "extract" LWE ciphertexts from a RLWE ciphertext, and
    // each LWE ciphertext could represent a single coefficient of the plaintext polynomial.
    // A LWE ciphertext typically consists of a complete polynomial from the RLWE ciphertext,
    // and one extra coefficient containing the information exclusively for the
    // one coefficient in question.
    
    // More excitingly, multiple LWE iphertexts (probably extracted from different RLWE 
    // ciphertexts) could be packed into one RLWE ciphertext. If we only require some coefficients
    // of the RLWE results, we could consider using this extraction and packing technique
    // to save communication cost. The below example demonstrates how to extract and pack LWEs.
    // We use BFV for example, but the technique is also applicable to CKKS and BGV.


    // Create the context and utility classes.
    size_t poly_degree = 8192;
    SchemeType scheme = SchemeType::BFV;
    EncryptionParameters parms(scheme);
    parms.set_poly_modulus_degree(poly_degree);
    parms.set_plain_modulus(PlainModulus::batching(poly_degree, 30));
    parms.set_coeff_modulus(CoeffModulus::create(poly_degree, { 60, 40, 40, 60 }));
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);
    BatchEncoder encoder(he);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    }
    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Decryptor decryptor(he, keygen.secret_key());
    Evaluator evaluator(he);

    // For LWE packing, we neet to create an additional GaloisKey called the
    // automorphism key.
    GaloisKeys automorphism_key = keygen.create_automorphism_keys(false);

    
    // Create a ciphertext encrypting some polynomial.
    vector<uint64_t> plain_coefficients = { 1, 2, 3, 4, 5, 6, 7, 8 };
    Plaintext plaintext = encoder.encode_polynomial_new(plain_coefficients);
    Ciphertext ciphertext = encryptor.encrypt_symmetric_new(plaintext, false);


    // Extract the coefficient on index 2 (which is 3).
    LWECiphertext lwe = evaluator.extract_lwe_new(ciphertext, 2);

    
    // We don't provide the directly utility API to decrypt a LWECiphertext,
    // but you can use the assemble LWE method to convert a single LWECiphertext
    // back to a RLWE ciphertext. The coefficient will be put on the constant
    // term of the polynomial; other terms will be random meaningless values.
    Ciphertext assembled = evaluator.assemble_lwe_new(lwe);
    Plaintext decrypted = decryptor.decrypt_new(assembled);
    vector<uint64_t> decoded = encoder.decode_polynomial_new(decrypted);
    custom_assert(decoded[0] == 3, "Decoded value should be 3");


    // Now we show LWE packing. For a RLWE system with polynomial degree N,
    // one can pack at most N LWE ciphertexts into one RLWE ciphertext.
    // Say, we pack n <= N LWEs, (c[0], c[1], ..., c[n-1]) into a RLWE ciphertext.
    // Denote k is the least 2-power such that k >= n, the resulting RLWE ciphertext's
    // corresponding polynomial will have c[i]'s corresponding coefficient
    // placed at every (N/k*i)-th coefficient.
    // For example, if we pack {1, 2, 3, 4} LWEs into an N=16 RLWE, we will get
    //    [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0];
    // If we pack {1, 2, 3, 4, 5, 6} (k = 8) LWEs into an N=16 RLWE, we will get
    //    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 0, 0, 0, 0].

    // Here we extract from the same ciphertext and pack them back;
    // but extracting from different ciphertexts and packing them into one
    // will also work.



    // First try packing 4 items.
    vector<LWECiphertext> lwes;
    for (size_t i = 0; i < 4; i++) {
        lwes.push_back(evaluator.extract_lwe_new(ciphertext, i));
    }
    Ciphertext packed = evaluator.pack_lwe_ciphertexts_new(lwes, automorphism_key);
    decrypted = decryptor.decrypt_new(packed);
    decoded = encoder.decode_polynomial_new(decrypted);

    // Expect {1, 2, 3, 4} placed at every (N/4)-th coefficients.
    vector<uint64_t> truth(poly_degree, 0);
    size_t interval = poly_degree / 4;
    for (size_t i = 0; i < 4; i++) {
        truth[interval * i] = plain_coefficients[i];
    }
    custom_assert(decoded == truth, "Decoded value is not correct for 4 packing.");
    
    
    // Then try packing 6 items.
    // Note that, the more items you pack into a RLWE ciphertext, the more
    // time it will take. Again, you can at most pack N items into one RLWE ciphertext.
    lwes.clear();
    for (size_t i = 0; i < 6; i++) {
        lwes.push_back(evaluator.extract_lwe_new(ciphertext, i));
    }
    packed = evaluator.pack_lwe_ciphertexts_new(lwes, automorphism_key);
    decrypted = decryptor.decrypt_new(packed);
    decoded = encoder.decode_polynomial_new(decrypted);

    // Expect {1, 2, 3, 4, 5, 6} placed at every (N/8)-th coefficients.
    truth = vector<uint64_t>(poly_degree, 0);
    interval = poly_degree / 8;
    for (size_t i = 0; i < 6; i++) {
        truth[interval * i] = plain_coefficients[i];
    }
    custom_assert(decoded == truth, "Decoded value is not correct for 6 packing.");


    std::cout << "Example finished without errors." << std::endl << std::endl;


}