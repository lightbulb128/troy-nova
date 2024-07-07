#include "examples.h"

using namespace troy;

void example_quickstart() {
    
    print_example_banner("Quickstart");
    
    // Setup encryption parameters.
    EncryptionParameters params(SchemeType::BFV);
    params.set_poly_modulus_degree(8192);
    params.set_coeff_modulus(CoeffModulus::create(8192, { 40, 40, 40 }));
    params.set_plain_modulus(PlainModulus::batching(8192, 20));

    // Create context and encoder
    HeContextPointer context = HeContext::create(params, true, SecurityLevel::Classical128);
    BatchEncoder encoder(context);

    // Convey them to the device memory.
    // The encoder must be conveyed to the device memory after creating it from a host-memory context.
    // i.e. you cannot create an encoder directly from a device-memory context.
    context->to_device_inplace();
    encoder.to_device_inplace();

    // Other utilities could directly be constructed from device-memory context.
    KeyGenerator keygen(context);
    PublicKey public_key = keygen.create_public_key(false);
    Encryptor encryptor(context); encryptor.set_public_key(public_key);
    Decryptor decryptor(context, keygen.secret_key());
    Evaluator evaluator(context);
    // Alternatively, you can create all of these (keygen, encryptor, etc.) 
    // on host memory and then convey them all to device memory at once.

    // Create plaintexts
    std::vector<uint64_t> message1 = { 1, 2, 3, 4 };
    Plaintext plain1 = encoder.encode_new(message1);
    std::vector<uint64_t> message2 = { 5, 6, 7, 8 };
    Plaintext plain2 = encoder.encode_new(message2);

    // Encrypt. Since we only set the public key, we can only use asymmetric encryption.
    Ciphertext encrypted1 = encryptor.encrypt_asymmetric_new(plain1);
    Ciphertext encrypted2 = encryptor.encrypt_asymmetric_new(plain2);

    // Add
    Ciphertext encrypted_sum = evaluator.add_new(encrypted1, encrypted2);

    // Decrypt and decode
    Plaintext decrypted_sum = decryptor.decrypt_new(encrypted_sum);
    std::vector<uint64_t> result = encoder.decode_new(decrypted_sum);

    // Check good?
    result.resize(message1.size());
    if (result == std::vector<uint64_t>({ 6, 8, 10, 12 })) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }

}