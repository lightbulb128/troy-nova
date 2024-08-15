#include "examples.h"

using namespace std;
using namespace troy;

void example_serialization()
{
    print_example_banner("Example: Serialization");

    // We show how to serialize public keys and ciphertexts.
    // These operations are similar in CKKS and BFV/BGV. We only show BFV.

    EncryptionParameters parms(SchemeType::BFV);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::bfv_default(poly_modulus_degree, SecurityLevel::Classical128));
    parms.set_plain_modulus(PlainModulus::batching(poly_modulus_degree, 20));
    
    auto context = HeContext::create(parms, true, SecurityLevel::Classical128);
    BatchEncoder batch_encoder(context);

    // Convey to device
    if (utils::device_count() > 0) {
        context->to_device_inplace();
        batch_encoder.to_device_inplace();
    }



    // Public key
    // If a public key, relin keys or galois keys are to
    // be serialized, you can choose to save its seed to save
    // some transmission.

    KeyGenerator keygen(context);
    PublicKey public_key_with_seed = keygen.create_public_key(true);
    PublicKey public_key_without_seed = keygen.create_public_key(false);
    stringstream pk_stream1;

    // Saving the public key requires the knowledge of the HE context.
    public_key_with_seed.save(pk_stream1, context);
    std::cout << "PublicKey with seed size                = " << pk_stream1.str().size() << " bytes" << std::endl;
    stringstream pk_stream2;
    public_key_without_seed.save(pk_stream2, context);
    std::cout << "PublicKey without seed size             = " << pk_stream2.str().size() << " bytes" << std::endl;
    
    // To deserialize, the HE context is also required.
    // If a public key with a seed is loaded, the seed is expanded automatically when
    // deserializing.
    PublicKey deserialized;
    deserialized.load(pk_stream1, context);

    // The ser/deserialization of RelinKeys and GaloisKeys are similar.

    

    // Ciphertext
    // For ciphertext, you could choose to encrypt asymmetrically (which requires 
    // the public key), or symmetrically (which requires the secret key).
    // The ciphertexts encrypted symmetrically could contain a seed to save
    // transmission bytes, similar to public keys.
    
    Encryptor encryptor(context); 
    
    // Note: a public key with a seed could not be directly used to encryptor/evaluator.
    // The seed must be expanded first. You could call `public_key.expand_seed` to do this.
    encryptor.set_public_key(public_key_without_seed);
    // Now we can encrypt asymmetrically.
    Plaintext plain = batch_encoder.encode_new({1, 2, 3, 4});
    Ciphertext encrypted = encryptor.encrypt_asymmetric_new(plain);

    stringstream ct_stream1;
    encrypted.save(ct_stream1, context);
    std::cout << "Ciphertext asymmetrical size            = " << ct_stream1.str().size() << " bytes" << std::endl;

    // Symmetric encryption
    SecretKey secret_key = keygen.secret_key();
    encryptor.set_secret_key(secret_key);
    // We symmetrically encrypt and save the seed.
    // Note that a ciphertext with a seed could not be used by the evaluator
    // for HE operations directly. The seed must be expanded first.
    // When it is deserialized, the seed is automatically expanded.
    Ciphertext encrypted_symmetric = encryptor.encrypt_symmetric_new(plain, true);
    stringstream ct_stream2;
    encrypted_symmetric.save(ct_stream2, context);
    std::cout << "Ciphertext with seed size               = " << ct_stream2.str().size() << " bytes" << std::endl;

    // To deserialize, the HE context is also required.
    Ciphertext deserialized_ct;
    deserialized_ct.load(ct_stream1, context);

    // Now we see encrypting without seed has the same size as an asymmetrically encrypted ciphertext.
    // Also, same to the seed-expanded ciphertext.
    Ciphertext encrypted_symmetric_no_seed = encryptor.encrypt_symmetric_new(plain, false);
    stringstream ct_stream3;
    encrypted_symmetric_no_seed.save(ct_stream3, context);
    std::cout << "Ciphertext without seed size            = " << ct_stream3.str().size() << " bytes" << std::endl;
    encrypted_symmetric.expand_seed(context);
    stringstream ct_stream4;
    encrypted_symmetric.save(ct_stream4, context);
    std::cout << "Ciphertext expanded size                = " << ct_stream4.str().size() << " bytes" << std::endl;





    if (troy::utils::compression::available(CompressionMode::Zstd)) {
        // Optionally, one can use Zstandard to compress the serialized objects,
        // by providing the compression mode to the save functions. The load functions will automatically
        // decompress it.
        // This feature requires that you provide "TROY_ZSTD" in the CMake configuration,
        // which is on by default.
        stringstream ct_stream5;
        encrypted.save(ct_stream5, context, CompressionMode::Zstd);
        // We can see this has a smaller size.
        std::cout << "Ciphertext asymmetrical compressed size = " << ct_stream5.str().size() << " bytes" << std::endl;
    }

    std::cout << std::endl;
    
}
