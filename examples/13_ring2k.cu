#include "examples.h"

using namespace std;
using namespace troy;

void example_ring2k() {
    
    print_example_banner("Ring2k");

    // Note: The APIs demonstrated below are subject to change in future versions.

    // In basic APIs for BFV, we only allowed the user to create the context
    // with plaintext modulus no more than 60 bits. We introduce a larger plaintext
    // space with partial support for the BFV scheme:
    //   - The space is expanded to 2^k, where k <= 128.
    //   - Only BFV allowed.
    //   - Only plaintext-ciphertext multiplication allowed, no cipher-cipher multiplication.
    //     Therefore, relinearization is meaningless, but LWE packing could work (for matrix multiplication).
    //   - No SIMD encoding, only polynomial encoding.
    //     Therefore, without SIMD, rotation is meaningless.

    // We demonstrate how to use this expanded ciphertext space with
    // an example of plain-cipher polynomial multiplication.

    
    size_t poly_degree = 16384;
    SchemeType scheme = SchemeType::BFV;
    EncryptionParameters parms(scheme);
    parms.set_poly_modulus_degree(poly_degree);
    // We need to explicitly call `set_plain_modulus` with a dummy number, because
    // creating the HE context requires it to be set,
    // but this will not be used in the following computation.
    parms.set_plain_modulus(1 << 20);
    parms.set_coeff_modulus(CoeffModulus::create(poly_degree, { 60, 60, 60, 60 }));
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);

    // The `k` in `Ring 2^k` semantics. Could be anywhere between 17 and 128, inclusive.
    size_t plain_bits = 64; 
    
    // We need to use a different encoder for the semantics.
    // The template parameter could only be one of `uint32_t`, `uint64_t`, `uint128_t`,
    // and the given `plain_bits` must be no less than half of the type's bit width.
    // That is, uint32_t supports 17~32 bits, uint64_t supports 33~64 bits, and so on.
    linear::PolynomialEncoderRing2k<uint64_t> encoder(he, plain_bits);

    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    }

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Decryptor decryptor(he, keygen.secret_key());
    Evaluator evaluator(he);

    // Computation
    vector<uint64_t> lhs = { 1, 2, 3 };
    vector<uint64_t> rhs = { 4, 5, 6 };

    // For cipher-plain multiplication, the cipher should be encoded with `scale_up`, while
    // the plain should be encoded with `centralize`. You can additionally provide
    // a parms_id to tell the encoder which level it should encode to. If not provided,
    // will use the first level.
    Plaintext plain_lhs = encoder.scale_up_new(lhs, std::nullopt);
    Plaintext plain_rhs = encoder.centralize_new(rhs, std::nullopt);

    // Encryption is same as vanilla BFV.
    Ciphertext cipher_lhs = encryptor.encrypt_symmetric_new(plain_lhs, false);

    // Cipher-plain multiplication.
    Ciphertext cipher_result = evaluator.multiply_plain_new(cipher_lhs, plain_rhs);

    // Decryption should use explicitly `bfv_decrypt_without_scaling_down`.
    Plaintext decrypted = decryptor.bfv_decrypt_without_scaling_down_new(cipher_result);

    // And we decode using `scale_down`.
    vector<uint64_t> decoded = encoder.scale_down_new(decrypted);

    // Check the result.
    vector<uint64_t> truth = { 4, 13, 28, 27, 18 };
    truth.resize(decoded.size(), 0);
    custom_assert(decoded == truth, "Decoded value incorrect");

    std::cout << "Example finished without errors." << std::endl << std::endl;

    // For those interested, here is some explanation of the computation.

    // In vanilla BFV, when a vector is encoded into a plaintext (SIMD or polynomial), it is not scaled up (i.e. multiplied by Delta = Q/t).
    // When the plaintext is encrypted, the scaling up is performed. Therefore, we see the plaintext is without a `parms_id`, because
    // it is not in the ciphertext modulus chain, but with modulus `t`. When we conduct cipher-cipher multiplication, the two ciphertexts'
    // products will have doubled scale `Delta^2`, so BFV designs ways to reduce one `Delta`. The infos related to this reduction
    // is stored in ContextData in this library. When we conduct plain-cipher multiplication, the plain is not scaled up but
    // directly conveyed to `mod Q` space by an operation I call `centralizing`. It simply put the `mod t` space in the center of 
    // the `mod Q` space, instead of scaling up. As plaintexts are stored with uint64_ts, naturally they cannot hold coefficients
    // in a `mod t` ring larger than 64 bits.

    // In Ring2k BFV, however, in order to store the coefficients in a `mod t = 2^k` ring, we have to carry out scaling up or
    // centralizing beforehand. This is why you need to provide a `parms_id` to decide which level to put the plaintext at
    // in the modulus chain. And you need to handle which plaintext to scale up, which to centralize. Actually, the vanilla
    // `BatchEncoder` also provides `scale_up` and `centralize` methods, which converts a encoded `mod t` plaintext to `mod Q`.
    
}