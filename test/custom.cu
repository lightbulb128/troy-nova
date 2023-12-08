#include <vector>
#include "test.cuh"
#include "../src/ckks_encoder.cuh"
#include "test_adv.cuh"

using namespace troy;
using troy::utils::Array;
using troy::utils::ConstSlice;
using troy::utils::Slice;
using troy::utils::DynamicArray;
using std::vector;
using std::complex;
using tool::GeneralEncoder;
using tool::GeneralVector;
using tool::GeneralHeContext;

void ASSERT_TRUE(bool condition) {
    if (!condition) {
        printf("ASSERTION FAILED\n");
    }
}


void test_keyswitching(const GeneralHeContext& context) {
    uint64_t t = context.t();
    double scale = context.scale();
    double tolerance = context.tolerance();

    // std::cerr << "t = " << t << "\n";
    // std::cerr << "qs = " << context.context()->key_context_data().value()->parms().coeff_modulus() << "\n";
    // std::cerr << "secret_key = " << context.key_generator().secret_key().data() << "\n";

    // create another keygenerator
    KeyGenerator keygen_other = KeyGenerator(context.context());
    SecretKey secret_key_other = keygen_other.secret_key();

    // std::cerr << "secret_key_other = " << secret_key_other.data() << "\n";

    Encryptor encryptor_other = Encryptor(context.context());
    encryptor_other.set_secret_key(secret_key_other);

    KSwitchKeys kswitch_key = context.key_generator().create_keyswitching_key(secret_key_other, false);
    size_t kn = kswitch_key.data()[0].size();
    for (size_t i = 0; i < kn; i++) {
        // std::cerr << "ks[" << i << "].data() = " << kswitch_key.data()[0][i].data() << "\n";
    }


    GeneralVector message = GeneralVector(vector<uint64_t>{ 1, 2, 3, 4 });
    Plaintext encoded = context.encoder().encode_polynomial(message, std::nullopt, scale);

    // std::cerr << "encoded.data() = " << encoded.data() << "\n";


    Ciphertext encrypted = encryptor_other.encrypt_symmetric_new(encoded, false);

    // std::cerr << "encrypted.data() = " << encrypted.data() << "\n";

    Ciphertext switched = context.evaluator().apply_keyswitching_new(encrypted, kswitch_key);
    Plaintext decrypted = context.decryptor().decrypt_new(switched);
    GeneralVector result = context.encoder().decode_polynomial(decrypted);
    GeneralVector truth = message;
    // std::cout << "message: " << message << "\n";
    // std::cout << "result:  " << result << "\n"; 
    ASSERT_TRUE(truth.near_equal(result, tolerance));
}
    

int main() {
    GeneralHeContext ghe(true, SchemeType::BFV, 8, 20, { 60, 60, 60 }, true, 0x123, 0);
    test_keyswitching(ghe);
    return 0;
}

// make custom && ./test/custom > a.log 2>&1