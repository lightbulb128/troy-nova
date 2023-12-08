#include <vector>
#include "test.cuh"
#include "../src/ckks_encoder.cuh"
#include "test_adv.cuh"

using namespace troy;
using troy::utils::Array;
using troy::utils::ConstSlice;
using troy::utils::Slice;
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


void test_multiply() {

    GeneralHeContext context     ( true, SchemeType::BFV, 8, 20, { 40, 40, 40 }, false, 0x123, 0);
    GeneralHeContext context_host(false, SchemeType::BFV, 8, 20, { 40, 40, 40 }, false, 0x123, 0);

    uint64_t t = context.t();
    double scale = context.scale();
    double tolerance = context.tolerance();

    GeneralVector message1 = context.random_simd_full();
    GeneralVector message2 = context.random_simd_full();
    Plaintext encoded1 = context.encoder().encode_simd(message1, std::nullopt, scale);
    Plaintext encoded2 = context.encoder().encode_simd(message2, std::nullopt, scale);
    Ciphertext encrypted1 = context.encryptor().encrypt_asymmetric_new(encoded1);
    Ciphertext encrypted2 = context.encryptor().encrypt_asymmetric_new(encoded2);

    Ciphertext multiplied = context.evaluator().multiply_new(encrypted1, encrypted2);
    
    Ciphertext multiplied_host = context_host.evaluator()
        .multiply_new(encrypted1.to_host(), encrypted2.to_host());
    Ciphertext multiplied_redevice = multiplied_host.to_device();

    std::cerr << "multipled = " << multiplied.data() << std::endl;
    std::cerr << "multipled_redevice = " << multiplied_redevice.data() << std::endl;

    Plaintext decrypted = context.decryptor().decrypt_new(multiplied_redevice);
    GeneralVector result = context.encoder().decode_simd(decrypted);
    GeneralVector truth = message1.mul(message2, t);
    ASSERT_TRUE(truth.near_equal(result, tolerance));
}
    

int main() {
    test_multiply();
    return 0;
}

// make custom && ./test/custom > a.log 2>&1