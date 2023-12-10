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
    

int main() {
    GeneralHeContext gheh(false, SchemeType::BFV, 8, 20, { 60, 60, 60 }, true, 0x123, 0);
    GeneralHeContext ghed( true, SchemeType::BFV, 8, 20, { 60, 60, 60 }, true, 0x123, 0);
    
    uint64_t t = gheh.t();
    double scale = gheh.scale();
    double tolerance = gheh.tolerance();

    GeneralVector message = gheh.random_polynomial_full();
    Plaintext encoded = gheh.encoder().encode_polynomial(message, std::nullopt, scale);
    Ciphertext encrypted = gheh.encryptor().encrypt_asymmetric_new(encoded);
    Ciphertext encrypted_device = encrypted.to_device();

    vector<size_t> terms = {1};
    for (size_t term : terms) {
        auto extracted_device = ghed.evaluator().extract_lwe_new(encrypted_device, term);
        auto extracted = extracted_device.to_host();
        // auto extracted = extracted_device.to_host();
        // auto assembled_device = ghed.evaluator().assemble_lwe_new(extracted_device);
        auto assembled = gheh.evaluator().assemble_lwe_new(extracted);
        if (gheh.params_host().scheme() == SchemeType::CKKS) {
            gheh.evaluator().transform_to_ntt_inplace(assembled);
        }
        auto decrypted = gheh.decryptor().decrypt_new(assembled);
        auto decoded = gheh.encoder().decode_polynomial(decrypted);
        ASSERT_TRUE(message.element(term).near_equal(decoded.element(0), tolerance));
    }

    return 0;
}

// make custom && ./test/custom > a.log 2>&1