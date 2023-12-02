#include <vector>
#include "test.cuh"
#include "../src/ckks_encoder.cuh"

using namespace troy;
using troy::utils::Array;
using troy::utils::ConstSlice;
using troy::utils::Slice;
using std::vector;
using std::complex;

void ASSERT_TRUE(bool condition) {
    if (!condition) {
        printf("ASSERTION FAILED\n");
    }
}

bool near_vector(vector<complex<double>> &a, vector<complex<double>> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i].real() - b[i].real()) > 0.5) {
            return false;
        }
        if (std::abs(a[i].imag() - b[i].imag()) > 0.5) {
            return false;
        }
    }
    return true;
}

void test_vector(bool device) {

    size_t slots;
    EncryptionParameters parms(SchemeType::CKKS);
    HeContextPointer context;
    vector<complex<double>> values;
    
    slots = 32;
    parms.set_poly_modulus_degree(slots << 1);
    parms.set_coeff_modulus(CoeffModulus::create(slots << 1, {40, 40, 40, 40}).const_reference());
    context = HeContext::create(parms, false, SecurityLevel::None);
    CKKSEncoder encoder = CKKSEncoder(context);
    if (device) {
        context->to_device_inplace();
        encoder.to_device_inplace();
    }
    double delta = std::pow(2.0, 16.0);

    values.resize(slots);
    for (size_t i = 0; i < slots; i++) {
        values[i] = complex<double>(0, 0);
    }
    // Plaintext plain; // = encoder.encode_complex64_simd_new(values, std::nullopt, delta);
    // vector<complex<double>> result; // = encoder.decode_complex64_simd_new(plain);
    // ASSERT_TRUE(near_vector(values, result));

    int bound = 16;
    for (size_t i = 0; i < slots; i++) {
        values[i] = complex<double>(rand() % bound, rand() % bound);
    }
    Plaintext plain = encoder.encode_complex64_simd_new(values, std::nullopt, delta);
    
    Array<complex<double>> empty(32, true);
    empty.to_host_inplace();

    auto result = encoder.decode_complex64_simd_new(plain);
    ASSERT_TRUE(near_vector(values, result));

}

int main() {
    test_vector(true);
    return 0;
}