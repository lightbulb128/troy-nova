#include "examples.h"

using namespace std;
using namespace troy;
using namespace troy::linear;

namespace ns_bfv_conv2d {

    inline uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t t) {
        __uint128_t c = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
        return static_cast<uint64_t>(c % static_cast<__uint128_t>(t));
    }

    inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t t) {
        if (a + b >= t) {
            return a + b - t;
        } else {
            return a + b;
        }
    }

    inline void add_mod_inplace(uint64_t& a, uint64_t b, uint64_t t) {
        a = add_mod(a, b, t);
    }

    std::vector<uint64_t> random_polynomial(size_t size, uint64_t max_value = 10) {
        std::vector<uint64_t> result(size);
        for (size_t i = 0; i < size; i++) {
            result[i] = rand() % max_value;
        }
        return result;
    }

    bool vector_equal(const vector<uint64_t>& a, const vector<uint64_t>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    void test_conv2d(
        bool is_bgv,
        size_t poly_modulus_degree,
        const Modulus& plain_modulus,
        const vector<Modulus>& coeff_modulus,
        size_t bs, size_t ic, size_t oc, size_t ih, size_t iw, size_t kh, size_t kw,
        bool mod_switch_to_next
    ) {

        // bs: batch size
        // ic: input channels
        // oc: output channels
        // ih: input height
        // iw: input width
        // kh: kernel height
        // kw: kernel width
        
        // This is very similar to the matmul example, so we don't explain much here.
        // The only difference is that we don't use LWE packing here.

        SchemeType scheme = is_bgv ? SchemeType::BGV : SchemeType::BFV;
        
        EncryptionParameters parms(scheme);
        parms.set_coeff_modulus(coeff_modulus);
        parms.set_plain_modulus(plain_modulus);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);

        BatchEncoder encoder(he);
        if (utils::device_count() > 0) {
            he->to_device_inplace();
            encoder.to_device_inplace();
        }

        uint64_t t = parms.plain_modulus_host().value();
        size_t oh = ih - kh + 1;
        size_t ow = iw - kw + 1;
        
        vector<uint64_t> x = random_polynomial(bs * ic * ih * iw);
        vector<uint64_t> w = random_polynomial(oc * ic * kh * kw);
        vector<uint64_t> s = random_polynomial(bs * oc * oh * ow);
        Conv2dHelper helper(bs, ic, oc, ih, iw, kh, kw,
            parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);

        KeyGenerator keygen(he);
        Encryptor encryptor(he); encryptor.set_secret_key(keygen.secret_key());
        Evaluator evaluator(he);
        Decryptor decryptor(he, keygen.secret_key());
        
        Plain2d x_encoded = helper.encode_inputs_uint64s(encoder, x.data());
        Plain2d w_encoded = helper.encode_weights_uint64s(encoder, w.data());
        Plain2d s_encoded = helper.encode_outputs_uint64s(encoder, s.data());

        Cipher2d x_encrypted = x_encoded.encrypt_symmetric(encryptor);

        stringstream x_serialized;
        x_encrypted.save(x_serialized, he);
        x_encrypted = Cipher2d::load_new(x_serialized, he);

        Cipher2d y_encrypted = helper.conv2d(evaluator, x_encrypted, w_encoded);
        if (mod_switch_to_next) {
            y_encrypted.mod_switch_to_next_inplace(evaluator);
        }

        y_encrypted.add_plain_inplace(evaluator, s_encoded);

        stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        y_encrypted = helper.deserialize_outputs(evaluator, y_serialized);

        vector<uint64_t> y_decrypted = helper.decrypt_outputs_uint64s(encoder, decryptor, y_encrypted);   

        vector<uint64_t> y_truth(bs * oc * oh * ow, 0);
        for (size_t b = 0; b < bs; b++) {
            for (size_t o = 0; o < oc; o++) {
                for (size_t i = 0; i < oh; i++) {
                    for (size_t j = 0; j < ow; j++) {
                        for (size_t c = 0; c < ic; c++) {
                            for (size_t p = 0; p < kh; p++) {
                                for (size_t q = 0; q < kw; q++) {
                                    add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                                        multiply_mod(
                                            x[b * ic * ih * iw + c * ih * iw + (i + p) * iw + (j + q)],
                                            w[o * ic * kh * kw + c * kh * kw + p * kw + q],
                                            t
                                        ), t);
                                }
                            }
                        }
                        add_mod_inplace(y_truth[b * oc * oh * ow + o * oh * ow + i * ow + j],
                            s[b * oc * oh * ow + o * oh * ow + i * ow + j], t);
                    }
                }
            }
        }
        
        bool success = vector_equal(y_decrypted, y_truth);
        if (success) {
            std::cout << "Conv2d test passed!" << std::endl;
        } else {
            std::cout << "Conv2d test failed!" << std::endl;
        }
        std::cout << std::endl;

    }

}


void example_bfv_conv2d()
{

    print_example_banner("BFV/BGV 2d convolution");

    // Test the conv2d function with BFV scheme.
    // the plaintext modulus can be a non-prime here.

    size_t poly_modulus_degree = 8192;
    ns_bfv_conv2d::test_conv2d(
        false, poly_modulus_degree, 
        Modulus(1<<21), CoeffModulus::create(poly_modulus_degree, { 60, 40, 40, 60 }).to_vector(),
        2, 3, 5, 15, 15, 3, 3, false
    );


}