#include "examples.h"

using namespace std;
using namespace troy;

static inline uint64_t multiply_mod(uint64_t a, uint64_t b, uint64_t t) {
    __uint128_t c = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    return static_cast<uint64_t>(c % static_cast<__uint128_t>(t));
}

static std::vector<uint64_t> random_vector(size_t size, uint64_t max_value = 10) {
    std::vector<uint64_t> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = rand() % max_value;
    }
    return result;
}

void example_batched_operation() {

    print_example_banner("Batched operation");
    
    // To further enhance performance by kernel fusing, we provide batched operations
    // for a subset of all available HE operations. If an operation supports batching,
    // we provide this functionality with a suffix "_batched" to the original method name.
    // In the following example, we demonstrate the usage of multiply_plain multiplication
    // in BFV.

    // The batch operation is only effective using GPU.
    // For CPU, the API is still available but it simply calls the original method
    // multiple times.
    if (utils::device_count() == 0) {
        return;
    }
    
    SchemeType scheme = SchemeType::BFV;
    constexpr uint64_t N = 8192;
    auto plain_modulus = PlainModulus::batching(N, 21);
    auto t = plain_modulus.value();
    auto coeff_modulus = CoeffModulus::create(N, { 60, 40, 40, 60 });

    EncryptionParameters parms(scheme);
    parms.set_coeff_modulus(coeff_modulus);
    parms.set_plain_modulus(plain_modulus);
    parms.set_poly_modulus_degree(N);
    HeContextPointer he = HeContext::create(parms, true, SecurityLevel::Classical128);

    // Create encoder and convey to GPU memory
    BatchEncoder encoder(he);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    }

    // Create other util classes
    KeyGenerator keygen(he);
    Encryptor encryptor(he); encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    constexpr size_t batch_size = 16;
    constexpr size_t repeat = 100;

    bench::TimerSingle timer_batched;
    bench::TimerSingle timer_single;

    vector<vector<uint64_t>> messages1; messages1.reserve(batch_size);
    vector<vector<uint64_t>> messages2; messages2.reserve(batch_size);
    vector<vector<uint64_t>> truths; truths.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        auto v1 = random_vector(N, t), v2 = random_vector(N, t);
        vector<uint64_t> truth(N); 
        for (size_t j = 0; j < N; j++) {
            truth[j] = multiply_mod(v1[j], v2[j], t);
        }
        messages1.push_back(std::move(v1));
        messages2.push_back(std::move(v2));
        truths.push_back(std::move(truth));
    }

    vector<Plaintext> encoded1; encoded1.reserve(batch_size);
    vector<Plaintext> encoded2; encoded2.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        encoded1.push_back(encoder.encode_new(messages1[i]));
        encoded2.push_back(encoder.encode_new(messages2[i]));
    }

    vector<Ciphertext> encrypted1; encrypted1.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        encrypted1.push_back(encryptor.encrypt_symmetric_new(encoded1[i], false));
    }

    for (size_t repeat_time = 0; repeat_time < repeat; repeat_time++) {

        // In all batched operations, we need to pass in a vector of pointers,
        // instead of references. Therefore, the input list of operand can have
        // repeated elements. This is useful if you want to carry out operation like 
        // `a * b[i]`, with a fixed `a` and a list of `b[i]`. You simple create a pointer list
        // where each element is the pointer to the same `a`.

        // In this example, we simply carry out `a[i] * b[i]`.
        // We provide some utilities within troy::batch_utils to create pointer lists.
        auto encrypted1_ptrs = batch_utils::collect_const_pointer(encrypted1);
        auto encoded2_ptrs = batch_utils::collect_const_pointer(encoded2);

        vector<Ciphertext> result(batch_size);
        auto result_ptrs = batch_utils::collect_pointer(result);

        // Here we go.
        timer_batched.tick();
        evaluator.multiply_plain_batched(encrypted1_ptrs, encoded2_ptrs, result_ptrs);
        timer_batched.tock();

        // Check correct
        for (size_t i = 0; i < batch_size; i++) {
            vector<uint64_t> decrypted = encoder.decode_new(decryptor.decrypt_new(result[i]));
            if (decrypted != truths[i]) {
                throw std::logic_error("Batched operation failed.");
            }
        }

        // For comparison we also carry out the operation one by one.
        timer_single.tick();
        for (size_t i = 0; i < batch_size; i++) {
            evaluator.multiply_plain(encrypted1[i], encoded2[i], result[i]);
        }
        timer_single.tock();

        // Check correct
        for (size_t i = 0; i < batch_size; i++) {
            vector<uint64_t> decrypted = encoder.decode_new(decryptor.decrypt_new(result[i]));
            if (decrypted != truths[i]) {
                throw std::logic_error("Single operation failed.");
            }
        }
    }

    // See if the batched operation is faster?
    timer_batched.print_divided("Batched", repeat);
    timer_single.print_divided("Single", repeat);

}
