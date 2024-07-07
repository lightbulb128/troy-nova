#include "header.h"

PYBIND11_MODULE(pytroy_raw, m) {

    m
        .def("it_works", []() {
            return 42;
        })
        .def("initialize_kernel", [](int device){
            troy::kernel_provider::initialize(device);
        })
        .def("destroy_memory_pool", []{
            troy::utils::MemoryPool::Destroy();
        })
        .def("device_count", [](){
            return troy::utils::device_count();
        })
    ;

    register_basics(m);
    register_modulus(m);
    register_encryption_parameters(m);
    register_he_context(m);
    register_plaintext(m);
    register_ciphertext(m);
    register_lwe_ciphertext(m);
    register_secret_key(m);
    register_public_key(m);
    register_kswitch_keys(m);
    register_key_generator(m);
    register_batch_encoder(m);
    register_ckks_encoder(m);
    register_polynomial_encoder_ring2k(m);
    register_random_generator(m);
    register_encryptor(m);
    register_decryptor(m);
    register_evaluator(m);
    register_matmul_helper(m);
    register_conv2d_helper(m);

}