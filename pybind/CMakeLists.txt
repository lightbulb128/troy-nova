pybind11_add_module(pytroy_raw 
    src/basics.cu
    src/modulus.cu
    src/encryption_parameters.cu
    src/he_context.cu
    src/plaintext.cu
    src/ciphertext.cu
    src/lwe_ciphertext.cu
    src/secret_key.cu
    src/public_key.cu
    src/kswitch_keys.cu
    src/key_generator.cu
    src/batch_encoder.cu
    src/ckks_encoder.cu
    src/polynomial_encoder_ring2k.cu
    src/random_generator.cu
    src/encryptor.cu
    src/decryptor.cu
    src/evaluator.cu
    src/matmul_helper.cu
    src/conv2d_helper.cu
    src/binder.cu
)

target_link_libraries(pytroy_raw PRIVATE troy)

# copy "tests" directory to build directory
file(COPY tests DESTINATION ${CMAKE_CURRENT_BINARY_DIR})