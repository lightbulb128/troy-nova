enable_testing()

add_executable(troytest)
target_sources(troytest PRIVATE 

    utils/box.cu
    utils/basics.cu
    utils/uint_small_mod.cu
    utils/ntt.cu
    utils/rns_base.cu
    utils/rns_tool.cu
    utils/galois.cu
    utils/random_generator.cu

    modulus.cu
    he_context.cu
    batch_encoder.cu
    ckks_encoder.cu
    encryptor.cu

    test_adv.cu
    evaluator.cu
    
)

target_link_libraries(troytest gtest gtest_main troy)

include(GoogleTest)
gtest_discover_tests(troytest)

add_executable(custom)
target_sources(custom PRIVATE custom.cu test_adv.cu)
target_link_libraries(custom troy)