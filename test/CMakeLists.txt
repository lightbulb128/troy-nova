
if(TROY_TEST)

    include(CTest)

    message(STATUS "Tests enabled")

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
        test_multithread.cu
        evaluator.cu
        serialize.cu

        lwe.cu
        
        app/bfv_ring2k.cu
        app/matmul.cu
        app/matmul_ckks.cu
        app/matmul_ring2k.cu
        app/conv2d.cu
        app/conv2d_ckks.cu
        app/conv2d_ring2k.cu

        special_prime_for_encryption.cu

        multithread.cu
        # temp.cu

        batch_encoder_batched.cu
        encryptor_batched.cu
        evaluator_batched.cu
    )

    if(TROY_ZSTD)
        target_sources(troytest PRIVATE serialize_zstd.cu)
    endif()

    target_link_libraries(troytest gtest gtest_main troy)

    include(GoogleTest)
    gtest_discover_tests(troytest)
    
endif()

if(TROY_BENCH)

    message(STATUS "Benchmarks enabled")

    add_executable(troybench)
    target_sources(troybench PRIVATE 
        argparse.cpp
        bench/he_operations.cu
        test_adv.cu
        test_multithread.cu
    )
    target_link_libraries(troybench troy)

    add_executable(bench_ntt)
    target_sources(bench_ntt PRIVATE 
        bench/ntt.cu
    )
    target_link_libraries(bench_ntt troy)

    add_executable(bench_matmul)
    target_sources(bench_matmul PRIVATE 
        argparse.cpp
        bench/matmul.cu
        test_adv.cu
        test_multithread.cu
    )
    target_link_libraries(bench_matmul troy)

    add_executable(bench_conv2d)
    target_sources(bench_conv2d PRIVATE 
        argparse.cpp
        bench/conv2d.cu
        test_adv.cu
    )
    target_link_libraries(bench_conv2d troy)

endif()

# add_executable(custom3)
# target_sources(custom3 PRIVATE 
# custom3.cu
# )
# target_link_libraries(custom3 troy)
