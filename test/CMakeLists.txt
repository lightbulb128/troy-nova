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

    modulus.cu
)

target_link_libraries(troytest gtest gtest_main troy)

include(GoogleTest)
gtest_discover_tests(troytest)