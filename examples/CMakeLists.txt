add_executable(troyexamples)

target_sources(troyexamples PRIVATE
    examples.cu
    1_bfv_basics.cu
    2_encoders.cu
    3_levels.cu
    4_bgv_basics.cu
    5_ckks_basics.cu
    6_rotation.cu
    7_serialization.cu
)

target_link_libraries(troyexamples PRIVATE troy)
