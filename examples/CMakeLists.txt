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
    10_bfv_matmul.cu
    11_ckks_matmul.cu
    12_lwes.cu
    13_ring2k.cu
    14_bfv_conv2d.cu
    15_batched_operation.cu
    20_memory_pools.cu
    30_issue_multithread.cu
    99_quickstart.cu
)

target_link_libraries(troyexamples PRIVATE troy)
