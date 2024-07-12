file(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES *.cpp *.cu *.c)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

add_library(troy SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(troy_static STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
set_target_properties(troy PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

target_compile_definitions(troy        PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
target_compile_definitions(troy_static PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)

target_link_libraries(troy PUBLIC Threads::Threads)
target_link_libraries(troy_static PUBLIC Threads::Threads)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>")
set(nvcc_cxx "$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>")

target_compile_options(troy INTERFACE
  "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-fPIC>>"
  "$<${nvcc_cxx}:$<BUILD_INTERFACE:--compiler-options=-fPIC>>"
)

# if TROY_MEMORY_POOL is on, add this as a macro
if(TROY_MEMORY_POOL)
  target_compile_definitions(troy        PUBLIC TROY_MEMORY_POOL)
  target_compile_definitions(troy_static PUBLIC TROY_MEMORY_POOL)
endif()

if(TROY_MEMORY_POOL_UNSAFE)
  target_compile_definitions(troy        PUBLIC TROY_MEMORY_POOL_UNSAFE)
  target_compile_definitions(troy_static PUBLIC TROY_MEMORY_POOL_UNSAFE)
endif()

if(TROY_ZSTD)
  target_compile_definitions(troy        PUBLIC TROY_ZSTD)
  target_compile_definitions(troy_static PUBLIC TROY_ZSTD)
  target_link_libraries(troy        PRIVATE ${TROY_ZSTD_STATIC_LINK_NAME})
  target_link_libraries(troy_static PRIVATE ${TROY_ZSTD_STATIC_LINK_NAME})
endif()

install(TARGETS troy troy_static
  EXPORT troyTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

file(GLOB MAIN_DIR_HEADERS *.h *.hpp *.cuh)
file(GLOB UTILS_HEADERS utils/*.h *.hpp *.cuh)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION include/troy
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")