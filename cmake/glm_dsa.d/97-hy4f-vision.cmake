# maya-hy4f vision: the Hy4 flash image encoder (mmproj "hy_v4_flash": 27-layer ViT + rmsnorm/conv projector) on one GPU.
# hy4f_vision (static library, API in src/hy4f/hy4f_vision.hpp) + hy4f_vision_test (encode, dump, compare with the
# llama.cpp reference from tools/hy4f_vision_ref.cpp).  No --use_fast_math: the position resample and erf/tanh gelu
# must match llama.cpp's float math.
if(STRATA_ENABLE_CUDA AND TARGET strata_kernels AND TARGET strata_engine)
  add_library(hy4f_vision STATIC src/hy4f/hy4f_vision.cu src/hy4f/hy4f_vision_pre.cpp)
  target_include_directories(hy4f_vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/hy4f
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  target_link_libraries(hy4f_vision PUBLIC strata_engine strata_kernels CUDA::cudart CUDA::cublas)
  set_target_properties(hy4f_vision PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
  set_source_files_properties(src/hy4f/hy4f_vision_pre.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unused-but-set-variable")
  find_package(Threads REQUIRED)
  target_link_libraries(hy4f_vision PUBLIC Threads::Threads)

  add_executable(hy4f_vision_test src/hy4f/hy4f_vision_test.cpp)
  target_link_libraries(hy4f_vision_test PRIVATE hy4f_vision)
endif()
