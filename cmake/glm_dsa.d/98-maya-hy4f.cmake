# maya-hy4f: Tencent Hy4 flash (hy_v4_flash GGUF) on NVIDIA GPUs - its own engine binary (src/hy4f), on Maya's
# kernels (native MMVQ, bf16 GEMV, the glm5-next k-pool indexer pieces, the sigmoid router).
if(STRATA_ENABLE_CUDA AND TARGET strata_kernels AND TARGET strata_engine)
  add_executable(maya-hy4f src/hy4f/hy4f_main.cpp src/hy4f/hy4f_serve.cpp src/hy4f/hy4f.cu)
  target_include_directories(maya-hy4f PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/hy4f)
  target_link_libraries(maya-hy4f PRIVATE strata_engine strata_kernels CUDA::cudart CUDA::cublas)
  if(TARGET hy4f_vision)
    target_link_libraries(maya-hy4f PRIVATE hy4f_vision)
    target_compile_definitions(maya-hy4f PRIVATE HY4F_VISION=1)
  endif()
  set_target_properties(maya-hy4f PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_compile_options(maya-hy4f PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
