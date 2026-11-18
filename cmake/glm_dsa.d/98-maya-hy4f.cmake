# maya-hy4f: Tencent Hy4 flash (hy_v4_flash GGUF) on NVIDIA GPUs - its own engine binary (src/hy4f), on Maya's
# kernels (native MMVQ, bf16 GEMV, the glm5-next k-pool indexer pieces, the sigmoid router).
# maya-hy4f is the production build (verify steps of up to 4 tokens); maya-hy4f-dfly adds -DHY4F_DT6 (up to 6: the
# DFly2 drafter's block of bonus + 5 drafts; ~40 MB more per GPU and a few % slower prefill / short verify steps).
if(STRATA_ENABLE_CUDA AND TARGET strata_kernels AND TARGET strata_engine)
  foreach(hy4f_t maya-hy4f maya-hy4f-dfly)
    add_executable(${hy4f_t} src/hy4f/hy4f_main.cpp src/hy4f/hy4f_serve.cpp src/hy4f/hy4f.cu)
    target_include_directories(${hy4f_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/hy4f)
    target_link_libraries(${hy4f_t} PRIVATE strata_engine strata_kernels CUDA::cudart CUDA::cublas)
    if(TARGET hy4f_vision)
      target_link_libraries(${hy4f_t} PRIVATE hy4f_vision)
      target_compile_definitions(${hy4f_t} PRIVATE HY4F_VISION=1)
    endif()
    set_target_properties(${hy4f_t} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_options(${hy4f_t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endforeach()
  target_compile_definitions(maya-hy4f-dfly PRIVATE HY4F_DT6=1)
endif()
