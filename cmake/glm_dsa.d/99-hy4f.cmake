# the Hy4 flash port (hy_v4_flash): the geometry reader and shape check join strata_engine; the probe is a host-only
# tool (no GPU, no weight read) built whenever strata_engine exists.
if(TARGET strata_engine)
  target_sources(strata_engine PRIVATE src/core/hy4f_geometry.cpp)
  add_executable(hy4f_geometry_probe tests/hy4f/hy4f_geometry_probe.cpp)
  target_link_libraries(hy4f_geometry_probe PRIVATE strata_engine)
  target_include_directories(hy4f_geometry_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  glm_dsa_hip_links(hy4f_geometry_probe)
endif()
