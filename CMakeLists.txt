cmake_minimum_required(VERSION 3.25)

if(
  NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS
  OR CMAKE_EXPORT_COMPILE_COMMANDS STREQUAL ""
)
  set(
    CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Generate compile_commands.json for tooling" FORCE
  )
endif()

project(Variational-Monte-Carlo LANGUAGES CXX)

option(VMC_ENABLE_CUDA "Enable CUDA acceleration if a CUDA compiler is available" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CUDA detection ---
set(VMC_CUDA OFF)
if(VMC_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 23)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(VMC_CUDA ON)
  endif()
endif()

# --- Floating point precision ---
option(FP_64 "Use double precision (real_t = double) instead of single precision (real_t = float)" ON)
option(VMC_FAST_MATH "Use fast approximate math intrinsics (__sincosf, __expf, ...) in CUDA device code; forces single precision (FP_64=OFF)" OFF)

if(VMC_FAST_MATH AND FP_64)
  message(STATUS "VMC_FAST_MATH is ON: forcing single precision (FP_64=OFF).")
  set(FP_64 OFF)
endif()

# --- Dependencies ---
include(FetchContent)

# xpu must use the same backend as VMC: mixing CPU and CUDA allocation in one
# linked program is undefined. Let VMC own its own -march flags.
set(XPU_ENABLE_CUDA ${VMC_ENABLE_CUDA} CACHE BOOL "" FORCE)
set(XPU_ARCH_NATIVE OFF CACHE BOOL "" FORCE)

fetchcontent_declare(
  xpu
  GIT_REPOSITORY https://github.com/UWHPC/xpu.git
  GIT_TAG 22fdac2
)
fetchcontent_makeavailable(xpu)

find_package(OpenMP QUIET COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found (CXX ${OpenMP_CXX_VERSION}); using it for '#pragma omp simd'.")
else()
  message(STATUS "OpenMP not found; building without it (we only use '#pragma omp simd').")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fopenmp-simd" VMC_HAS_FOPENMP_SIMD)
endif()

# --- Build type default (Release unless overridden) ---
if(NOT CMAKE_CONFIGURATION_TYPES)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# --- Compiler warnings and optimizations ---
if(MSVC)
  set(VMC_WARNING_OPTIONS /W4 /permissive- /arch:AVX2)
  set(VMC_RELEASE_OPTIONS /O2)
  set(VMC_DEBUG_OPTIONS   /Od /Zi)
  set(VMC_RELWITHDEBINFO_OPT_1 /O2)
  set(VMC_RELWITHDEBINFO_OPT_2 /Oy-)
  set(VMC_RELWITHDEBINFO_OPT_3 "")
else()
  set(VMC_WARNING_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
  )
  set(VMC_RELEASE_OPTIONS -O3 -march=native)
  set(VMC_DEBUG_OPTIONS   -g -O0)
  set(VMC_RELWITHDEBINFO_OPT_1 -O3)
  set(VMC_RELWITHDEBINFO_OPT_2 -march=native)
  set(VMC_RELWITHDEBINFO_OPT_3 -fno-omit-frame-pointer)
endif()

if(VMC_CUDA)
  if(MSVC)
    set(VMC_CUDA_XCOMPILE "-Xcompiler=/W4,/permissive-")
    set(VMC_CUDA_RELEASE_OPTIONS -Xcompiler=/O2)
    set(VMC_CUDA_DEBUG_OPTIONS   -Xcompiler=/Od,/Zi -G)
  else()
    set(VMC_CUDA_XCOMPILE "-Xcompiler=-Wall,-Wextra,-march=native")
    set(VMC_CUDA_RELEASE_OPTIONS -O3 -Xcompiler=-O3)
    set(VMC_CUDA_DEBUG_OPTIONS   -g -G -O0 -Xcompiler=-O0)
  endif()
endif()

# --- Source lists ---
set(VMC_CUDA_SOURCES
  src/jastrow_pade/jastrow_pade.cu
  src/jastrow_pade/value_delta_value.cu
  src/jastrow_pade/add_derivatives.cu
  src/wavefunction/wavefunction.cu
  src/slater_plane_wave/build_row_determinant_ratio.cu
  src/slater_plane_wave/initialize.cu
  src/slater_plane_wave/log_abs_det.cu
  src/slater_plane_wave/slater_plane_wave.cu
  src/slater_plane_wave/update_restore_trig.cu
  src/slater_plane_wave/add_derivatives.cu
  src/simulation/simulation.cu
  src/energy_tracking/energy_tracking.cu
  src/optimizer/jastrow_optimizer.cu
  src/optimizer/bayes_optimizer.cu
)

set(VMC_CXX_SOURCES
  src/blocking_analysis/blocking_analysis.cpp
  src/output_writer/output_writer.cpp
)

set(VMC_CUDA_EXTENSION_SOURCES
  ${VMC_CUDA_SOURCES}
)

# --- Library target for all source files ---
add_library(vmc_lib STATIC
  ${VMC_CXX_SOURCES}
  ${VMC_CUDA_SOURCES}
)

target_include_directories(vmc_lib PUBLIC src)

target_link_libraries(vmc_lib PUBLIC xpu::xpu)
find_package(Threads REQUIRED)
target_link_libraries(vmc_lib PRIVATE Threads::Threads)

# GaussianProcess uses the host LAPACK Cholesky routine with LP64 integers.
set(BLA_SIZEOF_INTEGER 4)
find_package(LAPACK REQUIRED)
target_link_libraries(vmc_lib PRIVATE LAPACK::LAPACK)

if(NOT VMC_CUDA AND VMC_CUDA_EXTENSION_SOURCES)
  set_source_files_properties(
    ${VMC_CUDA_EXTENSION_SOURCES}
    PROPERTIES LANGUAGE CXX
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
      ${VMC_CUDA_EXTENSION_SOURCES}
      PROPERTIES COMPILE_OPTIONS "-x;c++"
    )
  endif()
endif()

if(FP_64)
  target_compile_definitions(vmc_lib PUBLIC FP_64)
endif()

if(VMC_FAST_MATH)
  target_compile_definitions(vmc_lib PUBLIC VMC_FAST_MATH)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(vmc_lib PUBLIC OpenMP::OpenMP_CXX)
  if(MSVC)
    target_compile_options(vmc_lib PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
  endif()
elseif(VMC_HAS_FOPENMP_SIMD)
  target_compile_options(vmc_lib PUBLIC -fopenmp-simd)
endif()
target_compile_options(vmc_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${VMC_WARNING_OPTIONS}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${VMC_RELEASE_OPTIONS}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:${VMC_DEBUG_OPTIONS}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:${VMC_RELWITHDEBINFO_OPT_1}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:${VMC_RELWITHDEBINFO_OPT_2}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:${VMC_RELWITHDEBINFO_OPT_3}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${VMC_CUDA}>>:${VMC_CUDA_XCOMPILE}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${VMC_CUDA}>,$<CONFIG:Release>>:${VMC_CUDA_RELEASE_OPTIONS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${VMC_CUDA}>,$<CONFIG:Debug>>:${VMC_CUDA_DEBUG_OPTIONS}>
)

if(VMC_CUDA)
  find_package(CUDAToolkit REQUIRED)

  set_target_properties(vmc_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  target_compile_definitions(vmc_lib PUBLIC VMC_CUDA_BACKEND)

  target_link_libraries(vmc_lib PUBLIC
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
  )
endif()

# --- Main executable ---
add_executable(vmc
  src/main.cpp
)

target_link_libraries(vmc PRIVATE vmc_lib)
target_compile_options(vmc PRIVATE
  ${VMC_WARNING_OPTIONS}
  $<$<CONFIG:Release>:${VMC_RELEASE_OPTIONS}>
  $<$<CONFIG:Debug>:${VMC_DEBUG_OPTIONS}>
  $<$<CONFIG:RelWithDebInfo>:${VMC_RELWITHDEBINFO_OPT_1}>
  $<$<CONFIG:RelWithDebInfo>:${VMC_RELWITHDEBINFO_OPT_2}>
  $<$<CONFIG:RelWithDebInfo>:${VMC_RELWITHDEBINFO_OPT_3}>
)

# --- Testing (opt-in via BUILD_TESTING argument) ---
option(BUILD_BENCHMARKS "Build the optimizer comparison executable" OFF)
if(BUILD_BENCHMARKS)
  add_executable(optimizer_benchmark benchmarks/optimizer_benchmark.cpp)
  target_link_libraries(optimizer_benchmark PRIVATE vmc_lib)
  target_compile_options(optimizer_benchmark PRIVATE ${VMC_WARNING_OPTIONS})

  add_executable(fixed_b_scan benchmarks/fixed_b_scan.cpp)
  target_link_libraries(fixed_b_scan PRIVATE vmc_lib)
  target_compile_options(fixed_b_scan PRIVATE ${VMC_WARNING_OPTIONS})
endif()

include(CTest)

if(BUILD_TESTING)

  add_executable(vmc_bayes_tests tests/test_bayes_optimizer.cpp)
  target_link_libraries(vmc_bayes_tests PRIVATE vmc_lib)
  target_compile_options(vmc_bayes_tests PRIVATE ${VMC_WARNING_OPTIONS})
  add_test(NAME bayes_optimizer COMMAND vmc_bayes_tests)

  add_executable(vmc_jastrow_bayes_tests tests/test_jastrow_bayes.cpp)
  target_link_libraries(vmc_jastrow_bayes_tests PRIVATE vmc_lib)
  target_compile_options(vmc_jastrow_bayes_tests PRIVATE ${VMC_WARNING_OPTIONS})
  add_test(NAME jastrow_bayes COMMAND vmc_jastrow_bayes_tests)

  include(FetchContent)

  fetchcontent_declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
  )
  fetchcontent_makeavailable(Catch2)

  add_executable(vmc_tests
    tests/test_tests.cpp
  )

  target_link_libraries(vmc_tests PRIVATE
    Catch2::Catch2WithMain
    vmc_lib
  )
  target_compile_options(vmc_tests PRIVATE
    ${VMC_WARNING_OPTIONS}
    $<$<CONFIG:Release>:${VMC_RELEASE_OPTIONS}>
    $<$<CONFIG:Debug>:${VMC_DEBUG_OPTIONS}>
    $<$<CONFIG:RelWithDebInfo>:${VMC_RELWITHDEBINFO_OPT_1}>
    $<$<CONFIG:RelWithDebInfo>:${VMC_RELWITHDEBINFO_OPT_2}>
    $<$<CONFIG:RelWithDebInfo>:${VMC_RELWITHDEBINFO_OPT_3}>
  )

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(vmc_tests)

endif()
