#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX_COMPILER="/usr/bin/g++-15"
CUDA_COMPILER="/usr/local/cuda-13.3/bin/nvcc"

RUN_CPU=0
RUN_CUDA=0
BACKEND_SELECTED=0
FP64_FLAG="ON"
FAST_MATH_FLAG="OFF"

for arg in "$@"; do
  case "$arg" in
    --cpu)
      RUN_CPU=1
      BACKEND_SELECTED=1
      ;;
    --cuda)
      RUN_CUDA=1
      BACKEND_SELECTED=1
      ;;
    --fp32)
      FP64_FLAG="OFF"
      ;;
    --ffast-math)
      FAST_MATH_FLAG="ON"
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [ "$BACKEND_SELECTED" -eq 0 ]; then
  RUN_CPU=1
  RUN_CUDA=1
fi

if [ ! -x "$CXX_COMPILER" ]; then
  echo "missing C++ compiler: $CXX_COMPILER" >&2
  exit 1
fi

cuda_toolchain_available() {
  if [ ! -x "$CUDA_COMPILER" ]; then
    CUDA_UNAVAILABLE_REASON="missing CUDA compiler: $CUDA_COMPILER"
    return 1
  fi

  local cuda_root
  cuda_root="$(dirname "$(dirname "$CUDA_COMPILER")")"
  if [ ! -e "$cuda_root/targets/x86_64-linux/lib/libcusolver.so" ] \
     && [ ! -e "$cuda_root/lib64/libcusolver.so" ]; then
    CUDA_UNAVAILABLE_REASON="CUDA toolkit is missing cuSOLVER libraries under $cuda_root"
    return 1
  fi

  return 0
}

run_pipeline() {
  local backend="$1"
  local build_dir="$REPO_ROOT/build-test-$backend"
  local cuda_flag="OFF"
  local cmake_args=(
    -S "$REPO_ROOT"
    -B "$build_dir"
    -G Ninja
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER"
    -DFP_64="$FP64_FLAG"
    -DVMC_FAST_MATH="$FAST_MATH_FLAG"
    -DBUILD_TESTING=ON
    -DVMC_BUILD_VALIDATION=OFF
    -DCMAKE_BUILD_TYPE=Debug
  )

  if [ -d "$build_dir/_deps/xpu-src" ] && [ -d "$build_dir/_deps/catch2-src" ]; then
    cmake_args+=(-DFETCHCONTENT_UPDATES_DISCONNECTED=ON)
  fi

  if [ "$backend" = "cuda" ]; then
    cuda_flag="ON"
    cmake_args+=(
      -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER"
      -DCMAKE_CUDA_HOST_COMPILER="$CXX_COMPILER"
    )
  fi
  cmake_args+=("-DVMC_ENABLE_CUDA=$cuda_flag")

  echo "==> configuring $backend tests"
  cmake "${cmake_args[@]}"

  echo "==> compiling $backend tests"
  cmake --build "$build_dir" --target vmc_tests

  if [ "$backend" = "cuda" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
      echo "==> CUDA tests compiled; no NVIDIA GPU detected, skipping execution"
      return
    fi
  fi

  echo "==> running $backend tests"
  ctest --test-dir "$build_dir" --output-on-failure
}

if [ "$RUN_CPU" -eq 1 ]; then
  run_pipeline cpu
fi

if [ "$RUN_CUDA" -eq 1 ]; then
  if ! cuda_toolchain_available; then
    if [ "$BACKEND_SELECTED" -eq 1 ]; then
      echo "$CUDA_UNAVAILABLE_REASON" >&2
      exit 1
    fi
    echo "==> $CUDA_UNAVAILABLE_REASON; skipping CUDA pipeline"
  else
    run_pipeline cuda
  fi
fi
