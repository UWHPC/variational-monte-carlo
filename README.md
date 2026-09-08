# Variational-Monte-Carlo

## Prerequisites

- Linux or WSL2
- CMake 3.25+
- Ninja
- GCC 15
- OpenBLAS and LAPACKE development files
- Git
- CUDA 13.3 for CUDA builds

## Run

Build and run an optimized CPU executable:

`./scripts/run.sh`

Available options:

- `--cuda` enables CUDA.
- `--fp32` uses single precision.
- `--ffast-math` enables unsafe floating-point optimizations.

Options can be combined, for example:

`./scripts/run.sh --cuda --fp32 --ffast-math`

## Debug

Build and run with debug instrumentation:

`./scripts/debug.sh`

`debug.sh` accepts the same options as `run.sh`.

## Run Tests

Build and run the test suite:

`./scripts/test.sh` configures, compiles, and runs the CPU and CUDA test pipelines.
If CUDA can be compiled but no NVIDIA GPU is available, the CUDA tests are compiled
but not run. If no CUDA compiler is available, the default invocation skips CUDA.

Use `./scripts/test.sh --cpu` or `./scripts/test.sh --cuda` to select one backend.
The script also accepts `--fp32` and `--ffast-math`.
Behavior tests run on every push and pull request in all four CPU precision and
fast-math configurations. CUDA tests and validation targets are compile-checked;
CI does not run them on a GPU. The test script excludes scientific validation,
including when reusing a build directory that previously enabled it.

## Run Scientific Validation

Run analytical, independent numerical-reference, and literature validations:

`./scripts/validate.sh`

The validation script accepts `--cpu`, `--cuda`, `--fp32`, and `--ffast-math`.
It builds in Release. The separate **Scientific validation** workflow runs the
four CPU configurations nightly at 06:00 UTC and can also be started manually
with a branch, tag, or commit in its `ref` input. Run it against the proposed
commit before merging numerical or physics changes. This is a review requirement;
the workflow does not configure branch-protection rules.

It checks free-gas kinetic energy and zero variance, the same-spin cusp,
periodicity, Ewald energy, incremental Slater updates, and the fully polarized
electron-gas energy against the Perdew-Zunger parametrization of Ceperley-Alder
data ([PZ81](https://doi.org/10.1103/PhysRevB.23.5048),
[CA80](https://doi.org/10.1103/PhysRevLett.45.566)). The literature comparison
uses thermodynamic-limit reference values; this simulation uses a finite cubic
cell at the Gamma point, so its comparison tolerance also includes finite-size
and trial-wavefunction bias.

## LSP / clangd

By default, clangd uses the CPU compilation database so IntelliSense works with
or without CUDA. When CUDA 13.3 is installed, the setup script also generates a
separate CUDA compilation database in `build-cuda`.

When VS Code prompts for recommended extensions, install **clangd**. The
workspace disables the Microsoft C/C++ language server to prevent two
IntelliSense engines from competing.

Generate them with:

`./scripts/configure-clangd.sh`

The CPU database is generated even when CUDA is not installed.

After generating the database for the first time, run **clangd: Restart language
server** from the VS Code command palette (or reload the window).
