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

## Visualize

`render.py` reads a recorded trajectory and plays it back in
[Rerun](https://rerun.io) — particle positions in the periodic box, plus
local energy, running mean, standard error, and acceptance rate over time.

By default `./scripts/run.sh` does not record: `Simulation::run()` takes a
much faster resident-kernel path when nothing needs a per-proposal position
snapshot. Recording is opt-in via the `VMC_OUTPUT` environment variable,
which trades that speed for a trajectory:

`VMC_OUTPUT=output/vmc.bin ./scripts/run.sh`

(`main.cu` creates `output/` itself if it doesn't exist.)

One-time setup:

`python3 -m venv .venv` \
`.venv/bin/pip install -r requirements.txt`

Then visualize the result:

`.venv/bin/python render.py --input output/vmc.bin`

Script parameters:

- `--input <PATH>` (default: `output/vmc.bin`)
- `--stride <N>` render every Nth frame (default: `1`)

### Troubleshooting: blank viewer window on WSL2

WSLg can fall back to software rendering for the native Rerun viewer
window, which comes up but never actually paints anything. If that
happens, serve the recording over Rerun's web viewer instead — it renders
in an ordinary browser tab and sidesteps WSLg entirely. In one terminal:

`rerun --serve-web --bind 127.0.0.1`

Leave that running, then in another terminal run `render.py` as above; it
detects the running server and streams to it instead of spawning a
(broken) native window. Open the URL the first command printed
(`http://127.0.0.1:9090?url=...`) in any browser.

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

## Run Scientific Validation

Run analytical, independent numerical-reference, and literature validations:

`./scripts/validate.sh`

The validation script accepts `--cpu`, `--cuda`, `--fp32`, and `--ffast-math`.
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
