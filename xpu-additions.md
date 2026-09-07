# xpu additions and migrations

Track reusable backend functionality for implementation in xpu after the VMC work. Refactor VMC to consume those APIs afterward. Names below are proposals, not existing xpu APIs unless explicitly noted.

## Already implemented locally

| Candidate | Current location | Work for xpu |
| --- | --- | --- |
| Portable execution helpers | [execution.hpp](src/utilities/execution.hpp): `execution::thread`, `stride`, `sync` | Provide block-local thread index, execution width, and synchronization. CPU execution uses index zero, width one, and no-op synchronization. Keep CUDA intrinsics behind this interface. |
| Cooperative sum reduction | [energy_tracking_kernels.hpp](src/energy_tracking/energy_tracking_kernels.hpp): `store_partial_sum` | Provide a callable block reduction, such as `xpu::block_reduce_sum`. Accept thread-local contributions and scratch storage, and write the result to backend memory. Define scratch requirements and supported execution widths; the current implementation assumes a power-of-two GPU block width. |
| Batched RNG seeding | [simulation_kernels.hpp](src/simulation/simulation_kernels.hpp): `seed_generator`, `seed_generators` | Extend the existing generator API with batched seeding from a master seed and stream IDs. Preserve each generator's stream independently of launch geometry. VMC retains the mapping from walker identity to stream ID. |
| Checked allocation arithmetic | [energy_tracking.cu](src/energy_tracking/energy_tracking.cu): `initialize_reduction_storage` | Put generic checked products, byte counts, and padding calculations into xpu's allocation facilities. VMC should supply its dimensions and retain physical shape checks. |
| Launch-size validation | [energy_tracking_kernels.hpp](src/energy_tracking/energy_tracking_kernels.hpp): `validate_initialization_size` | Validate launch dimensions and narrowing conversions in xpu's launch layer. Keep device-specific limits out of owning/public VMC APIs. |

## Existing xpu facilities that need a cleaner consumer interface

### Standard-library portability: `xstd::`

`xstd` already comes from xpu's `xpu/config.hpp`: it aliases `std` on CPU and `cuda::std` in CUDA builds. It was not introduced by VMC.

VMC currently uses `xstd::numbers::pi_v` and `xstd::numeric_limits` directly, and selects `<numbers>`/`<limits>` versus their CUDA equivalents in kernel headers:

- [energy_tracking_kernels.hpp](src/energy_tracking/energy_tracking_kernels.hpp)
- [simulation_kernels.hpp](src/simulation/simulation_kernels.hpp)

Expose the needed constants and numeric limits through xpu-owned headers and names, following its existing math wrappers. Possible interfaces are `xpu::numbers::pi_v<T>` and `xpu::numeric_limits<T>`; settle naming in xpu. Consumers should not need to select standard-library backends themselves.

### Portable accumulation

Energy, Slater, and Jastrow stencils currently select CUDA `atomicAdd` versus ordinary CPU addition with `__CUDA_ARCH__` guards.

Consider an xpu accumulation primitive with explicit concurrency semantics. An ordinary CPU addition is sufficient only when one CPU task owns the accumulator; it is not a general substitute for an atomic operation. VMC keeps the physical contribution calculations, and xpu supplies any reusable synchronization primitive.

## Needed for upcoming initialization work

### Lightweight batched SoA views

The inspected xpu `soa_batch` owns storage and exposes `view(batch)` for one batch. It does not expose a lightweight view of the complete batch collection.

Add a non-owning batch view containing base pointers, element/batch counts, and array/batch strides. It should derive per-batch views on either backend, preserve const-correctness and padding, and support exposing subsets of arrays.

VMC can then compose its own particle, energy, and Slater batch views without passing owning objects or uploading arrays of per-walker descriptors. Generic SoA addressing belongs in xpu; the component composition belongs in VMC.

### Whole-buffer and segmented reductions

The immediate migration is the cooperative block reduction above. A later `xpu::parallel_reduce` can own multi-block scheduling and partial storage for whole-buffer reductions. Independent segments would also support one result per walker.

Keep outputs backend-resident, provide explicit host retrieval separately, and make scratch lifetime and stream ordering clear. Energy prefactors and other physical formulas remain in VMC stencils.

### Batched or asynchronous LU and inversion

The current xpu LU interface is stateful and single-matrix. Keep host-dispatched GPU LU initially; do not add a custom VMC device LU.

A future xpu interface should support multiple matrices, padded leading dimensions, independent backend status storage, and controlled stream/handle ownership. Avoid requiring a synchronous host status copy after every matrix when the caller can consume status on the backend.

Compare sequential cuSOLVER, bounded concurrent cuSOLVER calls, and vendor batched factorization/inversion on real hardware. Do not assume batched LU is fastest for 485-by-485 matrices. Walker retry selection and the retry limit remain VMC responsibilities.

## Ownership and verification

- Keep Ewald/Jastrow/orbital formulas, walker state, Metropolis orchestration, and scientific validation in VMC.
- Implement and test generic primitives in xpu; VMC tests should cover observable behavior and numerical results, not duplicate xpu tests.
- Preserve existing RNG streams, precision support, and numerical behavior when migrating.
- CUDA compilation is not real-device verification. Reductions, synchronization, RNG seeding, and LU need device tests; use compute-sanitizer where applicable and profiling for performance claims.
