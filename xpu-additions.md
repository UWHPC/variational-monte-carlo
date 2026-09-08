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

The inspected xpu checkout already provides `xpu::atomic_add`. Audit its concurrency semantics when migrating remaining guards. An ordinary CPU addition is sufficient only when one CPU task owns the accumulator; it is not a general substitute for an atomic operation. Prefer cooperative reduction where many threads contribute to one sum. VMC keeps the physical contribution calculations.

## Needed for upcoming initialization work

### Lightweight batched SoA views

Implemented in the inspected xpu checkout as `soa_batch_view` and already consumed by VMC's composed batch views. Generic SoA addressing belongs in xpu; the component composition belongs in VMC. This is no longer an outstanding addition.

### Whole-buffer and segmented reductions

`xpu::parallel_reduce_sum` now exists and VMC uses it for host-dispatched Slater reductions on both backends. It is distinct from a block reduction callable inside a resident kernel. Independent segments remain a candidate for producing one result per walker without separate host dispatches.

Keep outputs backend-resident, provide explicit host retrieval separately, and make scratch lifetime and stream ordering clear. Energy prefactors and other physical formulas remain in VMC stencils.

### Reduction workspace without a raw-byte contract

Keep the operation a function. Proposed consumer API:

```cpp
xpu::reduction_workspace<fp_t> workspace;

xpu::prepare_reduce_sum(workspace, range, contribution);
xpu::parallel_reduce_sum(range, result, contribution, workspace);
```

- Preparation queries and allocates backend scratch once, using the actual contribution type; it must not read input elements or retain input pointers.
- Execution reuses storage without allocating. Require explicit preparation again when the workload exceeds the prepared contract, and report incompatible workspace use clearly.
- xpu owns byte counts, alignment, capacity checks, and backend details. CPU workspace can have no scratch allocation. The raw pointer/byte API can remain underneath for advanced callers.
- Specify stream/device compatibility and prohibit overlapping reuse of one workspace unless explicitly supported. Independent concurrent operations need independent workspaces.
- VMC owners retain the workspace; kernel wrappers receive it. No per-call allocation or workspace metadata in the public physics API.

### Block reduction inside resident kernels

Provide a device-callable `xpu::block_reduce_sum` with scalar CPU semantics. Each thread first accumulates its assigned elements locally, the block reduces those contributions, and one designated thread writes the result.

This can replace per-particle kinetic-energy atomics in `evaluate_local_energy` and other resident sums after their dependencies are audited. Host-dispatched `parallel_reduce_sum` cannot perform this job inside an already-running walker kernel.

- Encapsulate barriers and backend intrinsics in xpu; keep the numerical contribution stencil common to CPU and CUDA.
- Hide raw scratch-byte bookkeeping behind typed block storage or an execution context. Specify storage lifetime, reuse synchronization, supported block widths, and which threads receive a valid result.
- Require uniform participation by the block; inactive lanes contribute zero. On the scalar CPU path, return the thread-local contribution.
- No dynamic allocation or implicit host launch. Preserve the caller's one-block-per-walker mapping for the sequential chain.
- Test precision, partial input tiles, supported block sizes, and repeated storage reuse in xpu on real CUDA hardware.

### CPU reduction execution policy

Deferred for later tuning, but belongs in xpu. The inspected implementation enters an OpenMP parallel reduction for every nonempty range. The local perf audit found substantial overhead for repeated 485-element host-dispatched reductions, including with one OpenMP thread.

Add a cheap serial/vectorizable path for small ranges and a measured threshold for parallel execution. Define thread-budget and nested-parallelism behavior so reductions do not oversubscribe surrounding walker work. Benchmark larger ranges and contribution costs before choosing a policy; those crossover measurements have not been performed. Keep VMC on the same reduction function for both backends.

### Optional squared-norm math helper

The inspected math header has `norm3d`, which includes a square root, but no squared-norm helper. A proposed `xpu::norm_squared3d(x, y, z)` could express `x*x + y*y + z*z` without that square root. This is an optional convenience, not a prerequisite or demonstrated performance improvement. Decide floating-point contraction semantics in xpu; VMC can retain the explicit expression meanwhile.

### Batched or asynchronous LU and inversion

The current xpu LU interface is stateful and single-matrix. Keep host-dispatched GPU LU initially; do not add a custom VMC device LU.

First prerequisite: factorization and inversion must accept backend-resident status output without requiring a synchronous host status copy. Specify status meanings, pivot lifetime, operation ordering, and handling of singular matrices. Preserve the existing synchronous convenience API. Distinguish asynchronous numerical status from host-visible library/launch failures.

VMC can then keep per-walker LU status and log determinants on the backend, classify failed walkers there, and retrieve one aggregate failure status in the common initialization path. VMC owns failed-walker selection, deterministic RNG streams, selective regeneration, and the existing 100-attempt limit. This removes per-walker scalar synchronization without requiring batched LU immediately.

Next addition: batched factorization and inversion with one status per matrix. Support multiple matrices, padded leading dimensions, batch strides or backend pointer arrays, backend pivot/output storage, and explicit stream/handle ownership. Define workspace sizing and reuse through an xpu-owned interface; avoid allocation during repeated execution. Define how invalid matrices are excluded from inversion and how callers submit only failed matrices for retries.

Compare sequential cuSOLVER, bounded concurrent cuSOLVER calls, and vendor batched factorization/inversion on real hardware. Do not assume batched LU is fastest for 485-by-485 matrices. Walker retry selection and the retry limit remain VMC responsibilities.

## Ownership and verification

- Keep Ewald/Jastrow/orbital formulas, walker state, Metropolis orchestration, and scientific validation in VMC.
- Implement and test generic primitives in xpu; VMC tests should cover observable behavior and numerical results, not duplicate xpu tests.
- Preserve existing RNG streams, precision support, and numerical behavior when migrating.
- CUDA compilation is not real-device verification. Reductions, synchronization, RNG seeding, and LU need device tests; use compute-sanitizer where applicable and profiling for performance claims.
