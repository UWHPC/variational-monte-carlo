#pragma once

#include <xpu/xpu.hpp>
#include <xpu/random.hpp>
#include "../energy_tracking/energy_tracking.hpp"
#include "../energy_tracking/energy_tracking_kernels.hpp"
#include "../jastrow_pade/jastrow_pade_kernels.hpp"
#include "../slater_plane_wave/slater_plane_wave.hpp"
#include "../slater_plane_wave/slater_plane_wave_kernels.hpp"
#include "../wavefunction/wavefunction_kernels.hpp"
#include "../utilities/execution.hpp"
#include "simulation.hpp"

#if defined(XPU_CUDA)
  #include <cuda/std/limits>
  #include <cuda/std/numbers>
#else
  #include <limits>
  #include <numbers>
#endif

#include <cstddef>
#include <cstdint>

namespace stencil {
namespace simulation {

DEVICE_ONLY
inline void seed_generator(
  xpu::random::generator* generator,
  std::uint64_t master_seed,
  std::uint64_t walker_id
) {
  generator->seed(master_seed, walker_id);
}

DEVICE_ONLY
inline void initialize_walker_positions(
  const Simulation::View simulation,
  fp_t box_length
) noexcept {
  auto& generator{*simulation.generator};
  auto positions{simulation.particles.pos};

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    for (auto particle{0uz}; particle < positions.count(); ++particle) {
      positions[axis][particle] = generator.uniform<fp_t>() * box_length;
    }
  }
}

[[nodiscard]] DEVICE_ONLY
inline Simulation::RandomProposal generate_random_proposal(
  xpu::random::generator& generator,
  std::size_t num_particles,
  fp_t step_size
) noexcept {
  const auto particle{generator.uniform_index(num_particles)};

  xpu::array<fp_t, idx(Axis::NUM)> displacement{};
  if (step_size > 0.0_fp) {
    for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
      displacement[axis] = generator.uniform(-step_size, step_size);
    }
  }

  return {
    particle,
    displacement,
    generator.uniform<fp_t>()
  };
}

DEVICE_ONLY
inline void propose_move(
  Simulation::View simulation,
  fp_t step_size,
  Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() != 0uz) { return; }

  auto& generator{*simulation.generator};
  const auto L{simulation.wave_function.jastrow.box_length};
  auto pos{simulation.particles.pos};
  auto local_generator{generator};
  scratch.proposal = generate_random_proposal(
    local_generator, pos.count(), step_size
  );
  generator = local_generator;

  const auto particle{scratch.proposal.particle};
  const auto inv_L{1.0_fp / L};

  scratch.result = {
    false,
    particle,
    {
      pos[idx(Axis::X)][particle],
      pos[idx(Axis::Y)][particle],
      pos[idx(Axis::Z)][particle]
    },
    {},
    0.0_fp,
    0.0_fp,
    0.0_fp
  };

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    auto new_position{
      scratch.result.old_pos[axis] + scratch.proposal.displacement[axis]
    };
    new_position -= L * xpu::floor(new_position * inv_L);

    pos[axis][particle] = new_position;
    scratch.result.new_pos[axis] = new_position;
  }

  scratch.slater_ratio = 0.0_fp;
  scratch.jastrow_delta = 0.0_fp;
}

CUDA_CALLABLE
inline void update_trig_cache(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  const auto slater{simulation.wave_function.slater};
  const auto particle{scratch.proposal.particle};
  const auto trig_row_offset{particle * slater.trig_row_stride};

  for (auto i{execution::thread()}; i < slater.num_unique_k; i += execution::stride()) {
    const auto trig_cache_index{trig_row_offset + i};
    slater.sin_saved[i] = slater.sin_cache[trig_cache_index];
    slater.cos_saved[i] = slater.cos_cache[trig_cache_index];

    stencil::slater::update_trig_cache(
      i,
      particle,
      slater,
      simulation.particles
    );
  }
}

CUDA_CALLABLE
inline void build_slater_row(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  const auto slater{simulation.wave_function.slater};
  const auto particle{scratch.proposal.particle};

  for (auto i{execution::thread()}; i < slater.num_orbitals; i += execution::stride()) {
    stencil::slater::build_row(
      i,
      particle,
      slater
    );
  }
}

CUDA_CALLABLE
inline void calculate_probability_ratio(
  Simulation::View simulation,
  Simulation::MetropolisScratch& scratch
) noexcept {
  const auto jastrow{simulation.wave_function.jastrow};
  const auto slater{simulation.wave_function.slater};
  const auto particle{scratch.proposal.particle};

  for (auto i{execution::thread()}; i < slater.num_orbitals; i += execution::stride()) {
    stencil::slater::determinant_ratio(
      i,
      particle,
      slater,
      &scratch.slater_ratio
    );
    stencil::jpade::delta_value(
      i,
      particle,
      jastrow,
      simulation.particles,
      scratch.result.old_pos,
      scratch.result.new_pos,
      &scratch.jastrow_delta
    );
  }
}

CUDA_CALLABLE
inline void decide_move(
  Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() != 0uz) { return; }

  const auto log_psi_delta{
    xpu::log(xpu::abs(scratch.slater_ratio)) + scratch.jastrow_delta
  };
  const auto log_ratio_sq{2.0_fp * log_psi_delta};
  const auto uniform{
    xpu::max(
      scratch.proposal.acceptance,
      xstd::numeric_limits<fp_t>::min()
    )
  };

  scratch.result.accepted = xpu::log(uniform) < xpu::min(0.0_fp, log_ratio_sq);
  scratch.result.log_psi_delta = log_psi_delta;
}

CUDA_CALLABLE
inline void reject_move(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  auto pos{simulation.particles.pos};
  const auto slater{simulation.wave_function.slater};
  const auto particle{scratch.proposal.particle};
  const auto trig_row_offset{particle * slater.trig_row_stride};

  for (auto i{execution::thread()}; i < slater.num_unique_k; i += execution::stride()) {
    const auto trig_cache_index{trig_row_offset + i};
    slater.sin_cache[trig_cache_index] = slater.sin_saved[i];
    slater.cos_cache[trig_cache_index] = slater.cos_saved[i];
  }

  if (execution::thread() == 0uz) {
    for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
      pos[axis][particle] = scratch.result.old_pos[axis];
    }
  }
}

CUDA_CALLABLE
inline void prepare_inverse_update(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  const auto slater{simulation.wave_function.slater};
  const auto particle{scratch.proposal.particle};
  const auto determinant_row_offset{particle * slater.matrix_row_stride};

  for (auto i{execution::thread()}; i < slater.num_orbitals; i += execution::stride()) {
    slater.inv_d_col[i] = slater.inv_determinant[determinant_row_offset + i];
    slater.solution[i] = 0.0_fp;
  }
}

CUDA_CALLABLE
inline void calculate_inverse_solution(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  const auto slater{simulation.wave_function.slater};
  const auto particle{scratch.proposal.particle};

  for (auto j{execution::thread()}; j < slater.num_orbitals; j += execution::stride()) {
    if (j == particle) { continue; }

    auto solution{0.0_fp};
    const auto inverse_row_offset{j * slater.matrix_row_stride};
    for (auto i{0uz}; i < slater.num_orbitals; ++i) {
      solution += slater.new_row[i] * slater.inv_determinant[inverse_row_offset + i];
    }
    slater.solution[j] = solution;
  }
}

CUDA_CALLABLE
inline void commit_slater_move(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  const auto slater{simulation.wave_function.slater};
  const auto particle{scratch.proposal.particle};
  const auto inv_ratio{1.0_fp / scratch.slater_ratio};
  const auto matrix_elements{slater.num_orbitals * slater.num_orbitals};

  for (auto element{execution::thread()}; element < matrix_elements; element += execution::stride()) {
    const auto j{element / slater.num_orbitals};
    const auto i{element - j * slater.num_orbitals};

    stencil::slater::k_update_inverse(
      i,
      j,
      particle,
      inv_ratio,
      slater
    );
  }

  const auto determinant_row_offset{particle * slater.matrix_row_stride};
  for (auto i{execution::thread()}; i < slater.num_orbitals; i += execution::stride()) {
    slater.determinant[determinant_row_offset + i] = slater.new_row[i];
  }
}

CUDA_CALLABLE
inline void update_structure_factors(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  const auto energy{simulation.energy_tracker};
  for (auto i{execution::thread()}; i < energy.num_g_vectors; i += execution::stride()) {
    stencil::energy::update_structure_factors(
      i,
      energy,
      scratch.result.old_pos,
      scratch.result.new_pos
    );
  }
}

CUDA_CALLABLE
inline void reset_energy_reductions(
  Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() != 0uz) { return; }
  scratch.real_energy_delta = 0.0_fp;
  scratch.reciprocal_sum = 0.0_fp;
}

CUDA_CALLABLE
inline void calculate_energy_deltas(
  Simulation::View simulation,
  Simulation::MetropolisScratch& scratch
) noexcept {
  const auto slater{simulation.wave_function.slater};
  const auto energy{simulation.energy_tracker};
  const auto particle{scratch.proposal.particle};

  for (auto i{execution::thread()}; i < energy.num_g_vectors; i += execution::stride()) {
    stencil::energy::initialize_reciprocal_energy(
      i,
      energy,
      &scratch.reciprocal_sum
    );
  }

  for (auto i{execution::thread()}; i < slater.num_orbitals; i += execution::stride()) {
    stencil::energy::update_real_energy(
      i,
      particle,
      energy,
      simulation.particles,
      scratch.result.old_pos,
      scratch.result.new_pos,
      &scratch.real_energy_delta
    );
  }
}

CUDA_CALLABLE
inline void finalize_energy(
  Simulation::View simulation,
  Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() != 0uz) { return; }

  const auto reciprocal_prefactor{
    1.0_fp / (
      2.0_fp * xstd::numbers::pi_v<fp_t>
      * simulation.energy_tracker.box_length
      * simulation.energy_tracker.box_length
      * simulation.energy_tracker.box_length
    )
  };
  scratch.result.real_energy_delta = scratch.real_energy_delta;
  scratch.result.reciprocal_energy = reciprocal_prefactor * scratch.reciprocal_sum;
}

CUDA_CALLABLE
inline void commit_energy(
  Simulation::View simulation,
  const Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() != 0uz) { return; }

  *simulation.energy_tracker.real_energy += scratch.result.real_energy_delta;
  *simulation.energy_tracker.reciprocal_energy = scratch.result.reciprocal_energy;
}

DEVICE_ONLY
inline void metropolis_step(
  Simulation::View simulation,
  fp_t step_size,
  Simulation::MetropolisScratch& scratch
) noexcept {
  propose_move(simulation, step_size, scratch);
  execution::sync();

  update_trig_cache(simulation, scratch);
  execution::sync();

  build_slater_row(simulation, scratch);
  execution::sync();

  calculate_probability_ratio(simulation, scratch);
  execution::sync();

  decide_move(scratch);
  execution::sync();

  if (!scratch.result.accepted) {
    reject_move(simulation, scratch);
    execution::sync();
    return;
  }

  prepare_inverse_update(simulation, scratch);
  execution::sync();

  calculate_inverse_solution(simulation, scratch);
  execution::sync();

  commit_slater_move(simulation, scratch);
  execution::sync();

  update_structure_factors(simulation, scratch);
  execution::sync();

  reset_energy_reductions(scratch);
  execution::sync();

  calculate_energy_deltas(simulation, scratch);
  execution::sync();

  finalize_energy(simulation, scratch);
  execution::sync();

  commit_energy(simulation, scratch);
  execution::sync();
}

CUDA_CALLABLE
inline void measure_walker(Simulation::View simulation) noexcept {
  stencil::wavefunction::evaluate_derivatives(
    simulation.wave_function,
    simulation.particles
  );
  execution::sync();

  stencil::energy::evaluate_local_energy(
    simulation.energy_tracker,
    simulation.particles,
    simulation.local_energy
  );
  execution::sync();
}

CUDA_CALLABLE
inline void initialize_walker_state(
  Simulation::WalkerState& state,
  fp_t step_size,
  fp_t box_length
) noexcept {
  constexpr auto maximum_step_fraction{0.5_fp};

  state = {};
  state.step_size = xpu::min(step_size, maximum_step_fraction * box_length);
}

CUDA_CALLABLE
inline void reset_walker_measurements(
  Simulation::WalkerState& state
) noexcept {
  const auto step_size{state.step_size};

  state = {};
  state.step_size = step_size;
}

CUDA_CALLABLE
inline void reset_sweep_result(
  Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() != 0uz) {
    return;
  }

  scratch.sweep_result = {};
}

CUDA_CALLABLE
inline void record_proposal(
  Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() != 0uz) {
    return;
  }

  ++scratch.sweep_result.proposed;
  scratch.sweep_result.accepted += scast<std::size_t>(scratch.result.accepted);
}

CUDA_CALLABLE
inline void update_walker_step_size(
  Simulation::WalkerState& state,
  const Simulation::SweepResult& result,
  std::size_t sweep,
  fp_t box_length
) noexcept {
  constexpr auto target_rate{0.5_fp};
  constexpr auto initial_gain{0.25_fp};
  constexpr auto maximum_step_fraction{0.5_fp};
  constexpr auto first_sweep{1uz};

  const auto acceptance_rate{result.acceptance_rate()};
  const auto gain{initial_gain * xpu::rsqrt(scast<fp_t>(sweep + first_sweep))};
  const auto maximum_step_size{maximum_step_fraction * box_length};

  state.step_size *= xpu::exp(gain * (acceptance_rate - target_rate));
  state.step_size = xpu::min(state.step_size, maximum_step_size);
}

CUDA_CALLABLE
inline void record_walker_sweep(
  Simulation::WalkerState& state,
  const Simulation::SweepResult& result
) noexcept {
  state.proposed += result.proposed;
  state.accepted += result.accepted;
}

CUDA_CALLABLE
inline void record_walker_energy(
  Simulation::WalkerState& state,
  fp_t local_energy,
  std::size_t block_size
) noexcept {
  state.energy_sum += local_energy;
  ++state.sample_count;

  state.block_sum += local_energy;
  ++state.samples_in_block;

  if (state.samples_in_block != block_size) {
    return;
  }

  const auto block_mean{state.block_sum / scast<fp_t>(block_size)};

  ++state.block_count;

  const auto delta{block_mean - state.blocked_mean};
  state.blocked_mean += delta / scast<fp_t>(state.block_count);
  state.blocked_m2 += delta * (block_mean - state.blocked_mean);

  state.block_sum = {};
  state.samples_in_block = {};
}

DEVICE_ONLY
inline void run_sweep(
  Simulation::View simulation,
  Simulation::WalkerState& state,
  const Simulation::RunConfig& config,
  Simulation::MetropolisScratch& scratch
) noexcept {
  reset_sweep_result(scratch);
  execution::sync();

  for (
    auto proposal{0uz};
    proposal < config.proposals_per_sweep;
    ++proposal
  ) {
    metropolis_step(
      simulation,
      state.step_size,
      scratch
    );

    record_proposal(scratch);
    execution::sync();
  }
}

DEVICE_ONLY
inline void run_walker(
  Simulation::View simulation,
  Simulation::WalkerState& state,
  const Simulation::RunConfig& config,
  Simulation::MetropolisScratch& scratch
) noexcept {
  if (execution::thread() == 0uz) {
    initialize_walker_state(
      state,
      config.initial_step_size,
      config.box_length
    );
  }
  execution::sync();

  for (auto sweep{0uz}; sweep < config.warmup_sweeps; ++sweep) {
    run_sweep(
      simulation,
      state,
      config,
      scratch
    );

    if (execution::thread() == 0uz) {
      update_walker_step_size(
        state,
        scratch.sweep_result,
        sweep,
        config.box_length
      );
    }
    execution::sync();
  }

  if (execution::thread() == 0uz) {
    reset_walker_measurements(state);
  }
  execution::sync();

  for (auto sweep{0uz}; sweep < config.measure_sweeps; ++sweep) {
    run_sweep(
      simulation,
      state,
      config,
      scratch
    );

    measure_walker(simulation);
    execution::sync();

    if (execution::thread() == 0uz) {
      record_walker_sweep(
        state,
        scratch.sweep_result
      );
      record_walker_energy(
        state,
        *simulation.local_energy,
        config.block_size
      );
    }
    execution::sync();
  }
}

CUDA_CALLABLE
inline void finalize_run(
  const Simulation::WalkerState* states,
  std::size_t walker_count,
  Simulation::RunResult* result
) noexcept {
  constexpr auto minimum_blocks{2uz};
  constexpr auto variance_correction{1uz};

  auto energy_sum{0.0_fp};
  auto blocked_mean{0.0_fp};
  auto blocked_m2{0.0_fp};

  auto proposed{0uz};
  auto accepted{0uz};
  auto sample_count{0uz};
  auto block_count{0uz};
  auto has_standard_error{true};

  for (auto walker{0uz}; walker < walker_count; ++walker) {
    const auto& state{states[walker]};

    proposed += state.proposed;
    accepted += state.accepted;
    energy_sum += state.energy_sum;
    sample_count += state.sample_count;

    if (state.block_count < minimum_blocks) {
      has_standard_error = false;
    }

    if (state.block_count == 0uz) {
      continue;
    }

    if (block_count == 0uz) {
      blocked_mean = state.blocked_mean;
      blocked_m2 = state.blocked_m2;
      block_count = state.block_count;
      continue;
    }

    const auto combined_block_count{block_count + state.block_count};
    const auto delta{state.blocked_mean - blocked_mean};
    const auto left_weight{scast<fp_t>(block_count)};
    const auto right_weight{scast<fp_t>(state.block_count)};
    const auto combined_weight{scast<fp_t>(combined_block_count)};

    blocked_mean += delta * right_weight / combined_weight;
    blocked_m2 += state.blocked_m2
      + delta * delta * left_weight * right_weight / combined_weight;
    block_count = combined_block_count;
  }

  *result = {
    .proposed = proposed,
    .accepted = accepted
  };

  if (sample_count != 0uz) {
    result->mean_energy = energy_sum / scast<fp_t>(sample_count);
  }

  if (proposed != 0uz) {
    result->acceptance_rate = scast<fp_t>(accepted) / scast<fp_t>(proposed);
  }

  if (!has_standard_error) {
    return;
  }

  const auto variance{blocked_m2 / scast<fp_t>(block_count - variance_correction)};

  result->mean_energy = blocked_mean;
  result->standard_error = xpu::sqrt(variance / scast<fp_t>(block_count));
  result->has_standard_error = true;
}

} // namespace stencil::simulation
} // namespace stencil

namespace kernel {
namespace simulation {

#if defined(XPU_CUDA)

namespace {

__global__
void cudaSeedGenerator(
  xpu::random::generator* generator,
  std::uint64_t master_seed,
  std::uint64_t walker_id
) {
  stencil::simulation::seed_generator(generator, master_seed, walker_id);
}

__global__
void cudaSeedGenerators(
  xpu::random::generator* generators,
  std::size_t walker_count,
  std::uint64_t master_seed,
  std::uint64_t first_walker_id
) {
  const auto [walker]{xpu::global_index<1>()};
  if (walker >= walker_count) { return; }

  stencil::simulation::seed_generator(
    generators + walker,
    master_seed,
    first_walker_id + walker
  );
}

__global__
void cudaInitializeWalkerPositions(
  const Simulation::View simulation,
  fp_t box_length
) {
  stencil::simulation::initialize_walker_positions(
    simulation,
    box_length
  );
}

__global__
void cudaInitializePositions(
  Simulation::BatchView simulations,
  fp_t box_length
) {
  const auto [walker]{xpu::global_index<1>()};
  if (walker >= simulations.walker_count()) { return; }

  const auto simulation{simulations.view(walker)};
  stencil::simulation::initialize_walker_positions(
    simulation,
    box_length
  );
}

__global__
void cudaMetropolisStep(
  Simulation::View simulation,
  fp_t step_size
) {
  __shared__ Simulation::MetropolisScratch scratch;

  stencil::simulation::metropolis_step(
    simulation,
    step_size,
    scratch
  );

  if (execution::thread() == 0uz) {
    *simulation.step_result = scratch.result;
  }
}

__global__
void cudaMetropolisSweep(
  Simulation::BatchView simulations,
  std::size_t proposals_per_walker,
  fp_t step_size,
  Simulation::SweepResult* result
) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  if (walker >= simulations.walker_count()) { return; }

  const auto simulation{simulations.view(walker)};

  __shared__ Simulation::MetropolisScratch scratch;
  __shared__ Simulation::SweepResult local_result;

  if (execution::thread() == 0uz) {
    local_result = {};
  }
  execution::sync();

  for (auto proposal{0uz}; proposal < proposals_per_walker; ++proposal) {
    stencil::simulation::metropolis_step(
      simulation,
      step_size,
      scratch
    );

    if (execution::thread() == 0uz) {
      ++local_result.proposed;
      local_result.accepted += scast<std::size_t>(scratch.result.accepted);
    }
    execution::sync();
  }

  if (execution::thread() == 0uz) {
    static_assert(sizeof(std::size_t) == sizeof(unsigned long long));
    atomicAdd(
      reinterpret_cast<unsigned long long*>(&result->proposed),
      scast<unsigned long long>(local_result.proposed)
    );
    atomicAdd(
      reinterpret_cast<unsigned long long*>(&result->accepted),
      scast<unsigned long long>(local_result.accepted)
    );
  }
}

__global__
void cudaMeasureWalkers(
  Simulation::BatchView simulations
) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  if (walker >= simulations.walker_count()) { return; }

  const auto simulation{simulations.view(walker)};
  stencil::simulation::measure_walker(simulation);
}

__global__
void cudaRunWalkers(
  Simulation::BatchView simulations,
  Simulation::WalkerState* states,
  Simulation::RunConfig config
) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  if (walker >= simulations.walker_count()) {
    return;
  }

  const auto simulation{simulations.view(walker)};
  __shared__ Simulation::MetropolisScratch scratch;

  stencil::simulation::run_walker(
    simulation,
    states[walker],
    config,
    scratch
  );
}

__global__
void cudaFinalizeRun(
  const Simulation::WalkerState* states,
  std::size_t walker_count,
  Simulation::RunResult* result
) {
  if (execution::thread() != 0uz || blockIdx.x != 0u) {
    return;
  }

  stencil::simulation::finalize_run(
    states,
    walker_count,
    result
  );
}

} // namespace

#endif

inline void seed_generator(
  xpu::random::generator* generator,
  std::uint64_t master_seed,
  std::uint64_t walker_id
) {
#if defined(XPU_CUDA)
  constexpr dim3 seedGeneratorThreads{1u};
  constexpr dim3 seedGeneratorBlocks{1u};

  cudaSeedGenerator<<<
    seedGeneratorBlocks, seedGeneratorThreads
  >>>(
    generator,
    master_seed,
    walker_id
  );
  xpu::cu_check(cudaGetLastError());
#else
  stencil::simulation::seed_generator(generator, master_seed, walker_id);
#endif
}

inline void seed_generators(
  xpu::random::generator* generators,
  std::size_t walker_count,
  std::uint64_t master_seed,
  std::uint64_t first_walker_id
) {
  if (walker_count == 0uz) { return; }

#if defined(XPU_CUDA)
  constexpr dim3 seedGeneratorsThreads{256u};
  const dim3 seedGeneratorsBlocks{
    xpu::block_per_dim(walker_count, seedGeneratorsThreads.x)
  };

  cudaSeedGenerators<<<
    seedGeneratorsBlocks, seedGeneratorsThreads
  >>>(
    generators,
    walker_count,
    master_seed,
    first_walker_id
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    stencil::simulation::seed_generator(
      generators + walker,
      master_seed,
      first_walker_id + walker
    );
  }
#endif
}

inline void initialize_positions(
  const Simulation::View simulation,
  fp_t box_length
) {
#if defined(XPU_CUDA)
  constexpr dim3 initializePositionsThreads{1u};
  constexpr dim3 initializePositionsBlocks{1u};

  cudaInitializeWalkerPositions<<<
    initializePositionsBlocks, initializePositionsThreads
  >>>(
    simulation,
    box_length
  );
  xpu::cu_check(cudaGetLastError());
#else
  stencil::simulation::initialize_walker_positions(
    simulation,
    box_length
  );
#endif
}

inline void initialize_positions(
  Simulation::BatchView simulations,
  fp_t box_length
) {
  const auto walker_count{simulations.walker_count()};

#if defined(XPU_CUDA)
  constexpr dim3 initializePositionsThreads{256u};
  const dim3 initializePositionsBlocks{
    xpu::block_per_dim(walker_count, initializePositionsThreads.x)
  };

  cudaInitializePositions<<<
    initializePositionsBlocks, initializePositionsThreads
  >>>(
    simulations,
    box_length
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    stencil::simulation::initialize_walker_positions(
      simulation,
      box_length
    );
  }
#endif
}

inline Simulation::StepResult metropolis_step(
  Simulation::View simulation,
  fp_t step_size
) {
#if defined(XPU_CUDA)
  dim3 metropolisStepThreads{256u};
  dim3 metropolisStepBlocks{1u};
  cudaMetropolisStep<<<
    metropolisStepBlocks, metropolisStepThreads
  >>>(
    simulation,
    step_size
  );
  xpu::cu_check(cudaGetLastError());

  Simulation::StepResult result{};
  xpu::copy_n(&result, simulation.step_result, 1uz);
  return result;
#else
  Simulation::MetropolisScratch scratch{};
  stencil::simulation::metropolis_step(
    simulation,
    step_size,
    scratch
  );
  *simulation.step_result = scratch.result;
  return scratch.result;
#endif
}

inline void metropolis_sweep(
  Simulation::BatchView simulations,
  std::size_t proposals_per_walker,
  fp_t step_size,
  Simulation::SweepResult* result
) {
  const auto walker_count{simulations.walker_count()};
  xpu::memset(result, 0, sizeof(Simulation::SweepResult));

#if defined(XPU_CUDA)
  dim3 metropolisSweepThreads{256u};
  dim3 metropolisSweepBlocks{scast<unsigned int>(walker_count)};
  cudaMetropolisSweep<<<
    metropolisSweepBlocks, metropolisSweepThreads
  >>>(
    simulations,
    proposals_per_walker,
    step_size,
    result
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    Simulation::SweepResult local_result{};

    for (auto proposal{0uz}; proposal < proposals_per_walker; ++proposal) {
      Simulation::MetropolisScratch scratch{};
      stencil::simulation::metropolis_step(
        simulation,
        step_size,
        scratch
      );

      ++local_result.proposed;
      local_result.accepted += scast<std::size_t>(scratch.result.accepted);
    }

    result->proposed += local_result.proposed;
    result->accepted += local_result.accepted;
  }
#endif
}

inline void measure_walkers(
  Simulation::BatchView simulations
) {
  const auto walker_count{simulations.walker_count()};

#if defined(XPU_CUDA)
  dim3 measureWalkersThreads{256u};
  dim3 measureWalkersBlocks{scast<unsigned int>(walker_count)};
  cudaMeasureWalkers<<<
    measureWalkersBlocks, measureWalkersThreads
  >>>(
    simulations
  );
  xpu::cu_check(cudaGetLastError());
#else
  for (auto walker{0uz}; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    stencil::simulation::measure_walker(simulation);
  }
#endif
}

inline Simulation::RunResult run_walkers(
  Simulation::BatchView simulations,
  Simulation::WalkerState* states,
  const Simulation::RunConfig& config,
  Simulation::RunResult* result_storage
) {
  const auto walker_count{simulations.walker_count()};

#if defined(XPU_CUDA)
  constexpr dim3 runWalkersThreads{256u};
  const dim3 runWalkersBlocks{scast<unsigned int>(walker_count)};

  cudaRunWalkers<<<
    runWalkersBlocks, runWalkersThreads
  >>>(
    simulations,
    states,
    config
  );
  xpu::cu_check(cudaGetLastError());

  constexpr dim3 finalizeRunThreads{1u};
  constexpr dim3 finalizeRunBlocks{1u};

  cudaFinalizeRun<<<
    finalizeRunBlocks, finalizeRunThreads
  >>>(
    states,
    walker_count,
    result_storage
  );
  xpu::cu_check(cudaGetLastError());
#else
  #if defined(_OPENMP)
    #pragma omp parallel for num_threads(config.num_threads)
  #endif
  for (auto walker = 0uz; walker < walker_count; ++walker) {
    const auto simulation{simulations.view(walker)};
    Simulation::MetropolisScratch scratch{};

    stencil::simulation::run_walker(
      simulation,
      states[walker],
      config,
      scratch
    );
  }

  stencil::simulation::finalize_run(
    states,
    walker_count,
    result_storage
  );
#endif

  Simulation::RunResult result{};
  xpu::copy_n(&result, result_storage, 1uz);
  return result;
}

} // namespace kernel::simulation
} // namespace kernel
