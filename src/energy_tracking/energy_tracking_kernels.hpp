#pragma once

#include <xpu/xpu.hpp>
#include "energy_tracking.hpp"
#include "../utilities/execution.hpp"
#include "../utilities/components.hpp"
#include "../utilities/macros.hpp"

#if defined(XPU_CUDA)
  #include <cuda/std/numbers>
#else
  #include <numbers>
#endif

namespace stencil {
namespace energy {

namespace {

CUDA_CALLABLE
inline fp_t evaluate_pair_energy(
  std::size_t other,
  fp_t L, fp_t half_L, fp_t alpha,
  const xpu::array<fp_t, idx(Axis::NUM)>& particle_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos
) noexcept {
  xpu::array<fp_t, idx(Axis::NUM)> displacement{};

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    auto delta{particle_pos[axis] - pos[axis][other]};

    delta += L * (delta <= -half_L) - L * (delta > half_L);

    displacement[axis] = delta;
  }

  const auto distance{
    xpu::sqrt(
      displacement[idx(Axis::X)] * displacement[idx(Axis::X)] +
      displacement[idx(Axis::Y)] * displacement[idx(Axis::Y)] +
      displacement[idx(Axis::Z)] * displacement[idx(Axis::Z)]
    )
  };
  const auto inverse_distance{
    (distance < 1e-12_fp) ? 1.0_fp : 1.0_fp / distance
  };

  return xpu::erfc(alpha * distance) * inverse_distance;
}

} // namespace

CUDA_CALLABLE
inline void initialize_reciprocal_energy(
  std::size_t i,
  const fp_t* RESTRICT g_weights,
  const fp_t* RESTRICT sum_real,
  const fp_t* RESTRICT sum_imag,
  fp_t* RESTRICT reciprocal_sum
) noexcept {
  const auto contribution{
    g_weights[i] * (
      sum_real[i] * sum_real[i] +
      sum_imag[i] * sum_imag[i]
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(reciprocal_sum, contribution);
#else
  *reciprocal_sum += contribution;
#endif
}

CUDA_CALLABLE
inline void update_real_energy(
  std::size_t other, std::size_t moved,
  fp_t L, fp_t half_L, fp_t alpha,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* RESTRICT delta
) noexcept {
  if (other == moved) { return; }

  const auto old_pair_energy{
    evaluate_pair_energy(
      other,
      L, half_L, alpha,
      old_pos, pos
    )
  };
  const auto new_pair_energy{
    evaluate_pair_energy(
      other,
      L, half_L, alpha,
      new_pos, pos
    )
  };

#if defined(__CUDA_ARCH__)
  atomicAdd(delta, new_pair_energy - old_pair_energy);
#else
  *delta += new_pair_energy - old_pair_energy;
#endif
}

CUDA_CALLABLE
inline fp_t kinetic_energy_contribution(
  std::size_t i,
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives
) noexcept {
  const auto gradient_x{derivatives[idx(Derivatives::GRAD_X)][i]};
  const auto gradient_y{derivatives[idx(Derivatives::GRAD_Y)][i]};
  const auto gradient_z{derivatives[idx(Derivatives::GRAD_Z)][i]};
  const auto gradient_squared{
    gradient_x * gradient_x +
    gradient_y * gradient_y +
    gradient_z * gradient_z
  };

  return derivatives[idx(Derivatives::LAP)][i] + gradient_squared;
}

CUDA_CALLABLE
inline void kinetic_energy(
  std::size_t i,
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives,
  fp_t* RESTRICT kinetic_sum
) noexcept {
  const auto contribution{kinetic_energy_contribution(i, derivatives)};

#if defined(__CUDA_ARCH__)
  atomicAdd(kinetic_sum, contribution);
#else
  *kinetic_sum += contribution;
#endif
}

CUDA_CALLABLE
inline void kinetic_energy(
  std::size_t particle,
  EnergyTracker::View energy,
  Particles::View particles
) noexcept;

CUDA_CALLABLE
inline void evaluate_local_energy(
  EnergyTracker::View energy,
  Particles::View particles,
  fp_t* local_energy
) noexcept {
  if (execution::thread() == 0uz) {
    *energy.reduction_scratch = 0.0_fp;
  }
  execution::sync();

  for (auto particle{execution::thread()}; particle < particles.count; particle += execution::stride()) {
    kinetic_energy(particle, energy, particles);
  }
  execution::sync();

  if (execution::thread() == 0uz) {
    *local_energy =
      -0.5_fp * *energy.reduction_scratch
      + *energy.real_energy
      + *energy.reciprocal_energy
      + energy.ewald_correction
      + energy.ewald_background;
  }
  execution::sync();
}

CUDA_CALLABLE
inline void update_structure_factors(
  std::size_t i,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector,
  fp_t* RESTRICT sum_real,
  fp_t* RESTRICT sum_imag
) noexcept {
  const auto old_dot{
    g_vector[idx(Axis::X)][i] * old_pos[idx(Axis::X)] +
    g_vector[idx(Axis::Y)][i] * old_pos[idx(Axis::Y)] +
    g_vector[idx(Axis::Z)][i] * old_pos[idx(Axis::Z)]
  };
  const auto new_dot{
    g_vector[idx(Axis::X)][i] * new_pos[idx(Axis::X)] +
    g_vector[idx(Axis::Y)][i] * new_pos[idx(Axis::Y)] +
    g_vector[idx(Axis::Z)][i] * new_pos[idx(Axis::Z)]
  };

  auto old_sin{0.0_fp};
  auto old_cos{0.0_fp};
  auto new_sin{0.0_fp};
  auto new_cos{0.0_fp};

  xpu::sincos(old_dot, &old_sin, &old_cos);
  xpu::sincos(new_dot, &new_sin, &new_cos);

  sum_real[i] += new_cos - old_cos;
  sum_imag[i] += new_sin - old_sin;
}

CUDA_CALLABLE
inline void initialize_reciprocal_energy(
  std::size_t g_index,
  EnergyTracker::View energy,
  fp_t* reciprocal_sum
) noexcept {
  initialize_reciprocal_energy(
    g_index,
    energy.g_weights,
    energy.sum_real,
    energy.sum_imag,
    reciprocal_sum
  );
}

CUDA_CALLABLE
inline void update_structure_factors(
  std::size_t g_index,
  EnergyTracker::View energy,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos
) noexcept {
  update_structure_factors(
    g_index,
    old_pos,
    new_pos,
    energy.g_vector,
    energy.sum_real,
    energy.sum_imag
  );
}

CUDA_CALLABLE
inline void update_real_energy(
  std::size_t other,
  std::size_t moved,
  EnergyTracker::View energy,
  Particles::View particles,
  const xpu::array<fp_t, idx(Axis::NUM)>& old_pos,
  const xpu::array<fp_t, idx(Axis::NUM)>& new_pos,
  fp_t* delta
) noexcept {
  update_real_energy(
    other,
    moved,
    energy.box_length,
    0.5_fp * energy.box_length,
    energy.ewald_alpha,
    old_pos,
    new_pos,
    particles.pos,
    delta
  );
}

CUDA_CALLABLE
inline void kinetic_energy(
  std::size_t particle,
  EnergyTracker::View energy,
  Particles::View particles
) noexcept {
  kinetic_energy(
    particle,
    particles.derivatives,
    energy.reduction_scratch
  );
}

CUDA_CALLABLE
inline fp_t reciprocal_contribution(std::size_t g, EnergyTracker::View energy) noexcept {
  return energy.g_weights[g] * (
    energy.sum_real[g] * energy.sum_real[g] +
    energy.sum_imag[g] * energy.sum_imag[g]
  );
}

CUDA_CALLABLE
inline void initialize_structure_factor(
  std::size_t g,
  EnergyTracker::View energy,
  Particles::View particles
) noexcept {
  const auto g_x{energy.g_vector[idx(Axis::X)][g]};
  const auto g_y{energy.g_vector[idx(Axis::Y)][g]};
  const auto g_z{energy.g_vector[idx(Axis::Z)][g]};
  auto real_sum{0.0_fp};
  auto imag_sum{0.0_fp};

  for (auto particle{0uz}; particle < particles.count; ++particle) {
    const auto phase{
      g_x * particles.pos[idx(Axis::X)][particle] +
      g_y * particles.pos[idx(Axis::Y)][particle] +
      g_z * particles.pos[idx(Axis::Z)][particle]
    };
    auto sine{0.0_fp};
    auto cosine{0.0_fp};
    xpu::sincos(phase, &sine, &cosine);
    real_sum += cosine;
    imag_sum += sine;
  }

  energy.sum_real[g] = real_sum;
  energy.sum_imag[g] = imag_sum;
}

CUDA_CALLABLE
inline void store_partial_sum(fp_t contribution, fp_t* scratch, fp_t* result) noexcept {
  const auto thread{execution::thread()};
  scratch[thread] = contribution;
  execution::sync();

  for (auto offset{execution::stride() / 2uz}; offset > 0uz; offset /= 2uz) {
    if (thread < offset) {
      scratch[thread] += scratch[thread + offset];
    }
    execution::sync();
  }

  if (thread == 0uz) {
    *result = scratch[0uz];
  }
  execution::sync();
}

CUDA_CALLABLE
inline void initialize_reciprocal_partial(
  EnergyTracker::InitializationView state,
  std::size_t begin,
  std::size_t end,
  bool build_structure,
  fp_t* scratch,
  fp_t* partial_sum
) noexcept {
  auto contribution{0.0_fp};
  for (auto g{begin + execution::thread()}; g < end; g += execution::stride()) {
    if (build_structure) {
      initialize_structure_factor(g, state.energy, state.particles);
    }
    contribution += reciprocal_contribution(g, state.energy);
  }
  store_partial_sum(contribution, scratch, partial_sum);
}

CUDA_CALLABLE
inline void initialize_real_partial(
  EnergyTracker::InitializationView state,
  std::size_t begin,
  std::size_t end,
  fp_t* scratch,
  fp_t* partial_sum
) noexcept {
  const auto energy{state.energy};
  const auto particles{state.particles};
  auto contribution{0.0_fp};

  for (auto element{begin + execution::thread()}; element < end; element += execution::stride()) {
    const auto first{element / particles.count};
    const auto second{element - first * particles.count};
    if (first >= second) { continue; }

    const xpu::array<fp_t, idx(Axis::NUM)> particle_pos{
      particles.pos[idx(Axis::X)][first],
      particles.pos[idx(Axis::Y)][first],
      particles.pos[idx(Axis::Z)][first]
    };
    contribution += evaluate_pair_energy(
      second,
      energy.box_length, 0.5_fp * energy.box_length, energy.ewald_alpha,
      particle_pos, particles.pos
    );
  }
  store_partial_sum(contribution, scratch, partial_sum);
}

CUDA_CALLABLE
inline void finalize_initial_energy(
  EnergyTracker::InitializationView state,
  std::size_t reciprocal_partial_count,
  std::size_t real_partial_count,
  fp_t* scratch
) noexcept {
  if (reciprocal_partial_count > 0uz) {
    auto sum{0.0_fp};
    for (auto partial{execution::thread()}; partial < reciprocal_partial_count; partial += execution::stride()) {
      sum += state.reciprocal_partials[partial];
    }
    const auto length{state.energy.box_length};
    const auto prefactor{1.0_fp / (2.0_fp * xstd::numbers::pi_v<fp_t> * length * length * length)};
    store_partial_sum(prefactor * sum, scratch, state.energy.reciprocal_energy);
  }
  if (real_partial_count > 0uz) {
    auto sum{0.0_fp};
    for (auto partial{execution::thread()}; partial < real_partial_count; partial += execution::stride()) {
      sum += state.real_partials[partial];
    }
    store_partial_sum(sum, scratch, state.energy.real_energy);
  }
}

CUDA_CALLABLE
inline void accept_move(
  EnergyTracker::View energy,
  fp_t real_delta,
  fp_t reciprocal,
  bool update_reciprocal
) noexcept {
  *energy.real_energy += real_delta;
  if (update_reciprocal) {
    *energy.reciprocal_energy = reciprocal;
  }
}

} // namespace stencil::energy
} // namespace stencil

namespace kernel {
namespace energy {

namespace {

struct KineticEnergyContribution {
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives{
    nullptr, 0uz
  };

  [[nodiscard]] CUDA_CALLABLE
  fp_t operator()(const xpu::array<std::size_t, 1uz>& index) const {
    return stencil::energy::kinetic_energy_contribution(
      index[0uz],
      derivatives
    );
  }
};

#if defined(XPU_CUDA)
__global__
void cudaUpdateRealEnergy(
  std::size_t moved,
  fp_t L, fp_t half_L, fp_t alpha,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* RESTRICT delta
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= pos.count()) { return; }

  xpu::array<fp_t, idx(Axis::NUM)> new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  stencil::energy::update_real_energy(
    i, moved,
    L, half_L, alpha,
    old_pos, new_pos, pos,
    delta
  );
}

__global__
void cudaUpdateStructureFactors(
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::array<fp_t, idx(Axis::NUM)> new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector,
  fp_t* RESTRICT sum_real,
  fp_t* RESTRICT sum_imag
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= g_vector.count()) { return; }

  stencil::energy::update_structure_factors(
    i,
    old_pos, new_pos,
    g_vector,
    sum_real, sum_imag
  );
}
#endif

} // namespace

inline fp_t update_real_energy(
  std::size_t moved,
  fp_t L, fp_t half_L, fp_t alpha,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> pos,
  fp_t* RESTRICT delta_scratch
) noexcept {
#if defined(XPU_CUDA)
  xpu::zero_n(delta_scratch, 1uz);

  dim3 updateRealEnergyThreads{256u};
  dim3 updateRealEnergyBlocks{
    xpu::block_per_dim(pos.count(), updateRealEnergyThreads.x)
  };

  cudaUpdateRealEnergy<<<
    updateRealEnergyBlocks, updateRealEnergyThreads
  >>>(
    moved,
    L, half_L, alpha,
    old_pos, pos,
    delta_scratch
  );
  xpu::cu_check(cudaGetLastError());

  auto delta{0.0_fp};
  xpu::copy_n(&delta, delta_scratch, 1uz);
  return delta;
#else
  scast<void>(delta_scratch);

  xpu::array<fp_t, idx(Axis::NUM)> new_pos{
    pos[idx(Axis::X)][moved],
    pos[idx(Axis::Y)][moved],
    pos[idx(Axis::Z)][moved]
  };

  auto delta{0.0_fp};

  #pragma omp simd reduction(+ : delta)
  for (auto i = 0uz; i < pos.count(); ++i) {
    stencil::energy::update_real_energy(
      i, moved,
      L, half_L, alpha,
      old_pos, new_pos, pos,
      &delta
    );
  }

  return delta;
#endif
}

inline std::size_t kinetic_energy_scratch_bytes(
  std::size_t num_particles
) {
  const xpu::range<1uz> range{
    {0uz},
    {num_particles},
    {1uz}
  };
  const KineticEnergyContribution contribution{
    xpu::soa_view<fp_t, idx(Derivatives::NUM)>{nullptr, num_particles}
  };

  return xpu::parallel_reduce_sum_bytes<fp_t>(range, contribution);
}

inline fp_t kinetic_energy(
  xpu::soa_view<fp_t, idx(Derivatives::NUM)> derivatives,
  fp_t* RESTRICT kinetic_sum_scratch,
  void* scratch,
  std::size_t scratch_bytes
) noexcept {
  const xpu::range<1uz> range{
    {0uz},
    {derivatives.count()},
    {1uz}
  };
  const KineticEnergyContribution contribution{derivatives};

  xpu::parallel_reduce_sum(
    range,
    kinetic_sum_scratch,
    contribution,
    scratch,
    scratch_bytes
  );

  auto kinetic_sum{0.0_fp};
  xpu::copy_n(&kinetic_sum, kinetic_sum_scratch, 1uz);
  return -0.5_fp * kinetic_sum;
}

inline void update_structure_factors(
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::array<fp_t, idx(Axis::NUM)> new_pos,
  xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector,
  fp_t* RESTRICT sum_real,
  fp_t* RESTRICT sum_imag
) noexcept {
#if defined(XPU_CUDA)
  dim3 updateStructureFactorsThreads{256u};
  dim3 updateStructureFactorsBlocks{
    xpu::block_per_dim(
      g_vector.count(), updateStructureFactorsThreads.x
    )
  };

  cudaUpdateStructureFactors<<<
    updateStructureFactorsBlocks, updateStructureFactorsThreads
  >>>(
    old_pos, new_pos,
    g_vector,
    sum_real, sum_imag
  );
  xpu::cu_check(cudaGetLastError());
#else
  #pragma omp simd
  for (auto i = 0uz; i < g_vector.count(); ++i) {
    stencil::energy::update_structure_factors(
      i,
      old_pos, new_pos,
      g_vector,
      sum_real, sum_imag
    );
  }
#endif
}

inline fp_t update_real_energy(
  EnergyTracker::View energy,
  std::size_t moved,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  Particles::View particles
) noexcept {
  return update_real_energy(
    moved,
    energy.box_length,
    0.5_fp * energy.box_length,
    energy.ewald_alpha,
    old_pos,
    particles.pos,
    energy.reduction_scratch
  );
}

inline fp_t kinetic_energy(
  EnergyTracker::View energy,
  Particles::View particles,
  void* scratch,
  std::size_t scratch_bytes
) noexcept {
  return kinetic_energy(
    particles.derivatives,
    energy.reduction_scratch,
    scratch,
    scratch_bytes
  );
}

inline void update_structure_factors(
  EnergyTracker::View energy,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::array<fp_t, idx(Axis::NUM)> new_pos
) noexcept {
  update_structure_factors(
    old_pos,
    new_pos,
    energy.g_vector,
    energy.sum_real,
    energy.sum_imag
  );
}

namespace {

constexpr auto initialization_threads{128uz};
constexpr auto initialization_elements{512uz};

enum class InitializationMode { ALL, STRUCTURE, RECIPROCAL, REAL };

#if defined(XPU_CUDA)
template <typename States>
__global__
void cudaInitializeReciprocalPartials(States states, bool build_structure) {
  const auto partial{scast<std::size_t>(blockIdx.x)};
  const auto walker{scast<std::size_t>(blockIdx.y)};
  const auto state{states[walker]};
  const auto begin{partial * initialization_elements};
  const auto end{xpu::min(begin + initialization_elements, state.energy.num_g_vectors)};
  __shared__ fp_t scratch[initialization_threads];

  stencil::energy::initialize_reciprocal_partial(
    state, begin, end, build_structure, scratch, state.reciprocal_partials + partial
  );
}

template <typename States>
__global__
void cudaInitializeRealPartials(States states) {
  const auto partial{scast<std::size_t>(blockIdx.x)};
  const auto walker{scast<std::size_t>(blockIdx.y)};
  const auto state{states[walker]};
  const auto begin{partial * initialization_elements};
  const auto end{xpu::min(begin + initialization_elements, state.particles.count * state.particles.count)};
  __shared__ fp_t scratch[initialization_threads];

  stencil::energy::initialize_real_partial(state, begin, end, scratch, state.real_partials + partial);
}

template <typename States>
__global__
void cudaFinalizeInitialEnergy(States states, std::size_t reciprocal_count, std::size_t real_count) {
  const auto walker{scast<std::size_t>(blockIdx.x)};
  __shared__ fp_t scratch[initialization_threads];
  stencil::energy::finalize_initial_energy(states[walker], reciprocal_count, real_count, scratch);
}
#endif

} // namespace

inline void validate_initialization_size(std::size_t walker_count, std::size_t partial_count) {
#if defined(XPU_CUDA)
  constexpr auto maximum_walker_blocks{65535uz};
  constexpr auto maximum_partial_blocks{2147483647uz};
  if (walker_count > maximum_walker_blocks || partial_count > maximum_partial_blocks) {
    throw std::length_error("Energy initialization exceeds CUDA grid limits");
  }
#else
  scast<void>(walker_count);
  scast<void>(partial_count);
#endif
}

inline std::size_t initialization_partial_count(std::size_t elements) noexcept {
  return xpu::max(1uz, elements / initialization_elements + scast<std::size_t>(elements % initialization_elements != 0uz));
}

namespace {

template <InitializationMode mode, typename States>
inline void initialize(
  States states,
  std::size_t walker_count,
  std::size_t particle_count,
  std::size_t g_count,
  std::size_t num_threads
) noexcept {
  if (walker_count == 0uz) { return; }
  constexpr auto build_structure{mode == InitializationMode::ALL || mode == InitializationMode::STRUCTURE};
  constexpr auto build_reciprocal{mode != InitializationMode::REAL};
  constexpr auto build_real{mode == InitializationMode::ALL || mode == InitializationMode::REAL};
  const auto reciprocal_count{build_reciprocal ? initialization_partial_count(g_count) : 0uz};
  const auto real_count{build_real ? initialization_partial_count(particle_count * particle_count) : 0uz};
  const auto finalize_reciprocal_count{mode == InitializationMode::STRUCTURE ? 0uz : reciprocal_count};

#if defined(XPU_CUDA)
  scast<void>(num_threads);
  if constexpr (build_reciprocal) {
    constexpr dim3 initializeReciprocalPartialsThreads{scast<unsigned int>(initialization_threads)};
    const dim3 initializeReciprocalPartialsBlocks{
      scast<unsigned int>(reciprocal_count), scast<unsigned int>(walker_count)
    };
    cudaInitializeReciprocalPartials<<<
      initializeReciprocalPartialsBlocks, initializeReciprocalPartialsThreads
    >>>(states, build_structure);
    xpu::cu_check(cudaGetLastError());
  }
  if constexpr (build_real) {
    constexpr dim3 initializeRealPartialsThreads{scast<unsigned int>(initialization_threads)};
    const dim3 initializeRealPartialsBlocks{
      scast<unsigned int>(real_count), scast<unsigned int>(walker_count)
    };
    cudaInitializeRealPartials<<<
      initializeRealPartialsBlocks, initializeRealPartialsThreads
    >>>(states);
    xpu::cu_check(cudaGetLastError());
  }
  if constexpr (mode != InitializationMode::STRUCTURE) {
    constexpr dim3 finalizeInitialEnergyThreads{scast<unsigned int>(initialization_threads)};
    const dim3 finalizeInitialEnergyBlocks{scast<unsigned int>(walker_count)};
    cudaFinalizeInitialEnergy<<<
      finalizeInitialEnergyBlocks, finalizeInitialEnergyThreads
    >>>(states, finalize_reciprocal_count, real_count);
    xpu::cu_check(cudaGetLastError());
  }
#else
  scast<void>(num_threads);
  #pragma omp parallel for num_threads(num_threads)
  for (auto walker = 0uz; walker < walker_count; ++walker) {
    const auto state{states[walker]};
    auto scratch{0.0_fp};
    for (auto partial{0uz}; partial < reciprocal_count; ++partial) {
      const auto begin{partial * initialization_elements};
      const auto end{xpu::min(begin + initialization_elements, g_count)};
      stencil::energy::initialize_reciprocal_partial(
        state, begin, end, build_structure, &scratch, state.reciprocal_partials + partial
      );
    }
    for (auto partial{0uz}; partial < real_count; ++partial) {
      const auto begin{partial * initialization_elements};
      const auto end{xpu::min(begin + initialization_elements, particle_count * particle_count)};
      stencil::energy::initialize_real_partial(state, begin, end, &scratch, state.real_partials + partial);
    }
    stencil::energy::finalize_initial_energy(state, finalize_reciprocal_count, real_count, &scratch);
  }
#endif
}

} // namespace

inline void initialize(
  EnergyTracker::InitializationBatchView states,
  std::size_t num_threads
) noexcept {
  initialize<InitializationMode::ALL>(
    states,
    states.energy.walker_count(),
    states.particles.count(),
    states.energy.num_g_vectors,
    num_threads
  );
}

inline void initialize_reciprocal_energy(EnergyTracker::InitializationView state) noexcept {
  const xpu::array<EnergyTracker::InitializationView, 1uz> states{state};
  initialize<InitializationMode::RECIPROCAL>(states, 1uz, state.particles.count, state.energy.num_g_vectors, 1uz);
}

inline void initialize_real_energy(EnergyTracker::InitializationView state) noexcept {
  const xpu::array<EnergyTracker::InitializationView, 1uz> states{state};
  initialize<InitializationMode::REAL>(states, 1uz, state.particles.count, state.energy.num_g_vectors, 1uz);
}

inline void initialize_structure_factors(EnergyTracker::InitializationView state) noexcept {
  const xpu::array<EnergyTracker::InitializationView, 1uz> states{state};
  initialize<InitializationMode::STRUCTURE>(states, 1uz, state.particles.count, state.energy.num_g_vectors, 1uz);
}

namespace {

#if defined(XPU_CUDA)
__global__
void cudaAcceptEnergyMove(EnergyTracker::View energy, fp_t real_delta, fp_t reciprocal, bool update_reciprocal) {
  stencil::energy::accept_move(energy, real_delta, reciprocal, update_reciprocal);
}
#endif

} // namespace

inline void accept_move(EnergyTracker::View energy, fp_t real_delta, fp_t reciprocal, bool update_reciprocal) noexcept {
#if defined(XPU_CUDA)
  constexpr dim3 acceptEnergyMoveThreads{1u};
  constexpr dim3 acceptEnergyMoveBlocks{1u};
  cudaAcceptEnergyMove<<<
    acceptEnergyMoveBlocks, acceptEnergyMoveThreads
  >>>(energy, real_delta, reciprocal, update_reciprocal);
  xpu::cu_check(cudaGetLastError());
#else
  stencil::energy::accept_move(energy, real_delta, reciprocal, update_reciprocal);
#endif
}

} // namespace kernel::energy
} // namespace kernel
