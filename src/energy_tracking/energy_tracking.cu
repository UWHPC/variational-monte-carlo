#include <xpu/xpu.hpp>
#include "energy_tracking.cuh"

#include <numbers>

EnergyTracker::EnergyTracker(real_t box_length, real_t num_particles)
: box_length_{box_length}
, ewald_alpha_{6.0_r / box_length}
, ewald_correction_{-6.0_r * num_particles / (xpu::sqrt(std::numbers::pi_v<real_t>) * box_length)}
, ewald_background_{-std::numbers::pi_v<real_t> * num_particles * num_particles / (72.0_r * box_length)}
, V_recip_{}
, V_real_{}
, num_g_vectors_{}
, data_{} {
  const real_t two_pi_over_L{2.0_r * std::numbers::pi_v<real_t> / box_length};
  const real_t four_alpha_sq{4.0_r * ewald_alpha_ * ewald_alpha_};
  const real_t cutoff_factor{-xpu::log(EWALD_RECIPROCAL_TOLERANCE)};

  const real_t g_max_mag_sq{four_alpha_sq * cutoff_factor};
  const int m_max{static_cast<int>(xpu::ceil(xpu::sqrt(g_max_mag_sq) / two_pi_over_L)) + 1};

  std::vector<real_t> tmp_x, tmp_y, tmp_z, tmp_w;

  auto& g_x{tmp_x};
  auto& g_y{tmp_y};
  auto& g_z{tmp_z};
  auto& weights{tmp_w};

  // G = 2*pi / L
  for (int m_x = -m_max; m_x <= m_max; ++m_x) {
    for (int m_y = -m_max; m_y <= m_max; ++m_y) {
      for (int m_z = -m_max; m_z <= m_max; ++m_z) {
        // Only keeps "positive half-sphere"
        // Make up for this by 2x the weights
        if (m_x < 0)
          continue;
        if (m_x == 0 && m_y < 0)
          continue;
        if (m_x == 0 && m_y == 0 && m_z <= 0)
          continue;

        const real_t g_cand_x{two_pi_over_L * static_cast<real_t>(m_x)};
        const real_t g_cand_y{two_pi_over_L * static_cast<real_t>(m_y)};
        const real_t g_cand_z{two_pi_over_L * static_cast<real_t>(m_z)};
        const real_t g_cand_mag_sq{
          g_cand_x * g_cand_x +
          g_cand_y * g_cand_y +
          g_cand_z * g_cand_z
        };

        if (g_cand_mag_sq > g_max_mag_sq)
          continue;

        g_x.emplace_back(g_cand_x);
        g_y.emplace_back(g_cand_y);
        g_z.emplace_back(g_cand_z);
        weights.emplace_back(
          8.0_r * std::numbers::pi_v<real_t> * std::numbers::pi_v<real_t> / g_cand_mag_sq *
          xpu::exp(-g_cand_mag_sq / four_alpha_sq)
        );
      }
    }
  }

  num_g_vectors_ = g_x.size();

  data_ = AlignedSoA<real_t>(num_g_vectors_, NUM_ARRAYS);

  const auto g_dst{g_vector()};
  std::copy_n(tmp_x.data(), num_g_vectors_, g_dst.x_);
  std::copy_n(tmp_y.data(), num_g_vectors_, g_dst.y_);
  std::copy_n(tmp_z.data(), num_g_vectors_, g_dst.z_);
  std::copy_n(tmp_w.data(), num_g_vectors_, g_weights());
}

#ifdef VMC_CUDA_BACKEND
namespace {

__global__
void cudaInitializeReciprocalEnergy(
  const std::size_t num_G,
  const real_t* RESTRICT g_weights,
  const real_t* RESTRICT sum_real,
  const real_t* RESTRICT sum_imag,
  real_t* RESTRICT G_sum
) {
  const auto [g]{xpu::global_index<1>()};
  if (g >= num_G) { return; }
  atomicAdd(G_sum, g_weights[g] * (sum_real[g] * sum_real[g] + sum_imag[g] * sum_imag[g]));
}

} // namespace
#endif


void EnergyTracker::initialize_reciprocal_energy() noexcept {
#ifdef VMC_CUDA_BACKEND

  const real_t prefactor{1.0_r / (2.0_r * std::numbers::pi_v<real_t> * box_length_ * box_length_ * box_length_)};
  AlignedSoA<real_t> G_sum{1, 1};
  const auto num_G{num_g_vectors_};

  dim3 initalizeReciprocalEnergyThreads(256);
  dim3 initalizeReciprocalEnergyBlocks(vmc::cudaNumBlocks(num_G, initalizeReciprocalEnergyThreads.x));

 cudaInitializeReciprocalEnergy<<<initalizeReciprocalEnergyBlocks, initalizeReciprocalEnergyThreads>>>(
   num_G,
    this->g_weights(),
    this->sum_real(),
    this->sum_imag(),
    G_sum[0]
 );
 CUDA_CHECK(cudaGetLastError());
 CUDA_CHECK(cudaDeviceSynchronize());
 V_recip_ = prefactor * *G_sum[0];
 return;

  #else
  const real_t L{box_length_};
  const real_t prefactor{1.0_r / (2.0_r * std::numbers::pi_v<real_t> * L * L * L)};

  const std::size_t num_G{num_g_vectors_};
  const real_t* RESTRICT g_weights{this->g_weights()};
  const real_t* RESTRICT sum_real{this->sum_real()};
  const real_t* RESTRICT sum_imag{this->sum_imag()};

  ASSUME_ALIGNED(g_weights, SIMD_BYTES);
  ASSUME_ALIGNED(sum_real, SIMD_BYTES);
  ASSUME_ALIGNED(sum_imag, SIMD_BYTES);

  real_t sum{};
  #pragma omp simd reduction(+ : sum)
  for (std::size_t g = 0; g < num_G; ++g) {
    sum += g_weights[g] * (sum_real[g] * sum_real[g] + sum_imag[g] * sum_imag[g]);
  }
  V_recip_ = prefactor * sum;
  #endif
}

#ifdef VMC_CUDA_BACKEND
namespace {

__global__
void cudaInitializeRealEnergy(
  const std::size_t N,
  const real_t L,
  const real_t neg_L,
  const real_t half_L,
  const real_t neg_half_L,
  const real_t alpha,
  const real_t* RESTRICT pos_x,
  const real_t* RESTRICT pos_y,
  const real_t* RESTRICT pos_z,
  real_t* RESTRICT sum
) {
  const auto [i, j]{xpu::global_index<2>()};

  if (i >= N || j >= N) return;
  if (j <= i) return;

  real_t dx{pos_x[i] - pos_x[j]};
  real_t dy{pos_y[i] - pos_y[j]};
  real_t dz{pos_z[i] - pos_z[j]};

  dx += L * (dx <= neg_half_L) + neg_L * (dx > half_L);
  dy += L * (dy <= neg_half_L) + neg_L * (dy > half_L);
  dz += L * (dz <= neg_half_L) + neg_L * (dz > half_L);

  const real_t r{xpu::sqrt(dx * dx + dy * dy + dz * dz)};
  const real_t inv_r{(r < 1e-12_r) ? 1.0_r : 1.0_r / r};

  atomicAdd(sum, xpu::erfc(alpha * r) * inv_r);
}

} // namespace
#endif

void EnergyTracker::initialize_real_energy(const Particles& particles) noexcept {
#ifdef VMC_CUDA_BACKEND

  const std::size_t N{particles.size()};
  const real_t L{box_length_};
  const real_t neg_L{-1.0_r * L};
  const real_t half_L{0.5_r * L};
  const real_t neg_half_L{-1.0_r * half_L};
  const real_t alpha{ewald_alpha_};

  const auto pos{particles.pos()};
  AlignedSoA<real_t> Sum{1, 1};

  dim3 initializeRealEnergyThreads(16, 16);
  dim3 initializeRealEnergyBlocks(
    vmc::cudaNumBlocks(N, initializeRealEnergyThreads.x),
    vmc::cudaNumBlocks(N, initializeRealEnergyThreads.y)
  );

  cudaInitializeRealEnergy<<<initializeRealEnergyBlocks, initializeRealEnergyThreads>>>(
    N, L, neg_L, half_L, neg_half_L, alpha,
    pos.x_, pos.y_, pos.z_,
    Sum[0]
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  V_real_ = *Sum[0];
  return;
  #else
  const std::size_t N{particles.size()};
  const real_t L{box_length_};
  const real_t neg_L{-1.0_r * L};
  const real_t half_L{0.5_r * L};
  const real_t neg_half_L{-1.0_r * half_L};
  const real_t alpha{ewald_alpha_};

  const auto pos{particles.pos().align()};

  real_t sum{};
  for (std::size_t i = 0; i < N; ++i) {
    real_t local_sum{};

    // Not vectorzied: loop contains a mathematical function
    #pragma omp simd reduction(+ : local_sum)
    for (std::size_t j = i + 1; j < N; ++j) {
      real_t dx{pos.x_[i] - pos.x_[j]};
      real_t dy{pos.y_[i] - pos.y_[j]};
      real_t dz{pos.z_[i] - pos.z_[j]};

      dx += L * (dx <= neg_half_L) + neg_L * (dx > half_L);
      dy += L * (dy <= neg_half_L) + neg_L * (dy > half_L);
      dz += L * (dz <= neg_half_L) + neg_L * (dz > half_L);

      const real_t r{
        xpu::sqrt(
          dx * dx +
          dy * dy +
          dz * dz
        )
      };
      const real_t inv_r{(r < 1e-12_r) ? 1.0_r : 1.0_r / r};
      local_sum += xpu::erfc(alpha * r) * inv_r;
    }
    sum += local_sum;
  }
  V_real_ = sum;
  #endif
}

#ifdef VMC_CUDA_BACKEND
namespace {

__global__
void cudaInitializeStructureFactors(
  const real_t* RESTRICT g_x, const real_t* RESTRICT g_y, const real_t* RESTRICT g_z,
  const real_t* RESTRICT pos_x, const real_t* RESTRICT pos_y, const real_t* RESTRICT pos_z,
  real_t* RESTRICT sum_real, real_t* RESTRICT sum_imag,
  const std::size_t num_G, const std::size_t N
) {
  const auto [i, j]{xpu::global_index<2>()};
  if (i >= num_G || j >= N) return;
  const real_t G_dot_r{
    g_x[i] * pos_x[j] +
    g_y[i] * pos_y[j] +
    g_z[i] * pos_z[j]
  };
  real_t cos_temp{};
  real_t sin_temp{};
  vmc::sincos(G_dot_r, &sin_temp, &cos_temp);

  atomicAdd(&sum_real[i], cos_temp);
  atomicAdd(&sum_imag[i], sin_temp);


 }
} // namespace
#endif

void EnergyTracker::initialize_structure_factors(const Particles& particles) noexcept {
  #ifdef VMC_CUDA_BACKEND
  const std::size_t N{particles.size()};
  const std::size_t num_G{num_g_vectors_};

  const auto pos{particles.pos()};
  const auto gv{g_vector()};

 
  auto const total_bytes{num_G * sizeof(real_t)};
  CUDA_CHECK(cudaMemset(this->sum_real(), 0, total_bytes));
  CUDA_CHECK(cudaMemset(this->sum_imag(), 0, total_bytes));

  dim3 initializeStructureFactorsThreads(16, 16);
  dim3 initializeStructureFactorsBlocks(
    vmc::cudaNumBlocks(num_G, initializeStructureFactorsThreads.x),
    vmc::cudaNumBlocks(N,     initializeStructureFactorsThreads.y)
  );

  cudaInitializeStructureFactors<<<initializeStructureFactorsBlocks, initializeStructureFactorsThreads>>>(
    gv.x_, gv.y_, gv.z_,
    pos.x_, pos.y_, pos.z_,
    this->sum_real(), this->sum_imag(),
    num_G, N
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  #else
  const std::size_t N{particles.size()};
  const std::size_t num_G{num_g_vectors_};

  const auto pos{particles.pos().align()};
  const auto gv{g_vector().align()};

  real_t* RESTRICT sum_real{this->sum_real()};
  real_t* RESTRICT sum_imag{this->sum_imag()};

  ASSUME_ALIGNED(sum_real, SIMD_BYTES);
  ASSUME_ALIGNED(sum_imag, SIMD_BYTES);

  for (std::size_t g = 0; g < num_G; ++g) {
    real_t cos_sum{};
    real_t sin_sum{};

    // Not vectorzied: loop contains a mathematical function
    #pragma omp simd reduction(+ : cos_sum, sin_sum)
    for (std::size_t j = 0; j < N; ++j) {
      const real_t G_dot_r{
        gv.x_[g] * pos.x_[j] +
        gv.y_[g] * pos.y_[j] +
        gv.z_[g] * pos.z_[j]
      };
      real_t cos_temp{};
      real_t sin_temp{};

      vmc::sincos(G_dot_r, &sin_temp, &cos_temp);

      cos_sum += cos_temp;
      sin_sum += sin_temp;
    }

    sum_real[g] = cos_sum;
    sum_imag[g] = sin_sum;
  }
  #endif
}


#ifdef VMC_CUDA_BACKEND
namespace {

__global__
void cudaUpdateStructureFactors(
  const real_t* RESTRICT g_x, const real_t* RESTRICT g_y, const real_t* RESTRICT g_z,
  real_t* RESTRICT sum_real, real_t* RESTRICT sum_imag,
  const std::size_t num_G,
  const real_t old_x, const real_t old_y, const real_t old_z,
  const real_t new_x, const real_t new_y, const real_t new_z
) {
  const auto [g]{xpu::global_index<1>()};
  if (g >= num_G) return;

  const real_t old_dot{
    g_x[g] * old_x +
    g_y[g] * old_y +
    g_z[g] * old_z
  };
  const real_t new_dot{
    g_x[g] * new_x +
    g_y[g] * new_y +
    g_z[g] * new_z
  };

  real_t new_sin{}, new_cos{};
  real_t old_sin{}, old_cos{};

  vmc::sincos(new_dot, &new_sin, &new_cos);
  vmc::sincos(old_dot, &old_sin, &old_cos);

  sum_real[g] += new_cos - old_cos;
  sum_imag[g] += new_sin - old_sin;
}

} // namespace

#endif

void EnergyTracker::update_structure_factors(
  real_t old_x,
  real_t old_y,
  real_t old_z,
  real_t new_x,
  real_t new_y,
  real_t new_z
) noexcept {
#ifdef VMC_CUDA_BACKEND
  const std::size_t num_G{this->num_g_vectors_};
  const auto gv{this->g_vector()};

  dim3 updateStructureFactorsThreads(256);
  dim3 updateStructureFactorsBlocks(vmc::cudaNumBlocks(num_G, updateStructureFactorsThreads.x));

  cudaUpdateStructureFactors<<<updateStructureFactorsBlocks, updateStructureFactorsThreads>>>(
    gv.x_, gv.y_, gv.z_,
    this->sum_real(), this->sum_imag(),
    num_G,
    old_x, old_y, old_z,
    new_x, new_y, new_z
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  initialize_reciprocal_energy();
#else
  const real_t L{box_length_};

  const std::size_t num_G{num_g_vectors_};
  const real_t prefactor{1.0_r / (2.0_r * std::numbers::pi_v<real_t> * L * L * L)};

  const real_t* RESTRICT g_weights{this->g_weights()};
  const auto gv{g_vector().align()};

  real_t* RESTRICT sum_real{this->sum_real()};
  real_t* RESTRICT sum_imag{this->sum_imag()};
  real_t* RESTRICT d_imag_temp{this->d_imag_temp()};
  real_t* RESTRICT d_real_temp{this->d_real_temp()};

  ASSUME_ALIGNED(g_weights, SIMD_BYTES);

  ASSUME_ALIGNED(sum_real, SIMD_BYTES);
  ASSUME_ALIGNED(sum_imag, SIMD_BYTES);
  ASSUME_ALIGNED(d_imag_temp, SIMD_BYTES);
  ASSUME_ALIGNED(d_real_temp, SIMD_BYTES);

  real_t delta{};

  // Not vectorzied: loop contains a mathematical function
  #pragma omp simd
  for (std::size_t g = 0; g < num_G; ++g) {
    const real_t old_dot{
      gv.x_[g] * old_x +
      gv.y_[g] * old_y +
      gv.z_[g] * old_z
    };
    const real_t new_dot{
      gv.x_[g] * new_x +
      gv.y_[g] * new_y +
      gv.z_[g] * new_z
    };

    real_t new_sin{}, new_cos{};
    real_t old_sin{}, old_cos{};

    vmc::sincos(new_dot, &new_sin, &new_cos);
    vmc::sincos(old_dot, &old_sin, &old_cos);

    d_real_temp[g] = new_cos - old_cos;
    d_imag_temp[g] = new_sin - old_sin;
  }

  // Accumulate delta and update sum_real / sum_imag
  #pragma omp simd reduction(+ : delta)
  for (std::size_t g = 0; g < num_G; ++g) {
    const real_t dr{d_real_temp[g]};
    const real_t di{d_imag_temp[g]};

    delta += g_weights[g] * (2.0_r * (sum_real[g] * dr + sum_imag[g] * di) + dr * dr + di * di);

    sum_real[g] += dr;
    sum_imag[g] += di;
  }

  V_recip_ += prefactor * delta;
#endif
}


#ifdef VMC_CUDA_BACKEND
namespace {

__global__
void cudaKineticEnergy(
  const std::size_t N,
  const real_t* RESTRICT grad_x,
  const real_t* RESTRICT grad_y,
  const real_t* RESTRICT grad_z,
  const real_t* RESTRICT lap,
  real_t* RESTRICT T_sum
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= N) return;

  // ||Grad(logPsi)||^2 for particle i
  const real_t grad_sq{
    grad_x[i] * grad_x[i] +
    grad_y[i] * grad_y[i] +
    grad_z[i] * grad_z[i]
  };

  // Accumulate Lapl(logPsi) + ||Grad(logPsi)||^2
  atomicAdd(T_sum, lap[i] + grad_sq);
}

} // namespace
#endif


real_t EnergyTracker::kinetic_energy(const Particles& particles) const noexcept {
 #ifdef VMC_CUDA_BACKEND

  const std::size_t N{particles.size()};
  const auto grad{particles.grad_log_psi()};   
  const real_t* lap{particles.lap_log_psi()};

  AlignedSoA<real_t> T_sum{1, 1};           

  dim3 kineticEnergyThreads(256);
  dim3 kineticEnergyBlocks(vmc::cudaNumBlocks(N, kineticEnergyThreads.x));

  cudaKineticEnergy<<<kineticEnergyBlocks, kineticEnergyThreads>>>(
    N, grad.x_, grad.y_, grad.z_, lap, T_sum[0]
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return *T_sum[0]*-0.5_r ;
#else
 const auto grad{particles.grad_log_psi().align()};
  const real_t* RESTRICT lap{particles.lap_log_psi()};

  ASSUME_ALIGNED(lap, SIMD_BYTES);

  // Kinetic
  real_t T_sum{};
  const std::size_t N{particles.size()};

  #pragma omp simd reduction(+ : T_sum)
  for (std::size_t i = 0; i < N; ++i) {
    // Computes ||Grad(logPsi)||^2
    const real_t grad_sq{
      grad.x_[i] * grad.x_[i] +
      grad.y_[i] * grad.y_[i] +
      grad.z_[i] * grad.z_[i]
    };

    // Accumulate Lapl(LogPsi) + ||Grad(LogPsi)||^2
    T_sum += (lap[i] + grad_sq);
  }

  return -0.5_r * T_sum;
#endif
}


#ifdef VMC_CUDA_BACKEND

namespace {

__global__
void cudaUpdateRealEnergy(
  const std::size_t N, const std::size_t moved_idx,
  const real_t L, const real_t alpha,
  const real_t old_x, const real_t old_y, const real_t old_z,
  const real_t new_x, const real_t new_y, const real_t new_z,
  const real_t* RESTRICT pos_x, const real_t* RESTRICT pos_y, const real_t* RESTRICT pos_z,
  real_t* RESTRICT delta
) {
  const real_t neg_L{-1.0_r * L};
  const real_t half_L{0.5_r * L};
  const real_t neg_half_L{-1.0_r * half_L};

  const auto [j]{xpu::global_index<1>()};
  if (j >= N) return;

  // Branchless mask to safely skip the moved particle
  const real_t valid_mask{(j == moved_idx) ? 0.0_r : 1.0_r};

  // Old pair
  real_t dx_old{old_x - pos_x[j]};
  real_t dy_old{old_y - pos_y[j]};
  real_t dz_old{old_z - pos_z[j]};

  dx_old += L * (dx_old <= neg_half_L) + neg_L * (dx_old > half_L);
  dy_old += L * (dy_old <= neg_half_L) + neg_L * (dy_old > half_L);
  dz_old += L * (dz_old <= neg_half_L) + neg_L * (dz_old > half_L);

  const real_t r_old{
    xpu::sqrt(
      dx_old * dx_old +
      dy_old * dy_old +
      dz_old * dz_old
    )
  };
  // Protect against 1.0 / 0.0 generating NaN
  const real_t inv_r_old{(r_old < 1e-12_r) ? 1.0_r : 1.0_r / r_old};
  const real_t erfc_old{xpu::erfc(alpha * r_old) * inv_r_old};

  real_t dx_new{new_x - pos_x[j]};
  real_t dy_new{new_y - pos_y[j]};
  real_t dz_new{new_z - pos_z[j]};

  dx_new += L * (dx_new <= neg_half_L) + neg_L * (dx_new > half_L);
  dy_new += L * (dy_new <= neg_half_L) + neg_L * (dy_new > half_L);
  dz_new += L * (dz_new <= neg_half_L) + neg_L * (dz_new > half_L);

  const real_t r_new{
    xpu::sqrt(
      dx_new * dx_new +
      dy_new * dy_new +
      dz_new * dz_new
    )
  };

  // Protect against 1.0 / 0.0 generating NaN
  const real_t inv_r_new{(r_new < 1e-12_r) ? 1.0_r : 1.0_r / r_new};
  const real_t erfc_new{xpu::erfc(alpha * r_new) * inv_r_new};

  atomicAdd(delta, valid_mask * (erfc_new - erfc_old));
}

}
#endif

void EnergyTracker::update_real_energy(
  std::size_t moved_idx,
  real_t old_x,
  real_t old_y,
  real_t old_z,
  const Particles& particles
) noexcept {
#ifdef VMC_CUDA_BACKEND
  const std::size_t N{particles.size()};
  const real_t L{box_length_};
  const real_t alpha{ewald_alpha_};

  const auto pos{particles.pos()};

  AlignedSoA<real_t> delta{1, 1};

  const real_t new_x{pos.x_[moved_idx]};
  const real_t new_y{pos.y_[moved_idx]};
  const real_t new_z{pos.z_[moved_idx]};

  dim3 updateRealEnergyThreads(256);
  dim3 updateRealEnergyBlocks(vmc::cudaNumBlocks(N, updateRealEnergyThreads.x));

 cudaUpdateRealEnergy<<<updateRealEnergyBlocks, updateRealEnergyThreads>>>(
    N, moved_idx,
    L, alpha,
    old_x, old_y, old_z,
    new_x, new_y, new_z,
    pos.x_, pos.y_, pos.z_,
    delta[0]
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  V_real_ += *delta[0];
#else
  const std::size_t N{particles.size()};
  const real_t L{box_length_};
  const real_t neg_L{-1.0_r * L};
  const real_t half_L{0.5_r * L};
  const real_t neg_half_L{-1.0_r * half_L};
  const real_t alpha{ewald_alpha_};

  const auto pos{particles.pos().align()};

  const real_t new_x{pos.x_[moved_idx]};
  const real_t new_y{pos.y_[moved_idx]};
  const real_t new_z{pos.z_[moved_idx]};

  real_t delta{};

  // Not vectorzied: loop contains a mathematical function
  #pragma omp simd reduction(+ : delta)
  for (std::size_t j = 0; j < N; ++j) {
    // Branchless mask to safely skip the moved particle
    const real_t valid_mask{(j == moved_idx) ? 0.0_r : 1.0_r};

    // Old pair
    real_t dx_old{old_x - pos.x_[j]};
    real_t dy_old{old_y - pos.y_[j]};
    real_t dz_old{old_z - pos.z_[j]};

    dx_old += L * (dx_old <= neg_half_L) + neg_L * (dx_old > half_L);
    dy_old += L * (dy_old <= neg_half_L) + neg_L * (dy_old > half_L);
    dz_old += L * (dz_old <= neg_half_L) + neg_L * (dz_old > half_L);

    const real_t r_old{
      xpu::sqrt(
        dx_old * dx_old +
        dy_old * dy_old +
        dz_old * dz_old
      )
    };
    // Protect against 1.0 / 0.0 generating NaN
    const real_t inv_r_old{(r_old < 1e-12_r) ? 1.0_r : 1.0_r / r_old};
    const real_t erfc_old{xpu::erfc(alpha * r_old) * inv_r_old};

    // New pair
    real_t dx_new{new_x - pos.x_[j]};
    real_t dy_new{new_y - pos.y_[j]};
    real_t dz_new{new_z - pos.z_[j]};

    dx_new += L * (dx_new <= neg_half_L) + neg_L * (dx_new > half_L);
    dy_new += L * (dy_new <= neg_half_L) + neg_L * (dy_new > half_L);
    dz_new += L * (dz_new <= neg_half_L) + neg_L * (dz_new > half_L);

    const real_t r_new{
      xpu::sqrt(
        dx_new * dx_new +
        dy_new * dy_new +
        dz_new * dz_new
      )
    };

    // Protect against 1.0 / 0.0 generating NaN
    const real_t inv_r_new{(r_new < 1e-12_r) ? 1.0_r : 1.0_r / r_new};
    const real_t erfc_new{xpu::erfc(alpha * r_new) * inv_r_new};

    delta += valid_mask * (erfc_new - erfc_old);
  }

  V_real_ += delta;
#endif
}

real_t EnergyTracker::potential_energy() const noexcept {
  return V_real_ + V_recip_ + ewald_correction_ + ewald_background_;
}

real_t EnergyTracker::eval_total_energy(const Particles& particles) const noexcept {
  return kinetic_energy(particles) + potential_energy();
}
