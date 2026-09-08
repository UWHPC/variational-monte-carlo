#include <xpu/xpu.hpp>
#include "energy_tracking.hpp"
#include "energy_tracking_kernels.hpp"

#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

EnergyTracker::EnergyTracker(
  fp_t box_length,
  std::size_t num_particles,
  std::size_t num_walkers
)
: box_length_{box_length}
, ewald_alpha_{6.0_fp / box_length}
, ewald_correction_{-6.0_fp * scast<fp_t>(num_particles)/(xpu::sqrt(std::numbers::pi_v<fp_t>) * box_length)}
, ewald_background_{-std::numbers::pi_v<fp_t>*scast<fp_t>(num_particles)*scast<fp_t>(num_particles)/(72.0_fp * box_length)}
, num_g_vectors_{}
, num_walkers_{num_walkers}
, num_particles_{num_particles}
, shared_data_{0uz}
, walker_data_{num_walkers, 0uz}
, walker_scalars_{num_walkers}
, reduction_scratch_{num_walkers}
, sum_scratch_{
    kernel::energy::kinetic_energy_scratch_bytes(num_particles)
  }
, reciprocal_partials_{0uz}
, real_partials_{0uz}
, reciprocal_partial_count_{}
, real_partial_count_{}
{
  const fp_t two_pi_over_L{2.0_fp * std::numbers::pi_v<fp_t> / box_length};
  const fp_t four_alpha_sq{4.0_fp * ewald_alpha_ * ewald_alpha_};
  const fp_t cutoff_factor{-xpu::log(EWALD_RECIPROCAL_TOLERANCE)};

  const fp_t g_max_mag_sq{four_alpha_sq * cutoff_factor};
  const int m_max{scast<int>(xpu::ceil(xpu::sqrt(g_max_mag_sq) / two_pi_over_L)) + 1};

  std::vector<fp_t> g_x{};
  std::vector<fp_t> g_y{};
  std::vector<fp_t> g_z{};
  std::vector<fp_t> weights{};

  for (int m_x = -m_max; m_x <= m_max; ++m_x) {
    for (int m_y = -m_max; m_y <= m_max; ++m_y) {
      for (int m_z = -m_max; m_z <= m_max; ++m_z) {
        if (m_x < 0) { continue; }
        if (m_x == 0 && m_y < 0) { continue; }
        if (m_x == 0 && m_y == 0 && m_z <= 0) { continue; }

        const fp_t g_cand_x{two_pi_over_L * scast<fp_t>(m_x)};
        const fp_t g_cand_y{two_pi_over_L * scast<fp_t>(m_y)};
        const fp_t g_cand_z{two_pi_over_L * scast<fp_t>(m_z)};
        const fp_t g_cand_mag_sq{
          g_cand_x * g_cand_x +
          g_cand_y * g_cand_y +
          g_cand_z * g_cand_z
        };

        if (g_cand_mag_sq > g_max_mag_sq) { continue; }

        g_x.emplace_back(g_cand_x);
        g_y.emplace_back(g_cand_y);
        g_z.emplace_back(g_cand_z);
        weights.emplace_back(
          8.0_fp *
          std::numbers::pi_v<fp_t> * std::numbers::pi_v<fp_t> /
          g_cand_mag_sq *
          xpu::exp(-g_cand_mag_sq / four_alpha_sq)
        );
      }
    }
  }

  num_g_vectors_ = g_x.size();

  shared_data_ = xpu::soa<fp_t, idx(SharedArray::NUM_ARRAYS)>{
    this->num_g_vectors()
  };
  walker_data_ = xpu::soa_batch<fp_t, idx(WalkerArray::NUM_ARRAYS)>{
    this->walker_count(),
    this->num_g_vectors()
  };

  initialize_reduction_storage();

  auto g_vector{this->g_vector()};

  xpu::copy_n(
    g_vector[idx(Axis::X)],
    g_x.data(),
    this->num_g_vectors()
  );
  xpu::copy_n(
    g_vector[idx(Axis::Y)],
    g_y.data(),
    this->num_g_vectors()
  );
  xpu::copy_n(
    g_vector[idx(Axis::Z)],
    g_z.data(),
    this->num_g_vectors()
  );
  xpu::copy_n(
    this->g_weights(),
    weights.data(),
    this->num_g_vectors()
  );
}

void EnergyTracker::initialize_reduction_storage() {
  if (num_particles_ != 0uz && num_particles_ > std::numeric_limits<std::size_t>::max() / num_particles_) {
    throw std::length_error("Energy initialization pair count is too large");
  }
  reciprocal_partial_count_ = kernel::energy::initialization_partial_count(num_g_vectors_);
  real_partial_count_ = kernel::energy::initialization_partial_count(num_particles_ * num_particles_);
  const auto maximum_partials{std::numeric_limits<std::size_t>::max() / sizeof(fp_t)};
  if (num_walkers_ != 0uz && xpu::max(reciprocal_partial_count_, real_partial_count_) > maximum_partials / num_walkers_) {
    throw std::length_error("Energy initialization storage is too large");
  }
  kernel::energy::validate_initialization_size(
    num_walkers_, xpu::max(reciprocal_partial_count_, real_partial_count_)
  );
  reciprocal_partials_ = xpu::buffer<fp_t>{num_walkers_ * reciprocal_partial_count_};
  real_partials_ = xpu::buffer<fp_t>{num_walkers_ * real_partial_count_};
}

EnergyTracker::InitializationView EnergyTracker::initialization_view(
  Particles::View particles,
  std::size_t walker
) noexcept {
  return {
    this->view(walker),
    particles,
    reciprocal_partials_.data() + walker * reciprocal_partial_count_,
    real_partials_.data() + walker * real_partial_count_
  };
}

void EnergyTracker::validate_initialization(const Particles& particles, std::size_t num_threads) const {
  if (particles.count() != num_particles_ || particles.walker_count() != num_walkers_) {
    throw std::invalid_argument("Energy initialization requires matching particle and walker counts");
  }
  if (num_threads == 0uz || num_threads > scast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Energy initialization requires a valid CPU thread count");
  }
}

void EnergyTracker::initialize(Particles& particles, std::size_t num_threads) {
  this->validate_initialization(particles, num_threads);

  kernel::energy::initialize(
    this->initialization_batch_view(particles.batch_view()),
    num_threads
  );
}

void EnergyTracker::initialize_reciprocal_energy(std::size_t walker) noexcept {
  kernel::energy::initialize_reciprocal_energy(initialization_view({}, walker));
}

void EnergyTracker::initialize_real_energy(
  Particles::View particles,
  std::size_t walker
) noexcept {
  kernel::energy::initialize_real_energy(initialization_view(particles, walker));
}

void EnergyTracker::initialize_structure_factors(
  Particles::View particles,
  std::size_t walker
) noexcept {
  kernel::energy::initialize_structure_factors(initialization_view(particles, walker));
}

void EnergyTracker::update_structure_factors(
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  xpu::array<fp_t, idx(Axis::NUM)> new_pos,
  std::size_t walker
) noexcept {
  kernel::energy::update_structure_factors(this->view(walker), old_pos, new_pos);

  this->initialize_reciprocal_energy(walker);
}
fp_t EnergyTracker::kinetic_energy(
  Particles::View particles,
  std::size_t walker
) noexcept {
  return kernel::energy::kinetic_energy(
    this->view(walker),
    particles,
    sum_scratch_.data(),
    sum_scratch_.count()
  );
}

void EnergyTracker::update_real_energy(
  std::size_t moved,
  xpu::array<fp_t, idx(Axis::NUM)> old_pos,
  Particles::View particles,
  std::size_t walker
) noexcept {
  const auto energy{this->view(walker)};
  const auto delta{kernel::energy::update_real_energy(energy, moved, old_pos, particles)};
  kernel::energy::accept_move(energy, delta, 0.0_fp, false);
}

void EnergyTracker::accept_move(
  fp_t real_energy_delta,
  fp_t reciprocal_energy,
  std::size_t walker
) noexcept {
  kernel::energy::accept_move(this->view(walker), real_energy_delta, reciprocal_energy, true);
}

fp_t EnergyTracker::potential_energy(std::size_t walker) const noexcept {
  auto real{0.0_fp};
  auto reciprocal{0.0_fp};
  xpu::copy_n(&real, walker_scalars_[idx(WalkerScalar::V_REAL)] + walker, 1uz);
  xpu::copy_n(&reciprocal, walker_scalars_[idx(WalkerScalar::V_RECIP)] + walker, 1uz);
  return real + reciprocal + ewald_correction_ + ewald_background_;
}
