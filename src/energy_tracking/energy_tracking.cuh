#pragma once

#include "../particles/particles.cuh"
#include "../utilities/aligned_soa.cuh"
#include "../utilities/math.cuh"
#include "../utilities/ptr3d.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

class EnergyTracker {
private:
  real_t box_length_;

  static constexpr real_t EWALD_RECIPROCAL_TOLERANCE{1.0e-6_r};
  real_t ewald_alpha_;
  real_t ewald_correction_;
  real_t ewald_background_;

  real_t V_recip_;
  real_t V_real_;

  std::size_t num_g_vectors_;

  enum ArrayIndex : std::size_t {
    G_X,
    G_Y,
    G_Z,
    G_WEIGHTS,
    S_REAL,
    S_IMAG,
    D_REAL_TEMP,
    D_IMAG_TEMP,
    NUM_ARRAYS
  };
  AlignedSoA<real_t> data_;

public:
  explicit EnergyTracker(real_t box_length, real_t num_particles);

  void initialize_reciprocal_energy() noexcept;
  void initialize_real_energy(const Particles& particles) noexcept;

  void initialize_structure_factors(const Particles& particles) noexcept;

  void update_structure_factors(
    real_t old_x,
    real_t old_y,
    real_t old_z,
    real_t new_x,
    real_t new_y,
    real_t new_z
  ) noexcept;

  void update_real_energy(
    std::size_t moved_idx,
    real_t old_x,
    real_t old_y,
    real_t old_z,
    const Particles& particles
  ) noexcept;

  real_t eval_total_energy(const Particles& particles) const noexcept;
  real_t kinetic_energy(const Particles& particles) const noexcept;
  real_t potential_energy() const noexcept;

private:
  Ptr3D<      real_t> g_vector()       noexcept { return {data_[G_X], data_[G_Y], data_[G_Z]}; }
  Ptr3D<const real_t> g_vector() const noexcept { return {data_[G_X], data_[G_Y], data_[G_Z]}; }

  [[nodiscard]] real_t*       g_weights()       noexcept { return data_[G_WEIGHTS]; }
  [[nodiscard]] real_t const* g_weights() const noexcept { return data_[G_WEIGHTS]; }

  [[nodiscard]] real_t*       sum_real()       noexcept { return data_[S_REAL]; }
  [[nodiscard]] real_t const* sum_real() const noexcept { return data_[S_REAL]; }

  [[nodiscard]] real_t*       sum_imag()       noexcept { return data_[S_IMAG]; }
  [[nodiscard]] real_t const* sum_imag() const noexcept { return data_[S_IMAG]; }

  [[nodiscard]] real_t*       d_real_temp()       noexcept { return data_[D_REAL_TEMP]; }
  [[nodiscard]] real_t const* d_real_temp() const noexcept { return data_[D_REAL_TEMP]; }

  [[nodiscard]] real_t*       d_imag_temp()       noexcept { return data_[D_IMAG_TEMP]; }
  [[nodiscard]] real_t const* d_imag_temp() const noexcept { return data_[D_IMAG_TEMP]; }

};
