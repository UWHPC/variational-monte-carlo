#include "../simulation/simulation.cuh"
#include "jastrow_optimizer.hpp"
#include <cstdint>
#include <random>
#include <xpu/xpu.hpp>

#include <algorithm>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

real_t JastrowOptimizer::compute_rs(const Config& cfg) {
  const real_t N{static_cast<real_t>(cfg.num_particles)};
  const real_t volume{cfg.box_length * cfg.box_length * cfg.box_length};
  const real_t density{N / volume};
  return xpu::cbrt(3.0_r / (4.0_r * std::numbers::pi_v<real_t> * density));
}

JastrowOptimizer::Result JastrowOptimizer::optimize(const Config& base_config, bool verbose) {
  const real_t R_S{compute_rs(base_config)};
  const std::size_t N{base_config.num_particles};

  // b controls the inverse Jastrow range: effective range ~ 1/b.
  // b_min = 1/r_s: Jastrow range ≤ one mean interparticle spacing.
  //         Beyond this, correlations are long-range and should be handled
  //         by the Ewald sum, not the Jastrow factor.
  // b_max = 5.0: at b=5 the Jastrow range is 0.2 bohr, capturing only
  //         the electron-electron cusp with negligible correlation beyond.
  //         The energy is flat well before this point.
  const real_t b_min{1.0_r / R_S};
  const real_t b_max{std::max(5.0_r, b_min + 0.5_r)};
  const std::size_t grid_points{2U * base_config.num_threads};

  // Adapt scan statistics based on system size.
  const std::size_t scan_warmup{
      std::max<std::size_t>(50U, 500U / std::max<std::size_t>(N / 7U, 1U))};
  const std::size_t scan_measure{
      std::max<std::size_t>(50U, 500U / std::max<std::size_t>(N / 7U, 1U))};

  if (verbose) {
    std::cout << "[Optimizer] r_s=" << std::fixed << std::setprecision(2) << R_S << ", N=" << N
              << ", scanning b in [" << b_min << ", " << b_max << "]"
              << " (" << grid_points << " points, " << scan_warmup << "+" << scan_measure
              << " sweeps)\n";
  }

  // Build b values
  std::vector<real_t> b_values(grid_points);
  for (std::size_t i = 0; i < grid_points; ++i) {
    b_values[i] =
        b_min + (b_max - b_min) * static_cast<real_t>(i) / static_cast<real_t>(grid_points - 1);
  }

  // Phase 1: parallel grid scan
  std::vector<std::future<EvalResult>> futures;
  futures.reserve(grid_points);

  std::mt19937_64 seed_generator(base_config.master_seed);

  for (std::size_t i = 0; i < grid_points; ++i) {
    uint64_t seed = seed_generator();
    const real_t b{b_values[i]};
    const std::size_t warmup{scan_warmup};
    const std::size_t measure{scan_measure};
    futures.push_back(std::async(std::launch::async, [&base_config, b, warmup, measure, seed]() {
      return evaluate(base_config, b, warmup, 4096, 64, seed);
    }));
  }

  std::vector<EvalResult> results;
  results.reserve(grid_points);

  for (auto& f : futures) {
    results.push_back(f.get());
    if (verbose) {
      const auto& r{results.back()};
      std::cout << "  b=" << std::fixed << std::setprecision(3) << r.b
                << "  E=" << std::setprecision(6) << r.energy << "\n";
    }
  }

  // Find the best point, but guard against boundary artifacts.
  // If the lowest energy is at b_min and it's significantly lower than the
  // second-best, the long-range Jastrow bias is dominating. In that case,
  // skip the boundary point and take the best of the interior points.
  // "Significantly lower" = more than 2x the spread of the interior points.
  real_t best_b{1.0_r};
  real_t best_energy{std::numeric_limits<real_t>::max()};
  real_t second_best_energy{std::numeric_limits<real_t>::max()};

  for (const auto& r : results) {
    if (r.energy < best_energy) {
      second_best_energy = best_energy;
      best_energy = r.energy;
      best_b = r.b;
    } else if (r.energy < second_best_energy) {
      second_best_energy = r.energy;
    }
  }

  // Check if the winner is the boundary point and an outlier
  const bool at_boundary{xpu::abs(best_b - b_min) < 1e-10_r};
  if (at_boundary && results.size() > 2) {
    // Compute the energy spread of all non-boundary points
    real_t interior_min{std::numeric_limits<real_t>::max()};
    real_t interior_max{std::numeric_limits<real_t>::lowest()};
    for (const auto& r : results) {
      if (xpu::abs(r.b - b_min) < 1e-10_r)
        continue;
      interior_min = std::min(interior_min, r.energy);
      interior_max = std::max(interior_max, r.energy);
    }
    const real_t interior_spread{interior_max - interior_min};
    const real_t boundary_gap{interior_min - best_energy};

    // If the boundary point is more than 2x the interior spread below
    // the interior minimum, it's an artifact; use the interior best instead.
    if (boundary_gap > 2.0_r * interior_spread) {
      if (verbose) {
        std::cout << "[Optimizer] Boundary b=" << std::setprecision(3) << best_b
                  << " rejected (artifact), using interior best\n";
      }
      best_b = 1.0_r;
      best_energy = std::numeric_limits<real_t>::max();
      for (const auto& r : results) {
        if (xpu::abs(r.b - b_min) < 1e-10_r)
          continue;
        if (r.energy < best_energy) {
          best_energy = r.energy;
          best_b = r.b;
        }
      }
    }
  }

  if (verbose) {
    std::cout << "[Optimizer] Result: b=" << std::setprecision(4) << best_b << "\n";
  }

  return Result{.optimal_b = best_b, .energy = best_energy, .standard_error = 0.0_r};
}

JastrowOptimizer::EvalResult JastrowOptimizer::evaluate(const Config& base_config, real_t b,
                                                        std::size_t warmup_sweeps,
                                                        std::size_t measure_sweeps,
                                                        std::size_t block_sweeps,
                                                        std::uint64_t seed) {
  // if (block_sweeps == 0) {}
  Config cfg{};
  cfg.num_particles = base_config.num_particles;
  cfg.box_length = base_config.box_length;
  cfg.jastrow_a = base_config.jastrow_a;
  cfg.jastrow_b = b;
  cfg.master_seed = seed;
  cfg.is_master_thread = false;
  cfg.num_threads = 1;

  cfg.warmup_sweeps = warmup_sweeps;
  cfg.measure_sweeps = measure_sweeps;
  cfg.block_size = block_sweeps * base_config.num_particles;
  cfg.warmup_steps = cfg.num_particles * warmup_sweeps;
  cfg.measure_steps = cfg.num_particles * measure_sweeps;
  cfg.step_size = base_config.box_length / 10.0_r;

  Simulation sim{cfg};
  const auto summary{sim.run()};

  return EvalResult{
      .b = b,
      .energy = summary.mean_energy,
      .standard_error =
          summary.standard_error.has_value() && std::isfinite(summary.standard_error.value())
              ? summary.standard_error.value()
              : -1,
  };
}
