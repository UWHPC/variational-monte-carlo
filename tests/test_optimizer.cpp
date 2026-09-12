#include "test_utilities.hpp"

#include "optimizer/jastrow_optimizer.hpp"
#include "simulation/simulation.cuh"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>

namespace {

real_t energy_at_b(std::size_t N, real_t r_s, real_t b, uint64_t seed) {
  const real_t L{box_length_from_rs(r_s, N)};
  Config config{make_config(N, L, 3000U, 40000U, L / 10.0_r, seed, 1000U)};
  config.jastrow_b = b;

  Simulation sim{config};
  const auto summary{sim.run()};
  return summary.mean_energy;
}

} // namespace

TEST_CASE("Simulation reports kinetic and potential energy components", "[simulation][energy]") {
  constexpr std::size_t N{7U};
  constexpr real_t R_S{5.0_r};
  const real_t L{box_length_from_rs(R_S, N)};
  Config config{make_config(N, L, 20U, 40U, L / 10.0_r, 1234U, 35U)};

  Simulation sim{config};
  const auto summary{sim.run()};

  REQUIRE(std::isfinite(summary.mean_kinetic_energy));
  REQUIRE(std::isfinite(summary.mean_potential_energy));
  require_near(summary.mean_energy,
               summary.mean_kinetic_energy + summary.mean_potential_energy,
               1e-9_r);
}

// The optimizer should find a b value, and it should be positive and finite
TEST_CASE("Optimizer: produces valid b parameter", "[optimizer]") {
  constexpr std::size_t N{7U};
  constexpr real_t R_S{5.0_r};
  const real_t L{box_length_from_rs(R_S, N)};

  Config config{};
  config.num_particles = N;
  config.box_length = L;
  config.master_seed = 12345U;
  config.warmup_sweeps = 100U;
  config.measure_sweeps = 100U;
  config.warmup_steps = N * config.warmup_sweeps;
  config.measure_steps = N * config.measure_sweeps;
  config.step_size = L / 10.0_r;
  config.block_size = 100U;

  const auto result{JastrowOptimizer::optimize(config)};

  INFO("optimal_b = " << result.optimal_b);
  INFO("energy    = " << result.energy);

  REQUIRE(std::isfinite(result.optimal_b));
  REQUIRE(result.optimal_b > 0.0_r);
  REQUIRE(std::isfinite(result.energy));
}

// The optimized b should give lower (or equal) energy than b=1.0
TEST_CASE("Optimizer: optimized b lowers energy vs default b=1", "[optimizer]") {
  constexpr std::size_t N{7U};
  constexpr real_t R_S{5.0_r};
  const real_t L{box_length_from_rs(R_S, N)};

  Config config{};
  config.num_particles = N;
  config.box_length = L;
  config.master_seed = 54321U;
  config.warmup_sweeps = 100U;
  config.measure_sweeps = 100U;
  config.warmup_steps = N * config.warmup_sweeps;
  config.measure_steps = N * config.measure_sweeps;
  config.step_size = L / 10.0_r;
  config.block_size = 100U;

  const auto result{JastrowOptimizer::optimize(config)};

  // Run a longer simulation at the optimized b and at the default b=1
  const real_t E_OPTIMIZED{energy_at_b(N, R_S, result.optimal_b, 9999U)};
  const real_t E_DEFAULT{energy_at_b(N, R_S, 1.0_r, 9999U)};

  INFO("E(b_opt=" << result.optimal_b << ") = " << E_OPTIMIZED);
  INFO("E(b=1.0) = " << E_DEFAULT);

  REQUIRE(E_OPTIMIZED <= E_DEFAULT);
}

// At r_s=10 the optimizer should also work
TEST_CASE("Optimizer: works at r_s=10", "[optimizer]") {
  constexpr std::size_t N{7U};
  constexpr real_t R_S{10.0_r};
  const real_t L{box_length_from_rs(R_S, N)};

  Config config{};
  config.num_particles = N;
  config.box_length = L;
  config.master_seed = 77777U;
  config.warmup_sweeps = 100U;
  config.measure_sweeps = 100U;
  config.warmup_steps = N * config.warmup_sweeps;
  config.measure_steps = N * config.measure_sweeps;
  config.step_size = L / 10.0_r;
  config.block_size = 100U;

  const auto result{JastrowOptimizer::optimize(config)};

  INFO("r_s=10, optimal_b = " << result.optimal_b);
  INFO("energy = " << result.energy);

  REQUIRE(std::isfinite(result.optimal_b));
  REQUIRE(result.optimal_b > 0.0_r);
  REQUIRE(result.optimal_b < 20.0_r);
}
