#include "optimizer/jastrow_optimizer.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
  Config cfg{};
  cfg.num_particles = 7;
  cfg.box_length = 20.0_r;
  cfg.num_threads = 2;
  cfg.master_seed = 12345;
  JastrowOptimizer::BayesBenchmarkSettings settings;
  settings.optimizer.initial_points = 2;
  settings.optimizer.candidate_points = 9;
  settings.optimizer.max_evaluations = 2;
  settings.optimizer.validation_evaluations = 2;
  settings.optimizer.max_pending = 2;
  settings.optimizer.energy_tolerance = 100.0;
  settings.warmup_sweeps = 8;
  settings.measure_sweeps = 32;
  settings.validation_measure_sweeps = 64;
  settings.block_sweeps = 4;
  auto invalid{settings};
  invalid.block_sweeps = 0;
  bool rejected{};
  try { (void)JastrowOptimizer::evaluate(cfg, invalid); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("invalid block size accepted");
  const auto result{JastrowOptimizer::evaluate(cfg, settings)};
  if (!result.validated || result.evaluations != 4 || result.failed_evaluations != 0 ||
      result.validation_evaluations != 2 || result.elapsed_seconds < 0.0 ||
      !std::isfinite(result.result.energy) || !std::isfinite(result.result.standard_error) ||
      result.result.standard_error < 0.0_r || result.result.optimal_b <= 0.0_r)
    throw std::runtime_error("Bayesian simulation benchmark failed");
  JastrowOptimizer::GridBenchmarkSettings grid;
  static_cast<JastrowOptimizer::BayesBenchmarkSettings&>(grid) = settings;
  grid.grid_points = 3;
  const auto grid_result{JastrowOptimizer::evaluate(cfg, grid)};
  if (grid_result.search_evaluations != 3 || grid_result.validation_evaluations != 2 ||
      !grid_result.validated || grid_result.peak_workers > 2 ||
      !std::isfinite(grid_result.result.standard_error) ||
      grid_result.search_seconds < 0.0 || grid_result.validation_seconds < 0.0)
    throw std::runtime_error("grid benchmark accounting failed");
  std::cout << "Bayesian simulation integration passed\n";
}
