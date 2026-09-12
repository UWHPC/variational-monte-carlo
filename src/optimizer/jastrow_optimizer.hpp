#pragma once

#include "../config/config.hpp"
#include "bayes_optimizer.hpp"

class JastrowOptimizer {
public:
  struct Result {
    real_t optimal_b;
    real_t energy;
    real_t standard_error;
  };

  // Compatibility wrapper: bounded grid search with independent validation.
  [[nodiscard]] static Result optimize(const Config& base_config, bool verbose = false);

  struct BayesBenchmarkSettings {
    BayesOptimizer::Settings optimizer{};
    std::size_t warmup_sweeps{}; // Zero uses the grid-search warmup rule.
    std::size_t measure_sweeps{4096};
    std::size_t block_sweeps{64};
    std::size_t validation_measure_sweeps{16384};
  };

  struct GridBenchmarkSettings : BayesBenchmarkSettings {
    std::size_t grid_points{16};
    bool logarithmic{false};
  };

  struct BayesBenchmarkResult {
    Result result;
    BayesOptimizer::Status status;
    bool validated;
    std::size_t evaluations;
    std::size_t failed_evaluations;
    std::size_t validation_evaluations;
    double elapsed_seconds;
    std::size_t search_evaluations;
    std::size_t peak_workers;
    double search_seconds;
    double validation_seconds;
  };

  // Uses grid-search bounds and at most min(num_threads, max_pending) workers.
  // Counts include failed attempts and independent validation simulations.
  // Throws if no valid energy estimate was obtained.
  [[nodiscard]] static BayesBenchmarkResult evaluate(
      const Config& base_config, const BayesBenchmarkSettings& settings, bool verbose = false);
  [[nodiscard]] static BayesBenchmarkResult evaluate(
      const Config& base_config, const GridBenchmarkSettings& settings, bool verbose = false);

private:
  [[nodiscard]] static BayesBenchmarkResult benchmark(
      const Config& base_config, const BayesBenchmarkSettings& settings, bool grid, bool verbose);
  struct EvalResult {
    real_t b;
    real_t energy;
    real_t standard_error;
  };

  [[nodiscard]] static EvalResult evaluate(
    const Config& base_config,
    real_t b,
    std::size_t warmup_sweeps,
    std::size_t measure_sweeps,
    std::size_t block_sweeps,
    std::uint64_t seed
  );
  [[nodiscard]] static real_t compute_rs(const Config& cfg);
};
