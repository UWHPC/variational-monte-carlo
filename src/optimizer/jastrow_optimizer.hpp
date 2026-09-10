#pragma once

#include "../config/config.hpp"

class JastrowOptimizer {
public:
  struct Result {
    real_t optimal_b;
    real_t energy;
    real_t standard_error;
  };

  // Optimize the Jastrow b parameter for the given config.
  // Phase 1: parallel grid scan to locate the basin.
  // Phase 2: serial golden-section refinement to pin down the minimum.
  // Uses variance-penalized energy to avoid unphysical long-range Jastrow states.
  [[nodiscard]] static Result optimize(const Config& base_config, bool verbose = false);

private:
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
