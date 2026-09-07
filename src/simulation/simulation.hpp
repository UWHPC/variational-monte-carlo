#pragma once

#include "../blocking_analysis/blocking_analysis.hpp"
#include "../config/config.hpp"
#include "../energy_tracking/energy_tracking.hpp"
#include "../output_writer/output_writer.hpp"
#include "../particles/particles.hpp"
#include "../wavefunction/wavefunction.hpp"
#include <xpu/buffer.hpp>
#include <xpu/random.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

class Simulation {
public:
  struct RandomProposal {
    std::size_t particle;
    xpu::array<fp_t, idx(Axis::NUM)> displacement;
    fp_t acceptance;
  };

  struct StepResult {
    bool accepted;
    std::size_t moved_particle;
    xpu::array<fp_t, idx(Axis::NUM)> old_pos;
    xpu::array<fp_t, idx(Axis::NUM)> new_pos;
    fp_t log_psi_delta;
    fp_t real_energy_delta;
    fp_t reciprocal_energy;
  };

  struct SweepResult {
    std::size_t proposed{};
    std::size_t accepted{};

    [[nodiscard]] CUDA_CALLABLE
    fp_t acceptance_rate() const noexcept {
      if (proposed == 0uz) {
        return 0.0_fp;
      }

      return scast<fp_t>(accepted) / scast<fp_t>(proposed);
    }
  };

  struct MetropolisScratch {
    RandomProposal proposal{};
    StepResult result{};
    SweepResult sweep_result{};
    fp_t slater_ratio{};
    fp_t jastrow_delta{};
    fp_t real_energy_delta{};
    fp_t reciprocal_sum{};
  };

  struct RunConfig {
    fp_t box_length{};
    fp_t initial_step_size{};
    std::size_t warmup_sweeps{};
    std::size_t measure_sweeps{};
    std::size_t proposals_per_sweep{};
    std::size_t block_size{};
    std::size_t num_threads{};
  };

  struct WalkerState {
    fp_t step_size{};
    fp_t energy_sum{};
    fp_t block_sum{};
    fp_t blocked_mean{};
    fp_t blocked_m2{};

    std::size_t proposed{};
    std::size_t accepted{};
    std::size_t sample_count{};
    std::size_t samples_in_block{};
    std::size_t block_count{};
  };

  struct RunResult {
    fp_t mean_energy{};
    fp_t standard_error{};
    fp_t acceptance_rate{};

    std::size_t proposed{};
    std::size_t accepted{};

    bool has_standard_error{};
  };

  struct View {
    Particles::View particles{};
    WaveFunction::View wave_function{};
    EnergyTracker::View energy_tracker{};
    xpu::random::generator* generator{};
    StepResult* step_result{};
    fp_t* local_energy{};
  };

  struct BatchView {
    Particles::BatchView particles{};
    WaveFunction::BatchView wave_function{};
    EnergyTracker::BatchView energy_tracker{};

    xpu::random::generator* generators{};
    StepResult* step_results{};
    fp_t* local_energies{};

    [[nodiscard]] CUDA_CALLABLE
    std::size_t walker_count() const noexcept {
      return particles.walker_count();
    }

    [[nodiscard]] CUDA_CALLABLE
    View view(std::size_t walker) noexcept {
      return {
        particles.view(walker),
        wave_function.view(walker),
        energy_tracker.view(walker),
        generators + walker,
        step_results + walker,
        local_energies + walker
      };
    }
  };

  struct MeasurementSummary {
    fp_t mean_energy;
    std::optional<fp_t> standard_error;
    fp_t acceptance_rate;
  };

private:
  Config config_;

  Particles particles_;
  WaveFunction wave_function_;
  BlockingAnalysis blocking_analysis_;
  EnergyTracker energy_tracker_;
  std::unique_ptr<OutputWriter> output_writer_;

  std::size_t proposed_;
  std::size_t accepted_;

  std::array<std::vector<fp_t>, idx(Axis::NUM)> positions_;

  [[nodiscard]] fp_t acceptance_rate() const {
    if (proposed_ == 0uz) {
      return 0.0_fp;
    } else {
      return scast<fp_t>(accepted_) / scast<fp_t>(proposed_);
    }
  }

  xpu::buffer<xpu::random::generator> walker_rng_;
  xpu::buffer<StepResult> step_result_;
  xpu::buffer<fp_t> local_energies_;
  xpu::buffer<View> walker_views_;
  xpu::buffer<SweepResult> sweep_result_;
  xpu::buffer<WalkerState> walker_states_;
  xpu::buffer<RunResult> run_result_;

  [[nodiscard]] const std::array<std::vector<fp_t>, idx(Axis::NUM)>& positions_snapshot();

public:
  explicit Simulation(
    Config cfg,
    std::unique_ptr<OutputWriter> output_writer = nullptr,
    std::uint64_t walker_id = 0
  );

  [[nodiscard]]
  BatchView batch_view() noexcept {
    return {
      particles_.batch_view(),
      wave_function_.batch_view(),
      energy_tracker_.batch_view(),
      walker_rng_.data(),
      step_result_.data(),
      local_energies_.data()
    };
  }

  [[nodiscard]]
  View view(std::size_t walker = 0uz) noexcept {
    return this->batch_view().view(walker);
  }

  MeasurementSummary run();
  StepResult metropolis_step();
  SweepResult metropolis_sweep();
  void measure_walkers();

private:
  void initialize_positions();
  void warmup();
  MeasurementSummary measure();
};
