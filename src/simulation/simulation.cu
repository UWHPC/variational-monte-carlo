#include <xpu/xpu.hpp>
#include "simulation.hpp"
#include "simulation_kernels.hpp"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

Simulation::Simulation(
  Config config,
  std::unique_ptr<OutputWriter> output_writer,
  std::uint64_t walker_id
)
: config_{std::move(config)}
, particles_{config_.num_particles, config_.num_walkers}
, wave_function_{particles_, config_.box_length, config_.jastrow_a, config_.jastrow_b}
, blocking_analysis_{config_.block_size}
, energy_tracker_{config_.box_length, particles_}
, output_writer_{std::move(output_writer)}
, proposed_{}
, accepted_{}
, positions_{
    std::vector<fp_t>(particles_.count()),
    std::vector<fp_t>(particles_.count()),
    std::vector<fp_t>(particles_.count())
  }
, walker_rng_{config_.num_walkers}
, step_result_{config_.num_walkers}
, local_energies_{config_.num_walkers}
, sweep_result_{1uz}
, walker_states_{config_.num_walkers}
, run_result_{1uz}
{
  kernel::simulation::seed_generators(
    walker_rng_.data(),
    config_.num_walkers,
    config_.master_seed,
    walker_id * config_.num_walkers
  );
}

const std::array<std::vector<fp_t>, idx(Axis::NUM)>& Simulation::positions_snapshot() {
  const std::size_t N{particles_.count()};
  auto [p_x, p_y, p_z]{particles_.pos().pointers()};

  xpu::copy_n(positions_[idx(Axis::X)].data(), p_x, N);
  xpu::copy_n(positions_[idx(Axis::Y)].data(), p_y, N);
  xpu::copy_n(positions_[idx(Axis::Z)].data(), p_z, N);

  return positions_;
}

void Simulation::initialize_walkers() {
  constexpr auto max_attempts{100uz};

  kernel::simulation::initialize_positions(
    this->batch_view(),
    config_.box_length
  );

  wave_function_.initialize_matrices(
    particles_.batch_view(),
    config_.num_threads
  );

  for (auto walker{0uz}; walker < particles_.walker_count(); ++walker) {
    const auto walker_particles{particles_.view(walker)};
    auto log_psi{
      wave_function_.evaluate_log_psi_from_matrix(
        walker_particles,
        walker
      )
    };

    for (auto attempt{1uz}; !std::isfinite(log_psi) && attempt < max_attempts; ++attempt) {
      kernel::simulation::initialize_positions(
        this->view(walker),
        config_.box_length
      );

      log_psi = wave_function_.evaluate_log_psi(
        walker_particles,
        walker
      );
    }

    if (!std::isfinite(log_psi)) {
      throw std::runtime_error(
        "Failed to find non-singular initial configuration"
      );
    }
  }

  energy_tracker_.initialize(particles_, config_.num_threads);
}

Simulation::StepResult Simulation::metropolis_step() {
  const Simulation::StepResult result{
    kernel::simulation::metropolis_step(
      this->view(),
      config_.step_size
    )
  };

  return result;
}

Simulation::SweepResult Simulation::metropolis_sweep() {
  kernel::simulation::metropolis_sweep(
    this->batch_view(),
    particles_.count(),
    config_.step_size,
    sweep_result_.data()
  );

  SweepResult result{};
  xpu::copy_n(&result, sweep_result_.data(), 1uz);
  return result;
}

void Simulation::measure_walkers() {
  kernel::simulation::measure_walkers(this->batch_view());
}

void Simulation::warmup() {
  auto& step_size{config_.step_size};

  constexpr auto target_rate{0.50_fp};
  constexpr auto initial_gain{0.25_fp};

  const auto max_step_size{0.5_fp * config_.box_length};
  step_size = xpu::min(step_size, max_step_size);

  for (auto sweep{0uz}; sweep < config_.warmup_sweeps; ++sweep) {
    const auto result{metropolis_sweep()};
    const auto acceptance_rate{result.acceptance_rate()};
    const auto gain{initial_gain * xpu::rsqrt(scast<fp_t>(sweep + 1uz))};

    step_size *= xpu::exp(gain * (acceptance_rate - target_rate));
    step_size = xpu::min(step_size, max_step_size);
  }
}

Simulation::MeasurementSummary Simulation::measure() {
  auto& particles{particles_};
  auto& blocking_analysis{blocking_analysis_};
  proposed_ = 0U;
  accepted_ = 0U;

  fp_t running_energy_sum{};
  fp_t final_mean_energy{};
  std::size_t sample_count{};
  std::optional<fp_t> final_standard_error{};
  std::vector<fp_t> local_energies(particles.walker_count());

  for (auto sweep{0uz}; sweep < config_.measure_sweeps; ++sweep) {
    const auto result{metropolis_sweep()};
    proposed_ += result.proposed;
    accepted_ += result.accepted;

    measure_walkers();
    xpu::copy_n(
      local_energies.data(),
      local_energies_.data(),
      local_energies.size()
    );

    auto sweep_energy_sum{0.0_fp};
    for (auto walker{0uz}; walker < particles.walker_count(); ++walker) {
      sweep_energy_sum += local_energies[walker];
    }

    const auto sweep_energy{
      sweep_energy_sum / scast<fp_t>(particles.walker_count())
    };

    running_energy_sum += sweep_energy;
    ++sample_count;
    blocking_analysis.add(sweep_energy);

    const fp_t running_mean{
      running_energy_sum / scast<fp_t>(sample_count)
    };
    final_mean_energy = running_mean;

    std::optional<fp_t> frame_standard_error{};
    if (blocking_analysis.ready()) {
      const auto [blocked_mean, standard_error]{blocking_analysis.mean_and_standard_error()};
      final_mean_energy = blocked_mean;
      final_standard_error = standard_error;
      frame_standard_error = standard_error;
    }

    if (output_writer_) {
      const auto& snapshot{positions_snapshot()};
      const std::size_t particle_count{particles_.count()};

      std::vector<fp_t> flat_positions(particle_count * 3U);
      for (std::size_t particle{}; particle < particle_count; ++particle) {
        flat_positions[particle * 3U] = snapshot[0][particle];
        flat_positions[particle * 3U + 1U] = snapshot[1][particle];
        flat_positions[particle * 3U + 2U] = snapshot[2][particle];
      }

      output_writer_->write_frame(FrameData{
        .step = sweep + 1uz,
        .accepted = accepted_,
        .proposed = proposed_,
        .acceptance_rate = acceptance_rate(),
        .local_energy = sweep_energy,
        .mean_energy = running_mean,
        .standard_error = frame_standard_error,
        .positions = std::move(flat_positions)
      });
    }

    if (config_.is_master_thread) {
      if ((sweep & 127uz) == 0uz || sweep + 1uz == config_.measure_sweeps) {
        std::cout << "\rProgress: "
                  << ((sweep + 1uz) * 100uz / config_.measure_sweeps)
                  << "%" << std::flush;
      }
    }
  }
  if (config_.is_master_thread) {
    std::cout << "\r" << std::string(20, ' ') << "\r";
  }

  return MeasurementSummary{
    .mean_energy = final_mean_energy,
    .standard_error = final_standard_error,
    .acceptance_rate = acceptance_rate()
  };
}

Simulation::MeasurementSummary Simulation::run() {
  this->initialize_walkers();

  if (output_writer_) {
    output_writer_->write_init(InitData{
      .run_id = "vmc-seed-" + std::to_string(config_.master_seed),
      .num_particles = config_.num_particles,
      .box_length = config_.box_length,
      .warmup_steps = config_.warmup_steps,
      .measure_steps = config_.measure_steps,
      .step_size = config_.step_size,
      .seed = config_.master_seed,
      .block_size = config_.block_size
    });
  }

  if (!output_writer_) {
    const RunConfig run_config{
      .box_length = config_.box_length,
      .initial_step_size = config_.step_size,
      .warmup_sweeps = config_.warmup_sweeps,
      .measure_sweeps = config_.measure_sweeps,
      .proposals_per_sweep = particles_.count(),
      .block_size = config_.block_size,
      .num_threads = config_.num_threads
    };

    const auto result{
      kernel::simulation::run_walkers(
        this->batch_view(),
        walker_states_.data(),
        run_config,
        run_result_.data()
      )
    };

    std::optional<fp_t> standard_error{};
    if (result.has_standard_error) {
      standard_error = result.standard_error;
    }

    return {
      .mean_energy = result.mean_energy,
      .standard_error = standard_error,
      .acceptance_rate = result.acceptance_rate
    };
  }

  warmup();
  const MeasurementSummary summary{measure()};

  if (output_writer_) {
    output_writer_->write_done(DoneData{
      .total_accepted = accepted_,
      .total_proposed = proposed_,
      .final_acceptance_rate = acceptance_rate(),
      .final_mean_energy = summary.mean_energy,
      .final_standard_error = summary.standard_error
    });
  }

  return summary;
}
