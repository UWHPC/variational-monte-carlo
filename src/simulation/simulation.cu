#include <xpu/xpu.hpp>
#include "simulation.cuh"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

Simulation::Simulation(Config config, std::unique_ptr<OutputWriter> output_writer)
: config_{std::move(config)}
, particles_{config_.num_particles}
, wave_function_{particles_, config_.box_length, config_.jastrow_a, config_.jastrow_b}
, blocking_analysis_{config_.block_size}
, energy_tracker_{config_.box_length, static_cast<real_t>(config_.num_particles)}
, output_writer_{std::move(output_writer)}
, proposed_{}
, accepted_{}
, log_psi_current_{}
, positions_{particles_.size(), 3}
, walker_rng_{}
{
  walker_rng_.init(config_, 0);
}

const AlignedSoA<real_t>& Simulation::positions_snapshot() {
  const std::size_t N{particles_.size()};
  auto [p_x, p_y, p_z]{particles_.pos().align()};

  for (std::size_t i{}; i < N; ++i) {
    positions_[0][i] = p_x[i];
    positions_[1][i] = p_y[i];
    positions_[2][i] = p_z[i];
  }
  return positions_;
}

void Simulation::initialize_positions() {
  const std::size_t N{particles_.size()};
  const real_t length{config_.box_length};
  auto [p_x, p_y, p_z]{particles_.pos().align()};

  constexpr std::size_t MAX_INIT_ATTEMPTS{100};

  for (std::size_t attempt = 0; attempt < MAX_INIT_ATTEMPTS; ++attempt) {
    for (std::size_t i = 0; i < N; i++) {
      p_x[i] = rand_uniform() * length;
      p_y[i] = rand_uniform() * length;
      p_z[i] = rand_uniform() * length;
    }

    log_psi_current_ = wave_function_.evaluate_log_psi(particles_);
    if (std::isfinite(log_psi_current_))
      break;

    if (attempt == MAX_INIT_ATTEMPTS - 1) {
      throw std::runtime_error("Failed to find non-singular initial configuration");
    }
  }

  energy_tracker_.initialize_structure_factors(particles_);
  energy_tracker_.initialize_reciprocal_energy();
  energy_tracker_.initialize_real_energy(particles_);
}

CUDA_CALLABLE
Simulation::StepResult Simulation::metropolis_step() {
  auto [p_x, p_y, p_z]{particles_.pos().align()};

  const std::size_t rand{rand_particle()};
  const real_t L{config_.box_length};
  const real_t inv_L{1.0_r / L};

  const real_t old_x{p_x[rand]};
  const real_t old_y{p_y[rand]};
  const real_t old_z{p_z[rand]};

  p_x[rand] += rand_proposal();
  p_y[rand] += rand_proposal();
  p_z[rand] += rand_proposal();

  // Branchless wrapping for [0, L)
  p_x[rand] -= L * xpu::floor(p_x[rand] * inv_L);
  p_y[rand] -= L * xpu::floor(p_y[rand] * inv_L);
  p_z[rand] -= L * xpu::floor(p_z[rand] * inv_L);

  auto& slater{wave_function_.slater_plane_wave()};

  slater.update_trig_cache(rand, particles_);

  const real_t* new_row{slater.build_row(rand)};
  const real_t slater_ratio{slater.determinant_ratio(rand, new_row)};

  const real_t delta_jastrow{
    wave_function_.jastrow_pade().delta_value(
      particles_,
      rand, old_x, old_y, old_z
    )
  };
  const real_t log_ratio_sq{2.0_r * xpu::log(xpu::abs(slater_ratio)) + 2.0_r * delta_jastrow};

  const real_t u{xpu::max(rand_uniform(), std::numeric_limits<real_t>::min())};
  const real_t log_u{xpu::log(u)};
  const real_t min_term{xpu::min(0.0_r, log_ratio_sq)};

  const bool accepted{log_u < min_term};

  if (accepted) {
    log_psi_current_ += xpu::log(xpu::abs(slater_ratio)) + delta_jastrow;
    slater.accept_move(rand, new_row, slater_ratio);

    energy_tracker_.update_structure_factors(
      old_x, old_y, old_z,
      p_x[rand], p_y[rand], p_z[rand]
    );
    energy_tracker_.update_real_energy(rand, old_x, old_y, old_z, particles_);

    return StepResult{true, rand, old_x, old_y, old_z};
  }

  slater.restore_trig_row(rand);
  p_x[rand] = old_x;
  p_y[rand] = old_y;
  p_z[rand] = old_z;

  return StepResult{false, rand, old_x, old_y, old_z};
}

void Simulation::warmup() {
  real_t& step_size{config_.step_size};

  const std::size_t warmup_steps{config_.warmup_steps};
  const std::size_t warmup_batch_size{particles_.size()};

  std::size_t window_proposed{};
  std::size_t window_accepted{};

  real_t acceptance_rate_window{};
  const real_t acceptance_target{0.50_r};
  const real_t gain{0.25_r};

  for (std::size_t i{}; i < warmup_steps; i++) {
    window_proposed++;
    const bool accepted{metropolis_step().accepted};
    if (accepted)
      ++window_accepted;

    if (window_proposed % warmup_batch_size == 0) {
      acceptance_rate_window =
          static_cast<real_t>(window_accepted) / static_cast<real_t>(window_proposed);
      step_size *= xpu::exp(gain * (acceptance_rate_window - acceptance_target));

      const real_t MAX_STEP{config_.box_length * 0.5_r};
      if (step_size > MAX_STEP) {
        step_size = MAX_STEP;
      }

      walker_rng_.change_step_size(step_size);

      window_accepted = 0;
      window_proposed = 0;
    }
  }
}

Simulation::MeasurementSummary Simulation::measure() {
  const std::size_t measure_steps{config_.measure_steps};

  auto& wavefunction{wave_function_};
  auto& particles{particles_};
  auto& blocking_analysis{blocking_analysis_};
  auto& energy_tracker{energy_tracker_};
  proposed_ = 0U;
  accepted_ = 0U;

  real_t running_energy_sum{};
  real_t running_kinetic_sum{};
  real_t running_potential_sum{};
  real_t final_mean_energy{};
  std::optional<real_t> final_standard_error{};

  for (std::size_t i = 0; i < measure_steps; ++i) {
    ++proposed_;
    const StepResult result{metropolis_step()};
    if (result.accepted) {
      ++accepted_;
    }

    wavefunction.evaluate_derivatives(
      particles, result.accepted, result.moved_particle,
      result.old_x, result.old_y, result.old_z
    );

    const real_t kinetic{energy_tracker.kinetic_energy(particles)};
    const real_t potential{energy_tracker.potential_energy()};
    const real_t E_local{kinetic + potential};
    running_energy_sum += E_local;
    running_kinetic_sum += kinetic;
    running_potential_sum += potential;
    blocking_analysis.add(E_local);

    const real_t running_mean{running_energy_sum / static_cast<real_t>(i + 1U)};
    final_mean_energy = running_mean;

    std::optional<real_t> frame_standard_error{};
    if (blocking_analysis.ready()) {
      const auto [blocked_mean, standard_error]{blocking_analysis.mean_and_standard_error()};
      final_mean_energy = blocked_mean;
      final_standard_error = standard_error;
      frame_standard_error = standard_error;
    }

    if (output_writer_) {
      const auto& snapshot{positions_snapshot()};
      const std::size_t N{particles_.size()};

      std::vector<real_t> flat_positions(N * 3U);
      for (std::size_t p{}; p < N; ++p) {
        flat_positions[p * 3U] = snapshot[0][p];
        flat_positions[p * 3U + 1U] = snapshot[1][p];
        flat_positions[p * 3U + 2U] = snapshot[2][p];
      }

      output_writer_->write_frame(FrameData{
        .step = i + 1U,
        .accepted = accepted_,
        .proposed = proposed_,
        .acceptance_rate = acceptance_rate(),
        .local_energy = E_local,
        .mean_energy = running_mean,
        .standard_error = frame_standard_error,
        .positions = std::move(flat_positions)
      });
    }

    if (config_.is_master_thread) {
      if ((i & 127) == 0 || i == measure_steps) {
        std::cout << "\rProgress: " << (i * 100 / measure_steps) << "%" << std::flush;
      }
    }
  }
  if (config_.is_master_thread) {
    std::cout << "\r" << std::string(20, ' ') << "\r";
  }

  return MeasurementSummary{
    .mean_energy = final_mean_energy,
    .mean_kinetic_energy = running_kinetic_sum / static_cast<real_t>(measure_steps),
    .mean_potential_energy = running_potential_sum / static_cast<real_t>(measure_steps),
    .standard_error = final_standard_error,
    .acceptance_rate = acceptance_rate()
  };
}

Simulation::MeasurementSummary Simulation::run() {
  initialize_positions();

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
