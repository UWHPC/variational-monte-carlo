#include "config/config.hpp"
#include "optimizer/jastrow_optimizer.hpp"
#include "output_writer/output_writer.hpp"
#include "simulation/simulation.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <print>
#include <string_view>

namespace {

void print_startup() {
  std::print(
    "\n<--- Variational Monte Carlo Simulation --->\n\n"
    "<--- Optimizing Jastrow b parameter --->\n"
  );
}

void print_config(const Config& config) {
  std::print(
    "\n<--- Config Settings --->\n"
    "Number of CPU threads: {}\n"
    "Number of walkers: {}\n"
    "Number of particles: {}\n"
    "Number of warmup sweeps: {}\n"
    "Number of measure sweeps: {}\n"
    "Length of box: {}\n"
    "Samples per block: {}\n"
    "Master seed: {}\n"
    "Jastrow a: {}\n"
    "Jastrow b: {} (optimized)\n\n",
    config.num_threads,
    config.num_walkers,
    config.num_particles,
    config.warmup_sweeps,
    config.measure_sweeps,
    config.box_length,
    config.block_size,
    config.master_seed,
    config.jastrow_a,
    config.jastrow_b
  );
}

void print_summary(
  const Simulation::MeasurementSummary& summary,
  const std::chrono::duration<double>& elapsed
) {
  std::print("<--- Final Measurements --->\nFinal Energy: {:.6}", summary.mean_energy);

  if (summary.standard_error.has_value()) {
    std::print(" +/- {:.6}", *summary.standard_error);
  } else {
    std::print(" +/- N/A (insufficient blocks)");
  }

  constexpr auto percent_scale{100.0_fp};
  std::print(
    "\nElapsed: {} s\nAcceptance Rate: {}%\n\n",
    elapsed.count(),
    summary.acceptance_rate * percent_scale
  );
}

void print_error(const std::string_view message) {
  std::print(stderr, "Exception: {}\n", message);
}

// Recording is opt-in: Simulation::run() takes a slower per-proposal path
// whenever an OutputWriter is present (needed to emit a frame per proposal),
// instead of the fast resident kernel. Set VMC_OUTPUT to a file path to
// trade throughput for a trajectory that render.py can play back.
std::unique_ptr<OutputWriter> make_recording_writer(std::ofstream& bin_out) {
  const char* const path{std::getenv("VMC_OUTPUT")};
  if (path == nullptr || *path == '\0') {
    return nullptr;
  }

  const std::filesystem::path output_path{path};
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  bin_out.open(output_path, std::ios::binary | std::ios::trunc);
  if (!bin_out) {
    std::print(stderr, "Warning: could not open {} for writing\n", path);
    return nullptr;
  }

  std::print("Recording trajectory to {}\n", path);
  return make_output_writer(OutputFormat::BIN, bin_out);
}

}

int main() {
  try {
    auto config{Config::from_file("config.cfg")};

    print_startup();

    constexpr auto verbose{true};
    const auto optimization{JastrowOptimizer::optimize(config, verbose)};
    config.jastrow_b = optimization.optimal_b;

    print_config(config);

    std::ofstream bin_out;
    auto output_writer{make_recording_writer(bin_out)};

    const auto start{std::chrono::steady_clock::now()};

    Simulation simulation{config, std::move(output_writer)};
    const auto summary{simulation.run()};

    const auto end{std::chrono::steady_clock::now()};
    const auto elapsed{std::chrono::duration<double>{end - start}};

    print_summary(summary, elapsed);

    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    print_error(exception.what());
    return EXIT_FAILURE;
  }
}
