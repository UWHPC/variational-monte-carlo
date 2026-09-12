#include "../simulation/simulation.cuh"
#include "jastrow_optimizer.hpp"
#include <cstdint>
#include <random>
#include <xpu/xpu.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <atomic>
#include <variant>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

real_t JastrowOptimizer::compute_rs(const Config& cfg) {
  const real_t N{static_cast<real_t>(cfg.num_particles)};
  const real_t volume{cfg.box_length * cfg.box_length * cfg.box_length};
  const real_t density{N / volume};
  return xpu::cbrt(3.0_r / (4.0_r * std::numbers::pi_v<real_t> * density));
}

JastrowOptimizer::Result JastrowOptimizer::optimize(const Config& base_config, bool verbose) {
  GridBenchmarkSettings settings;
  if (base_config.num_threads > std::numeric_limits<std::size_t>::max() / 2)
    throw std::invalid_argument("Grid search: thread count overflow");
  settings.grid_points = 2 * base_config.num_threads;
  settings.optimizer.max_pending = base_config.num_threads;
  return evaluate(base_config, settings, verbose).result;
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

namespace {

// Grid scheduling shares the Bayesian executor and independent validation protocol.
class GridSearch {
public:
  using Job = BayesOptimizer::Job;
  using Purpose = BayesOptimizer::Purpose;
  using Status = BayesOptimizer::Status;
  using Measurement = BayesOptimizer::Measurement;
  using Recommendation = BayesOptimizer::Recommendation;

  explicit GridSearch(BayesOptimizer::Settings settings) : settings_{settings} {
    if (settings.max_evaluations < 2 || settings.max_pending == 0 || settings.validation_evaluations == 0 ||
        !std::isfinite(settings.lower_bound) || !std::isfinite(settings.upper_bound) ||
        settings.lower_bound <= 0.0 || settings.upper_bound <= settings.lower_bound ||
        !std::isfinite(settings.energy_tolerance) || settings.energy_tolerance <= 0.0)
      throw std::invalid_argument("Grid search: invalid settings");
    pending_.reserve(settings.max_pending);
  }

  std::span<const Job> pending() const { return pending_; }
  Status status() {
    if (!validating_) {
      if (search_ < settings_.max_evaluations) return Status::searching;
      if (!pending_.empty()) return Status::waiting;
      if (!std::isfinite(best_energy_)) return Status::budget_exhausted;
      validating_ = true;
    }
    if (validation_ < settings_.validation_evaluations || !pending_.empty()) return Status::validating;
    return count_ && error() <= settings_.energy_tolerance ? Status::complete : Status::budget_exhausted;
  }
  std::vector<Job> ask(std::size_t slots) {
    slots = std::min(slots, settings_.max_pending - pending_.size());
    std::vector<Job> jobs;
    jobs.reserve(slots);
    while (jobs.size() < slots) {
      const auto state{status()};
      if (state == Status::waiting || state == Status::complete || state == Status::budget_exhausted) break;
      double b{best_b_};
      Purpose purpose{Purpose::validation};
      if (state == Status::searching) {
        const double x{static_cast<double>(search_) / static_cast<double>(settings_.max_evaluations - 1)};
        b = settings_.logarithmic
            ? xpu::exp((1.0 - x) * xpu::log(settings_.lower_bound) + x * xpu::log(settings_.upper_bound))
            : settings_.lower_bound + x * (settings_.upper_bound - settings_.lower_bound);
        if (search_ == 0) b = settings_.lower_bound;
        if (search_ + 1 == settings_.max_evaluations) b = settings_.upper_bound;
        ++search_;
        purpose = Purpose::exploration;
      } else {
        if (validation_ >= settings_.validation_evaluations) break;
        ++validation_;
      }
      const Job job{.id = next_id_++, .parameter = b, .purpose = purpose};
      pending_.push_back(job);
      jobs.push_back(job);
    }
    return jobs;
  }
  void tell(std::uint64_t id, Measurement value) {
    const auto it{find(id)};
    if (it->purpose == Purpose::validation) {
      ++count_;
      const double delta{value.energy - mean_};
      mean_ += delta / static_cast<double>(count_);
      m2_ += delta * (value.energy - mean_);
      noise_ += value.standard_error * value.standard_error;
    } else if (value.energy < best_energy_ || (value.energy == best_energy_ && it->parameter < best_b_)) {
      best_energy_ = value.energy;
      best_error_ = value.standard_error;
      best_b_ = it->parameter;
    }
    pending_.erase(it);
  }
  void fail(std::uint64_t id) { pending_.erase(find(id)); }
  Recommendation recommendation() {
    if (!std::isfinite(best_energy_)) throw std::runtime_error("Grid search: no successful observations");
    return {best_b_, count_ ? mean_ : best_energy_, count_ ? error() : best_error_, count_ != 0};
  }
private:
  std::vector<Job>::iterator find(std::uint64_t id) {
    const auto it{std::find_if(pending_.begin(), pending_.end(), [id](const Job& j) { return j.id == id; })};
    if (it == pending_.end()) throw std::logic_error("Grid search: unknown job");
    return it;
  }
  double error() const {
    const double n{static_cast<double>(count_)};
    return xpu::sqrt(count_ > 1 ? std::max(noise_ / n / n, m2_ / (n - 1.0) / n) : noise_);
  }
  BayesOptimizer::Settings settings_;
  std::vector<Job> pending_;
  std::size_t search_{}, validation_{}, count_{};
  std::uint64_t next_id_{1};
  bool validating_{};
  double best_b_{}, best_energy_{std::numeric_limits<double>::infinity()}, best_error_{};
  double mean_{}, m2_{}, noise_{};
};

class SearchPolicy {
  using Policy = std::variant<BayesOptimizer, GridSearch>;
  Policy policy_;
public:
  SearchPolicy(bool grid, BayesOptimizer::Settings settings)
      : policy_{grid ? Policy{std::in_place_type<GridSearch>, settings}
                     : Policy{std::in_place_type<BayesOptimizer>, settings}} {}
  auto status() { return std::visit([](auto& p) { return p.status(); }, policy_); }
  auto pending() { return std::visit([](auto& p) { return p.pending(); }, policy_); }
  auto ask(std::size_t slots) { return std::visit([slots](auto& p) { return p.ask(slots); }, policy_); }
  void tell(std::uint64_t id, BayesOptimizer::Measurement value) {
    std::visit([&](auto& p) { p.tell(id, value); }, policy_);
  }
  void fail(std::uint64_t id) { std::visit([id](auto& p) { p.fail(id); }, policy_); }
  auto recommendation() { return std::visit([](auto& p) { return p.recommendation(); }, policy_); }
};

} // namespace

JastrowOptimizer::BayesBenchmarkResult JastrowOptimizer::evaluate(
    const Config& cfg, const BayesBenchmarkSettings& settings, bool verbose) {
  return benchmark(cfg, settings, false, verbose);
}

JastrowOptimizer::BayesBenchmarkResult JastrowOptimizer::evaluate(
    const Config& cfg, const GridBenchmarkSettings& settings, bool verbose) {
  auto common{static_cast<const BayesBenchmarkSettings&>(settings)};
  common.optimizer.max_evaluations = settings.grid_points;
  common.optimizer.logarithmic = settings.logarithmic;
  return benchmark(cfg, common, true, verbose);
}

JastrowOptimizer::BayesBenchmarkResult JastrowOptimizer::benchmark(
    const Config& base_config, const BayesBenchmarkSettings& settings, bool grid, bool verbose) {
  const auto started{std::chrono::steady_clock::now()};
  const std::size_t N{base_config.num_particles};
  const auto max_size{std::numeric_limits<std::size_t>::max()};
  const std::size_t warmup{settings.warmup_sweeps == 0
      ? std::max<std::size_t>(50U, 500U / std::max<std::size_t>(N / 7U, 1U))
      : settings.warmup_sweeps};
  if (N == 0 || base_config.num_threads == 0 ||
      !std::isfinite(base_config.box_length) || base_config.box_length <= 0.0_r ||
      !std::isfinite(base_config.jastrow_a) || settings.block_sweeps == 0 ||
      settings.measure_sweeps / settings.block_sweeps < 2 ||
      settings.validation_measure_sweeps / settings.block_sweeps < 2) {
    throw std::invalid_argument("Bayesian benchmark: invalid configuration or fewer than two measurement blocks");
  }
  if (warmup > max_size / N || settings.measure_sweeps > max_size / N ||
      settings.validation_measure_sweeps > max_size / N || settings.block_sweeps > max_size / N) {
    throw std::invalid_argument("Bayesian benchmark: sweep count overflow");
  }
  auto options{settings.optimizer};
  options.lower_bound = static_cast<double>(1.0_r / compute_rs(base_config));
  options.upper_bound = static_cast<double>(std::max(5.0_r, static_cast<real_t>(options.lower_bound) + 0.5_r));
  options.max_pending = std::min(options.max_pending, base_config.num_threads);
  SearchPolicy optimizer{grid, options};

  struct Task {
    BayesOptimizer::Job job;
    std::uint64_t seed;
  };
  struct Completion {
    BayesOptimizer::Job job;
    EvalResult result{};
    std::exception_ptr error;
  };
  std::mutex mutex;
  std::condition_variable_any available;
  std::condition_variable completed;
  std::deque<Task> tasks;
  std::deque<Completion> results;
  std::atomic<std::size_t> active{}, peak{};
  std::vector<std::jthread> workers;
  workers.reserve(options.max_pending);
  // Stop and wake every worker before any queues or captured state are destroyed.
  struct StopWorkers {
    std::vector<std::jthread>& workers;
    std::condition_variable_any& available;
    ~StopWorkers() {
      for (auto& worker : workers) worker.request_stop();
      available.notify_all();
      for (auto& worker : workers) if (worker.joinable()) worker.join();
    }
  } stop_workers{workers, available};

  for (std::size_t i = 0; i < options.max_pending; ++i) {
    workers.emplace_back([&](std::stop_token stop) {
      while (true) {
        Task task;
        {
          std::unique_lock lock{mutex};
          if (!available.wait(lock, stop, [&] { return !tasks.empty(); })) return;
          if (stop.stop_requested()) return;
          task = tasks.front();
          tasks.pop_front();
        }
        const auto running{active.fetch_add(1) + 1};
        auto previous{peak.load()};
        while (previous < running && !peak.compare_exchange_weak(previous, running)) {}
        Completion completion{.job = task.job, .result = {}, .error = {}};
        try {
          const auto sweeps{task.job.purpose == BayesOptimizer::Purpose::validation
              ? settings.validation_measure_sweeps : settings.measure_sweeps};
          completion.result = evaluate(base_config, static_cast<real_t>(task.job.parameter),
              warmup, sweeps, settings.block_sweeps, task.seed);
        } catch (...) {
          completion.error = std::current_exception();
        }
        {
          std::lock_guard lock{mutex};
          results.push_back(completion);
        }
        active.fetch_sub(1);
        completed.notify_one();
      }
    });
  }

  std::mt19937_64 seeds{base_config.master_seed};
  // Separate deterministic seed sequence makes validation independent of search length.
  std::mt19937_64 validation_seeds{base_config.master_seed ^ UINT64_C(0x9e3779b97f4a7c15)};
  auto validation_started{started};
  bool has_validation{false};
  std::size_t evaluations{}, failures{}, validation_evaluations{};
  while (true) {
    const auto state{optimizer.status()};
    if (state == BayesOptimizer::Status::complete || state == BayesOptimizer::Status::budget_exhausted) break;
    const auto jobs{optimizer.ask(options.max_pending - optimizer.pending().size())};
    {
      std::lock_guard lock{mutex};
      for (const auto& job : jobs) {
        const bool validation{job.purpose == BayesOptimizer::Purpose::validation};
        if (validation && !has_validation) {
          validation_started = std::chrono::steady_clock::now();
          has_validation = true;
        }
        tasks.push_back(Task{.job = job, .seed = validation ? validation_seeds() : seeds()});
        ++evaluations;
        if (job.purpose == BayesOptimizer::Purpose::validation) ++validation_evaluations;
      }
    }
    available.notify_all();
    if (optimizer.pending().empty()) {
      throw std::runtime_error("Bayesian benchmark: optimizer stalled without pending work");
    }
    Completion completion;
    {
      std::unique_lock lock{mutex};
      completed.wait(lock, [&] { return !results.empty(); });
      completion = results.front();
      results.pop_front();
    }
    const auto& result{completion.result};
    if (completion.error || !std::isfinite(result.energy) ||
        !std::isfinite(result.standard_error) || result.standard_error < 0.0_r) {
      optimizer.fail(completion.job.id);
      ++failures;
      if (verbose) std::cout << "[Bayesian benchmark] evaluation " << completion.job.id << " failed\n";
    } else {
      optimizer.tell(completion.job.id, {.energy = static_cast<double>(result.energy),
                                        .standard_error = static_cast<double>(result.standard_error)});
      if (verbose) {
        std::cout << "[Bayesian benchmark] b=" << result.b << " E=" << result.energy
                  << " standard error=" << result.standard_error << '\n';
      }
    }
  }
  const auto recommendation{optimizer.recommendation()};
  const auto final_status{optimizer.status()};
  for (auto& worker : workers) worker.request_stop();
  available.notify_all();
  for (auto& worker : workers) if (worker.joinable()) worker.join();
  const auto finished{std::chrono::steady_clock::now()};
  return BayesBenchmarkResult{
      .result = {.optimal_b = static_cast<real_t>(recommendation.parameter),
                 .energy = static_cast<real_t>(recommendation.energy),
                 .standard_error = static_cast<real_t>(recommendation.standard_error)},
      .status = final_status,
      .validated = recommendation.validated,
      .evaluations = evaluations,
      .failed_evaluations = failures,
      .validation_evaluations = validation_evaluations,
      .elapsed_seconds = std::chrono::duration<double>(finished - started).count(),
      .search_evaluations = evaluations - validation_evaluations,
      .peak_workers = peak.load(),
      .search_seconds = std::chrono::duration<double>((has_validation ? validation_started : finished) - started).count(),
      .validation_seconds = has_validation ? std::chrono::duration<double>(finished - validation_started).count() : 0.0};
}
