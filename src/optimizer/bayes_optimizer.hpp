#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

class GaussianProcess {
public:
  // Host LAPACK uses double precision independently of the simulation's real_t.
  struct Parameters {
    double length_scale{0.2};
    double signal_variance{1.0};
    double mean{0.0};
    double jitter{1e-10};
  };

  struct Prediction {
    double mean;
    double variance;
  };

  GaussianProcess();
  explicit GaussianProcess(Parameters parameters);

  void fit(std::span<const double> inputs, std::span<const double> targets,
           std::span<const double> noise_variances);
  [[nodiscard]] Prediction predict(double input) const;
  [[nodiscard]] std::vector<Prediction> predict(std::span<const double> inputs) const;
  [[nodiscard]] double negative_log_marginal_likelihood() const;

private:
  Parameters parameters_;
  std::vector<double> inputs_;
  std::vector<double> lower_; // row-major cholesky factor
  std::vector<double> alpha_;
  double negative_log_likelihood_{};

  [[nodiscard]] double kernel(double left, double right) const;
  [[nodiscard]] Prediction predict(double input, std::span<double> workspace) const;
};

class BayesOptimizer {
public:
  struct Settings {
    double lower_bound{0.1};
    double upper_bound{5.0};
    bool logarithmic{true};
    std::size_t initial_points{5};
    std::size_t candidate_points{257};
    std::size_t max_pending{4};
    std::size_t max_exploration_pending{static_cast<std::size_t>(-1)};
    std::size_t max_evaluations{64}; // Search attempts, including failures.
    bool exact_budget{false}; // Disable early stopping; replicate when the grid is exhausted.
    std::size_t validation_evaluations{4}; // Separate fixed-parameter attempts.
    double energy_tolerance{1e-3};
    double exploration{2.0};
    GaussianProcess::Parameters kernel{};
  };

  enum class Purpose { exploration, replication, validation };
  enum class Status { searching, waiting, validating, complete, budget_exhausted };
  struct Job {
    std::uint64_t id;
    double parameter;
    Purpose purpose;
  };
  struct Measurement {
    double energy;
    double standard_error; // Error of an independent, equally weighted chain mean.
  };
  struct Recommendation {
    double parameter;
    double energy;
    double standard_error;
    bool validated;
  };

  explicit BayesOptimizer(Settings settings);
  // Coordinator-owned: calls must be serialized. Each job is an independent,
  // equilibrated chain; the executor owns seeds, simulation state and threads.
  // Empty ask results can mean initial coverage is still pending, not termination.
  [[nodiscard]] std::vector<Job> ask(std::size_t available_slots);
  void tell(std::uint64_t job_id, Measurement result);
  void fail(std::uint64_t job_id);
  [[nodiscard]] std::span<const Job> pending() const noexcept { return pending_; }
  [[nodiscard]] Recommendation recommendation();
  // Convergence is a fixed-kernel, finite-grid heuristic, not a global guarantee.
  // budget_exhausted also covers validation that cannot meet energy_tolerance.
  [[nodiscard]] Status status();

private:
  struct Aggregate {
    std::size_t count{};
    double mean{};
    double m2{};
    double noise_sum{};
    void add(Measurement result);
    [[nodiscard]] double variance() const;
  };
  struct ProposalScratch {
    std::vector<double> inputs;
    std::vector<double> targets;
    std::vector<double> noise;
    std::vector<double> standard_deviations;
  };
  Settings settings_;
  GaussianProcess model_;
  std::vector<double> coordinates_;
  std::vector<double> parameters_;
  std::vector<Aggregate> observations_;
  std::vector<Job> pending_;
  std::vector<std::size_t> pending_counts_;
  std::vector<GaussianProcess::Prediction> predictions_;
  std::vector<double> training_x_, training_y_, training_noise_;
  ProposalScratch proposal_scratch_;
  Aggregate validation_;
  std::size_t search_attempts_{};
  std::size_t validation_attempts_{};
  std::size_t incumbent_{};
  double center_{};
  double scale_{1.0};
  double typical_noise_{1e-12};
  bool dirty_{true};
  bool validating_{};
  std::uint64_t next_id_{1};

  void update_model();
  [[nodiscard]] bool converged() const;
  [[nodiscard]] std::size_t choose_replication_candidate() const;
  [[nodiscard]] bool is_pending(std::size_t index) const noexcept { return pending_counts_[index] != 0; }
  [[nodiscard]] std::size_t choose_candidate();
};
