#include "bayes_optimizer.hpp"

#include <xpu/math.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

extern "C" void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info,
                        std::size_t uplo_length);

namespace {

[[nodiscard]] double dot_product(const double* RESTRICT left, const double* RESTRICT right,
                                 std::size_t count) noexcept {
  double sum{};
#pragma omp simd reduction(+ : sum)
  for (std::size_t i = 0; i < count; ++i) {
    sum += left[i] * right[i];
  }
  return sum;
}

} // namespace

GaussianProcess::GaussianProcess() : GaussianProcess(Parameters{}) {}

GaussianProcess::GaussianProcess(Parameters parameters) : parameters_{parameters} {
  if (!std::isfinite(parameters.length_scale) || parameters.length_scale <= 0.0 ||
      !std::isfinite(parameters.signal_variance) || parameters.signal_variance <= 0.0 ||
      !std::isfinite(parameters.mean) || !std::isfinite(parameters.jitter) ||
      parameters.jitter <= 0.0) {
    throw std::invalid_argument("GaussianProcess: invalid kernel parameters");
  }
}

double GaussianProcess::kernel(double left, double right) const {
  const double r{xpu::abs(left - right) / parameters_.length_scale};
  if (r > 350.0)
    return 0.0; // avoid underflow
  const double t{xpu::sqrt(5.0) * r};
  return parameters_.signal_variance * ((1.0 + t + t * t / 3.0) * xpu::exp(-t));
}

void GaussianProcess::fit(std::span<const double> inputs, std::span<const double> targets,
                          std::span<const double> noise_variances) {
  const std::size_t n{inputs.size()};
  if (n == 0 || targets.size() != n || noise_variances.size() != n) {
    throw std::invalid_argument(
        "GaussianProcess: observation arrays must be nonempty and equal-sized");
  }
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      n > std::vector<double>{}.max_size() / n) {
    throw std::length_error("GaussianProcess: covariance matrix is too large");
  }

  std::vector<double> residual(n);
  double diagonal_scale{parameters_.signal_variance};
  for (std::size_t i = 0; i < n; ++i) {
    residual[i] = targets[i] - parameters_.mean;
    const double diagonal{parameters_.signal_variance + noise_variances[i]};
    if (!std::isfinite(inputs[i]) || !std::isfinite(residual[i]) ||
        !std::isfinite(noise_variances[i]) || noise_variances[i] < 0.0 ||
        !std::isfinite(diagonal)) {
      throw std::invalid_argument(
          "GaussianProcess: observations must be finite with nonnegative noise variances");
    }
    diagonal_scale = std::max(diagonal_scale, diagonal);
  }

  std::vector<double> covariance(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      covariance[i * n + j] = kernel(inputs[i], inputs[j]);
    }
    covariance[i * n + i] += noise_variances[i];
  }

  std::vector<double> lower(n * n);
  double jitter{std::max(parameters_.jitter * diagonal_scale, std::numeric_limits<double>::min())};
  bool factored{false};
  const int order{static_cast<int>(n)};
  // Row-major L is column-major U for LAPACK; no transpose or copy is needed.
  constexpr char triangle{'U'};
  for (int attempt = 0; attempt < 8 && !factored; ++attempt) {
    lower = covariance;
    for (std::size_t i = 0; i < n; ++i)
      lower[i * n + i] += jitter;
    int info{};
    dpotrf_(&triangle, &order, lower.data(), &order, &info, 1);
    if (info < 0) {
      throw std::runtime_error("GaussianProcess: invalid LAPACK Cholesky argument");
    }
    factored = info == 0;
    for (std::size_t i = 0; i < n && factored; ++i) {
      factored = std::isfinite(lower[i * n + i]) && lower[i * n + i] > 0.0;
    }
    jitter *= 10.0;
  }
  if (!factored)
    throw std::runtime_error("GaussianProcess: Cholesky factorization failed");

  std::vector<double> alpha{residual};
  double quadratic{};
  double log_diagonal{};
  for (std::size_t i = 0; i < n; ++i) {
    const double* row{lower.data() + i * n};
    alpha[i] = (alpha[i] - dot_product(row, alpha.data(), i)) / row[i];
    quadratic += alpha[i] * alpha[i];
    log_diagonal += xpu::log(lower[i * n + i]);
  }
  for (std::size_t i = n; i-- > 0;) {
    const double* RESTRICT row{lower.data() + i * n};
    double* RESTRICT values{alpha.data()};
    values[i] /= row[i];
    const double solved{values[i]};
// Update a contiguous row instead of traversing a strided column of L.
#pragma omp simd
    for (std::size_t j = 0; j < i; ++j) {
      values[j] -= row[j] * solved;
    }
  }
  const double likelihood{0.5 * quadratic + log_diagonal +
                          0.5 * static_cast<double>(n) * xpu::log(2.0 * std::numbers::pi)};
  if (!std::isfinite(likelihood) ||
      !std::all_of(alpha.begin(), alpha.end(), [](double value) { return std::isfinite(value); })) {
    throw std::runtime_error("GaussianProcess: nonfinite fitted coefficients");
  }

  std::vector<double> fitted_inputs(inputs.begin(), inputs.end());
  inputs_.swap(fitted_inputs);
  lower_.swap(lower);
  alpha_.swap(alpha);
  negative_log_likelihood_ = likelihood;
}

GaussianProcess::Prediction GaussianProcess::predict(double input, std::span<double> work) const {
  if (!std::isfinite(input))
    throw std::invalid_argument("GaussianProcess: query must be finite");
  const std::size_t n{inputs_.size()};
  double mean{parameters_.mean};
  double reduction{};
  for (std::size_t i = 0; i < n; ++i) {
    const double covariance{kernel(inputs_[i], input)};
    mean += covariance * alpha_[i];
    const double* row{lower_.data() + i * n};
    work[i] = (covariance - dot_product(row, work.data(), i)) / row[i];
    reduction += work[i] * work[i];
  }
  const double variance{parameters_.signal_variance - reduction};
  const double tolerance{64.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(n) *
                         parameters_.signal_variance};
  if (!std::isfinite(mean) || !std::isfinite(variance) || variance < -tolerance) {
    throw std::runtime_error("GaussianProcess: invalid posterior prediction");
  }
  return Prediction{.mean = mean, .variance = std::max(0.0, variance)};
}

GaussianProcess::Prediction GaussianProcess::predict(double input) const {
  if (inputs_.empty())
    throw std::logic_error("GaussianProcess: fit before predicting");
  std::vector<double> workspace(inputs_.size());
  return predict(input, workspace);
}

std::vector<GaussianProcess::Prediction>
GaussianProcess::predict(std::span<const double> inputs) const {
  if (inputs_.empty())
    throw std::logic_error("GaussianProcess: fit before predicting");
  std::vector<Prediction> predictions;
  predictions.reserve(inputs.size());
  std::vector<double> workspace(inputs_.size());
  for (double input : inputs)
    predictions.push_back(predict(input, workspace));
  return predictions;
}

double GaussianProcess::negative_log_marginal_likelihood() const {
  if (inputs_.empty())
    throw std::logic_error("GaussianProcess: fit before evaluating likelihood");
  return negative_log_likelihood_;
}

BayesOptimizer::BayesOptimizer(Settings settings) : settings_{settings}, model_{settings.kernel} {
  if (!std::isfinite(settings.lower_bound) || !std::isfinite(settings.upper_bound) ||
      settings.lower_bound >= settings.upper_bound ||
      !std::isfinite(settings.upper_bound - settings.lower_bound) ||
      (settings.logarithmic && settings.lower_bound <= 0.0) || settings.initial_points < 2 ||
      settings.candidate_points < settings.initial_points || settings.max_pending == 0 ||
      settings.max_exploration_pending == 0 ||
      settings.max_evaluations < settings.initial_points || settings.validation_evaluations == 0 ||
      !std::isfinite(settings.energy_tolerance) || settings.energy_tolerance <= 0.0 ||
      !std::isfinite(settings.exploration) || settings.exploration <= 0.0) {
    throw std::invalid_argument("BayesOptimizer: invalid settings");
  }
  const std::size_t n{settings.candidate_points};
  coordinates_.resize(n);
  parameters_.resize(n);
  observations_.resize(n);
  pending_counts_.resize(n);
  pending_.reserve(settings.max_pending);
  const std::size_t training_capacity{std::min(n, settings.max_evaluations)};
  training_x_.reserve(training_capacity);
  training_y_.reserve(training_capacity);
  training_noise_.reserve(training_capacity);
  proposal_scratch_.inputs.reserve(training_capacity);
  proposal_scratch_.targets.reserve(training_capacity);
  proposal_scratch_.noise.reserve(training_capacity);
  proposal_scratch_.standard_deviations.resize(n);
  const double log_lower{settings.logarithmic ? xpu::log(settings.lower_bound) : 0.0};
  const double log_upper{settings.logarithmic ? xpu::log(settings.upper_bound) : 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    const double x{static_cast<double>(i) / static_cast<double>(n - 1)};
    coordinates_[i] = x;
    parameters_[i] = settings.logarithmic
                         ? xpu::exp((1.0 - x) * log_lower + x * log_upper)
                         : settings.lower_bound + x * (settings.upper_bound - settings.lower_bound);
  }
  parameters_.front() = settings.lower_bound;
  parameters_.back() = settings.upper_bound;
  for (std::size_t i = 1; i < n; ++i) {
    if (parameters_[i] <= parameters_[i - 1])
      throw std::invalid_argument("BayesOptimizer: bounds too narrow for candidate resolution");
  }
}

void BayesOptimizer::Aggregate::add(Measurement result) {
  ++count;
  const double delta{result.energy - mean};
  mean += delta / static_cast<double>(count);
  m2 += delta * (result.energy - mean);
  noise_sum += result.standard_error * result.standard_error;
  if (!std::isfinite(mean) || !std::isfinite(m2) || !std::isfinite(noise_sum))
    throw std::invalid_argument("BayesOptimizer: measurement aggregation overflow");
}

double BayesOptimizer::Aggregate::variance() const {
  if (count == 0)
    throw std::logic_error("BayesOptimizer: no measurements");
  const double n{static_cast<double>(count)};
  const double within{noise_sum / n / n};
  return count > 1 ? std::max(within, m2 / (n - 1.0) / n) : within;
}

void BayesOptimizer::update_model() {
  if (!dirty_)
    return;
  training_x_.clear();
  training_y_.clear();
  training_noise_.clear();
  center_ = 0.0;
  std::size_t count{};
  for (const auto& observation : observations_) {
    if (observation.count)
      center_ += (observation.mean - center_) / static_cast<double>(++count);
  }
  if (count == 0) {
    predictions_.clear();
    dirty_ = false;
    return;
  }
  scale_ = settings_.energy_tolerance;
  for (const auto& observation : observations_) {
    if (observation.count) {
      scale_ = std::max(
          {scale_, xpu::abs(observation.mean - center_), xpu::sqrt(observation.variance())});
    }
  }
  typical_noise_ = 1e-12;
  for (std::size_t i = 0; i < observations_.size(); ++i) {
    const auto& observation{observations_[i]};
    if (!observation.count)
      continue;
    typical_noise_ = std::max(typical_noise_, observation.noise_sum /
        static_cast<double>(observation.count) / scale_ / scale_);
    training_x_.push_back(coordinates_[i]);
    training_y_.push_back((observation.mean - center_) / scale_);
    training_noise_.push_back(observation.variance() / scale_ / scale_);
  }
  model_.fit(training_x_, training_y_, training_noise_);
  predictions_ = model_.predict(coordinates_);
  incumbent_ = static_cast<std::size_t>(
      std::min_element(predictions_.begin(), predictions_.end(),
                       [](const auto& a, const auto& b) { return a.mean < b.mean; }) -
      predictions_.begin());
  dirty_ = false;
}

bool BayesOptimizer::converged() const {
  if (training_x_.size() < settings_.initial_points)
    return false;
  double optimistic{std::numeric_limits<double>::infinity()};
  for (std::size_t i = 0; i < predictions_.size(); ++i) {
    if (i != incumbent_)
      optimistic = std::min(optimistic, predictions_[i].mean -
          settings_.exploration * xpu::sqrt(predictions_[i].variance));
  }
  const auto& best{predictions_[incumbent_]};
  // Search only needs to rule out a materially better parameter. Independent
  // validation applies the requested precision to the final energy estimate.
  return scale_ * (best.mean - optimistic) <= settings_.energy_tolerance;
}

std::size_t BayesOptimizer::choose_candidate() {
  // wait for initial set of samples to complete before splitting
  bool initial_complete{true};
  for (std::size_t i = 0; i < settings_.initial_points; ++i) {
    const std::size_t index{i * (settings_.candidate_points - 1) / (settings_.initial_points - 1)};
    if (!observations_[index].count) {
      initial_complete = false;
      if (!is_pending(index))
        return index;
    }
  }
  if (!initial_complete && predictions_.empty())
    return parameters_.size();

  // The completed-data posterior is already fitted and cached.
  std::span<const GaussianProcess::Prediction> proposed{predictions_};
  std::vector<GaussianProcess::Prediction> pending_predictions;
  auto& scratch{proposal_scratch_};
  if (!pending_.empty()) {
    scratch.inputs.assign(training_x_.begin(), training_x_.end());
    scratch.targets.assign(training_y_.begin(), training_y_.end());
    scratch.noise.assign(training_noise_.begin(), training_noise_.end());
    for (const auto& job : pending_) {
      const auto index{static_cast<std::size_t>(
          std::lower_bound(parameters_.begin(), parameters_.end(), job.parameter) - parameters_.begin())};
      scratch.inputs.push_back(coordinates_[index]);
      scratch.targets.push_back(predictions_[index].mean);
      scratch.noise.push_back(typical_noise_);
    }
    GaussianProcess temporary{settings_.kernel};
    temporary.fit(scratch.inputs, scratch.targets, scratch.noise);
    pending_predictions = temporary.predict(coordinates_);
    proposed = pending_predictions;
  }
  // Reuse each square root in the confidence bounds and acquisition score.
  const auto* RESTRICT posterior{proposed.data()};
  double* RESTRICT standard_deviations{scratch.standard_deviations.data()};
  #pragma omp simd
  for (std::size_t i = 0; i < proposed.size(); ++i) {
    standard_deviations[i] = xpu::sqrt(posterior[i].variance);
  }
  double best_upper{std::numeric_limits<double>::infinity()};
  for (std::size_t i = 0; i < proposed.size(); ++i)
    if (observations_[i].count)
      best_upper = std::min(best_upper, proposed[i].mean + settings_.exploration *
                                                               standard_deviations[i]);

  std::size_t selected{parameters_.size()};
  double best_score{-std::numeric_limits<double>::infinity()};
  bool has_new_candidate{false};
  for (std::size_t i = 0; i < observations_.size(); ++i) {
    if (!observations_[i].count && !is_pending(i)) {
      has_new_candidate = true;
      break;
    }
  }
  const bool global_exploration{has_new_candidate && search_attempts_ % 5 == 0};
  for (std::size_t i = 0; i < proposed.size(); ++i) {
    if (is_pending(i))
      continue;
    const auto& prediction{proposed[i]};
    const double lower{prediction.mean - settings_.exploration * standard_deviations[i]};
    double score{};
    if (!observations_[i].count) {
      score = global_exploration ? standard_deviations[i]
                                 : predictions_[incumbent_].mean - lower;
    } else {
      if (global_exploration || lower > best_upper + settings_.energy_tolerance / scale_)
        continue;
      const double error{xpu::sqrt(observations_[i].variance())};
      if (error <= settings_.energy_tolerance)
        continue;
      const double n{static_cast<double>(observations_[i].count)};
      score = settings_.exploration * error / scale_ * (1.0 - xpu::sqrt(n / (n + 1.0)));
    }
    if (score > best_score) {
      best_score = score;
      selected = i;
    }
  }
  return selected;
}

std::size_t BayesOptimizer::choose_replication_candidate() const {
  std::size_t selected{parameters_.size()};
  double best{std::numeric_limits<double>::infinity()};
  for (std::size_t i = 0; i < observations_.size(); ++i) {
    if (observations_[i].count && predictions_[i].mean < best &&
        xpu::sqrt(observations_[i].variance()) > settings_.energy_tolerance) {
      best = predictions_[i].mean;
      selected = i;
    }
  }
  return selected;
}

BayesOptimizer::Status BayesOptimizer::status() {
  update_model();
  if (validating_) {
    if (validation_attempts_ < settings_.validation_evaluations || !pending_.empty())
      return Status::validating;
    return validation_.count && xpu::sqrt(validation_.variance()) <= settings_.energy_tolerance
               ? Status::complete
               : Status::budget_exhausted;
  }
  const bool stop{search_attempts_ >= settings_.max_evaluations || (!settings_.exact_budget && converged())};
  if (stop) {
    if (!pending_.empty())
      return Status::waiting;
    if (predictions_.empty())
      return Status::budget_exhausted;
    validating_ = true; // freeze until all validation attempts finish
    return Status::validating;
  }
  return pending_.size() >= settings_.max_pending ? Status::waiting : Status::searching;
}

std::vector<BayesOptimizer::Job> BayesOptimizer::ask(std::size_t available_slots) {
  std::vector<Job> jobs;
  const std::size_t slots{std::min(available_slots, settings_.max_pending - pending_.size())};
  std::size_t exploration_pending{static_cast<std::size_t>(std::count_if(
      pending_.begin(), pending_.end(), [](const Job& job) { return job.purpose == Purpose::exploration; }))};
  jobs.reserve(slots);
  for (std::size_t i = 0; i < slots; ++i) {
    const auto state{status()};
    if (state == Status::complete || state == Status::budget_exhausted || state == Status::waiting)
      break;
    std::size_t index{incumbent_};
    Purpose purpose{Purpose::validation};
    if (state == Status::validating) {
      if (validation_attempts_ >= settings_.validation_evaluations)
        break;
    } else {
      index = choose_candidate();
      if (index == parameters_.size() && settings_.exact_budget) {
        // Spend remaining work on the best available observed candidate.
        double best{std::numeric_limits<double>::infinity()};
        for (std::size_t j = 0; j < observations_.size(); ++j) {
          if (observations_[j].count && !is_pending(j) && predictions_[j].mean < best) {
            best = predictions_[j].mean;
            index = j;
          }
        }
      }
      if (index == parameters_.size()) {
        if (!pending_.empty())
          break;
        // grid has no eligible work left, validate instead of stall
        validating_ = true;
        index = incumbent_;
      } else {
        purpose = observations_[index].count ? Purpose::replication : Purpose::exploration;
        if (purpose == Purpose::exploration &&
            exploration_pending >= settings_.max_exploration_pending) {
          index = choose_replication_candidate();
          if (index == parameters_.size())
            break;
          purpose = Purpose::replication;
        }
      }
    }
    const Job job{next_id_, parameters_[index], purpose};
    pending_.push_back(job);
    ++pending_counts_[index];
    jobs.push_back(job);
    ++next_id_;
    if (purpose == Purpose::validation)
      ++validation_attempts_;
    else {
      ++search_attempts_;
      exploration_pending += purpose == Purpose::exploration;
    }
  }
  return jobs;
}

void BayesOptimizer::tell(std::uint64_t job_id, Measurement result) {
  const auto it{std::find_if(pending_.begin(), pending_.end(),
                             [job_id](const Job& job) { return job.id == job_id; })};
  if (it == pending_.end())
    throw std::invalid_argument("BayesOptimizer: unknown or completed job");
  if (!std::isfinite(result.energy) || !std::isfinite(result.standard_error) ||
      result.standard_error < 0.0 || !std::isfinite(result.standard_error * result.standard_error))
    throw std::invalid_argument("BayesOptimizer: invalid measurement");
  const auto index{static_cast<std::size_t>(
      std::lower_bound(parameters_.begin(), parameters_.end(), it->parameter) -
      parameters_.begin())};
  auto& destination{it->purpose == Purpose::validation ? validation_ : observations_[index]};
  auto updated{destination};
  updated.add(result);
  destination = updated;
  if (it->purpose != Purpose::validation)
    dirty_ = true;
  --pending_counts_[index];
  pending_.erase(it);
}

void BayesOptimizer::fail(std::uint64_t job_id) {
  const auto it{std::find_if(pending_.begin(), pending_.end(),
                             [job_id](const Job& job) { return job.id == job_id; })};
  if (it == pending_.end())
    throw std::invalid_argument("BayesOptimizer: unknown or completed job");
  const auto index{static_cast<std::size_t>(
      std::lower_bound(parameters_.begin(), parameters_.end(), it->parameter) - parameters_.begin())};
  --pending_counts_[index];
  pending_.erase(it);
}

BayesOptimizer::Recommendation BayesOptimizer::recommendation() {
  update_model();
  if (predictions_.empty())
    throw std::logic_error("BayesOptimizer: no successful observations");
  if (validating_ && validation_.count)
    return {parameters_[incumbent_], validation_.mean, xpu::sqrt(validation_.variance()), true};
  const auto& best{predictions_[incumbent_]};
  return {parameters_[incumbent_], center_ + scale_ * best.mean, scale_ * xpu::sqrt(best.variance),
          false};
}
