#include "bayes_optimizer.hpp"

#include <xpu/math.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

extern "C" void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info,
                        std::size_t uplo_length);

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
  const double r{std::abs(left - right) / parameters_.length_scale};
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
    for (std::size_t j = 0; j < i; ++j)
      alpha[i] -= lower[i * n + j] * alpha[j];
    alpha[i] /= lower[i * n + i];
    quadratic += alpha[i] * alpha[i];
    log_diagonal += xpu::log(lower[i * n + i]);
  }
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t j = i + 1; j < n; ++j)
      alpha[i] -= lower[j * n + i] * alpha[j];
    alpha[i] /= lower[i * n + i];
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
    double value{covariance};
    for (std::size_t j = 0; j < i; ++j)
      value -= lower_[i * n + j] * work[j];
    work[i] = value / lower_[i * n + i];
    reduction += work[i] * work[i];
  }
  const double variance{parameters_.signal_variance - reduction};
  const double tolerance{64.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(n) *
                         parameters_.signal_variance};
  if (!std::isfinite(mean) || !std::isfinite(variance) || variance < -tolerance) {
    throw std::runtime_error("GaussianProcess: invalid posterior prediction");
  }
  return {mean, std::max(0.0, variance)};
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
