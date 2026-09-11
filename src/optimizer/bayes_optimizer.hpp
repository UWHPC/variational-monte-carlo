#pragma once

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

};
