#include "optimizer/bayes_optimizer.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <source_location>
#include <string>

namespace {
void check(bool condition, std::source_location location = std::source_location::current()) {
  if (!condition) throw std::runtime_error("BayesOptimizer check failed at line " + std::to_string(location.line()));
}
}

int main() {
  using BO = BayesOptimizer;
  BO::Settings settings;
  settings.lower_bound = 0.1;
  settings.upper_bound = 1.0;
  settings.logarithmic = false;
  settings.initial_points = 5;
  settings.max_pending = 4;
  settings.max_evaluations = 40;
  settings.validation_evaluations = 4;
  settings.energy_tolerance = 0.01;
  BO optimizer(settings);
  auto jobs = optimizer.ask(20);
  check(jobs.size() == 4 && optimizer.ask(1).empty());
  for (std::size_t i = 0; i < jobs.size(); ++i)
    for (std::size_t j = 0; j < i; ++j) check(jobs[i].parameter != jobs[j].parameter);
  bool invalid = false;
  try { optimizer.tell(jobs[0].id, {1.0, -1.0}); }
  catch (const std::invalid_argument&) { invalid = true; }
  check(invalid && optimizer.pending().size() == 4);
  optimizer.fail(jobs.back().id);
  jobs.pop_back();
  for (auto it = jobs.rbegin(); it != jobs.rend(); ++it)
    optimizer.tell(it->id, {std::pow(it->parameter - 0.37, 2), 0.005});
  invalid = false;
  try { optimizer.tell(jobs[0].id, {0.0, 0.005}); }
  catch (const std::invalid_argument&) { invalid = true; }
  check(invalid);
  std::size_t rounds{};
  while (optimizer.status() != BO::Status::complete &&
         optimizer.status() != BO::Status::budget_exhausted) {
    auto batch = optimizer.ask(4);
    check(!batch.empty() && ++rounds < 100);
    for (auto it = batch.rbegin(); it != batch.rend(); ++it) {
      const double x = it->parameter;
      // Two competing basins; the lower one is near 0.37.
      const double energy = std::min(std::pow(x - 0.37, 2), 0.035 + std::pow(x - 0.85, 2));
      const double noise = (it->id % 2 == 0 ? 1.0 : -1.0) * 0.002;
      optimizer.tell(it->id, {energy + noise, 0.005});
    }
  }
  const auto result = optimizer.recommendation();
  check(result.validated && std::abs(result.parameter - 0.37) < 0.15);
  check(std::abs(result.standard_error - 0.0025) < 1e-12);
  check(optimizer.ask(4).empty());

  BO failures(settings);
  for (std::size_t i = 0; i < settings.max_evaluations; ++i) {
    auto batch = failures.ask(1);
    check(batch.size() == 1);
    failures.fail(batch[0].id);
  }
  check(failures.status() == BO::Status::budget_exhausted);

  auto small{settings};
  small.initial_points = 2;
  small.candidate_points = 2;
  small.max_pending = 2;
  BO replication(small);
  auto initial = replication.ask(2);
  replication.tell(initial[0].id, {1.0, 1.0});
  replication.tell(initial[1].id, {1.0, 1.0});
  auto repeat = replication.ask(1);
  check(repeat.size() == 1 && repeat[0].purpose == BO::Purpose::replication);
  replication.tell(repeat[0].id, {1.0, 1.0});
  for (int i = 0; i < 5; ++i) {
    repeat = replication.ask(1);
    check(repeat.size() == 1 && repeat[0].purpose == BO::Purpose::replication);
    replication.tell(repeat[0].id, {1.0, 1.0});
  }

  BO exhausted_grid(small);
  initial = exhausted_grid.ask(2);
  exhausted_grid.tell(initial[0].id, {0.0, 0.0});
  exhausted_grid.tell(initial[1].id, {1.0, 0.0});
  auto validation = exhausted_grid.ask(2);
  check(!validation.empty() && validation[0].purpose == BO::Purpose::validation);
  for (const auto& job : validation) exhausted_grid.fail(job.id);
  for (const auto& job : exhausted_grid.ask(2)) exhausted_grid.fail(job.id);
  check(exhausted_grid.status() == BO::Status::budget_exhausted);

  auto logarithmic{settings};
  logarithmic.logarithmic = true;
  logarithmic.lower_bound = 0.01;
  logarithmic.upper_bound = 100.0;
  logarithmic.max_pending = 5;
  BO log_search(logarithmic);
  auto logarithmic_jobs = log_search.ask(5);
  check(logarithmic_jobs.front().parameter == 0.01 && logarithmic_jobs.back().parameter == 100.0);
  check(std::abs(logarithmic_jobs[2].parameter - 1.0) < 1e-12);
  auto exact_settings{small};
  exact_settings.exact_budget = true;
  exact_settings.max_evaluations = 9;
  BO exact(exact_settings);
  std::size_t search_jobs{};
  for (int round = 0; round < 30; ++round) {
    auto batch = exact.ask(2);
    if (batch.empty()) break;
    for (const auto& job : batch) {
      if (job.purpose != BO::Purpose::validation) ++search_jobs;
      exact.tell(job.id, {1.0, 0.0});
    }
  }
  check(search_jobs == exact_settings.max_evaluations);
  check(exact.status() == BO::Status::complete);

  auto asynchronous{settings};
  asynchronous.candidate_points = 17;
  asynchronous.max_pending = 4;
  asynchronous.max_evaluations = 20;
  BO asynchronous_initialization(asynchronous);
  auto first_wave = asynchronous_initialization.ask(4);
  check(first_wave.size() == 4);
  asynchronous_initialization.tell(first_wave[0].id, {0.0, 0.005});
  auto final_initial = asynchronous_initialization.ask(1);
  check(final_initial.size() == 1);
  for (std::size_t i = 1; i < first_wave.size(); ++i)
    asynchronous_initialization.tell(first_wave[i].id, {1.0, 0.005});
  check(asynchronous_initialization.ask(3).size() == 3);

  auto settled{settings};
  settled.candidate_points = settled.initial_points;
  settled.max_pending = settled.initial_points;
  settled.max_evaluations = 16;
  settled.energy_tolerance = 0.001;
  BO settled_search(settled);
  auto settled_jobs = settled_search.ask(settled.initial_points);
  for (std::size_t i = 0; i < settled_jobs.size(); ++i)
    settled_search.tell(settled_jobs[i].id, {i == 0 ? 0.0 : 1.0, 0.0015});
  check(settled_search.status() == BO::Status::validating);

  auto bounded{settings};
  bounded.initial_points = 5;
  bounded.candidate_points = 17;
  bounded.max_pending = 8;
  bounded.max_exploration_pending = 4;
  bounded.max_evaluations = 16;
  bounded.energy_tolerance = 0.001;
  BO bounded_search(bounded);
  auto bounded_initial = bounded_search.ask(8);
  check(bounded_initial.size() == 4);
  for (const auto& job : bounded_initial)
    bounded_search.tell(job.id, {job.parameter == bounded.lower_bound ? 0.0 : 1.0, 0.01});
  auto bounded_batch = bounded_search.ask(8);
  std::size_t novel{}, replications{};
  for (const auto& job : bounded_batch) {
    novel += job.purpose == BO::Purpose::exploration;
    replications += job.purpose == BO::Purpose::replication;
  }
  check(bounded_batch.size() == 8 && novel <= bounded.max_exploration_pending && replications != 0);

  std::cout << "BayesOptimizer scheduling, failure, noisy search and validation checks passed\n";
}
