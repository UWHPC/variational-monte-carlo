#pragma once

#include "../support/checks.hpp"

#include <catch2/catch_test_macros.hpp>

#include "particles/particles.hpp"
#include "slater_plane_wave/slater_plane_wave.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace test {

inline constexpr auto slater_particle_count{3uz};
inline constexpr auto slater_walker_count{2uz};
inline constexpr auto slater_box_length{11.0_fp};
inline constexpr std::array slater_positions{
  std::array{
    std::array{0.3_fp, 0.4_fp, 0.5_fp},
    std::array{1.7_fp, 0.2_fp, 2.1_fp},
    std::array{2.2_fp, 1.8_fp, 0.9_fp}
  },
  std::array{
    std::array{0.8_fp, 1.1_fp, 0.3_fp},
    std::array{2.7_fp, 0.4_fp, 1.9_fp},
    std::array{4.2_fp, 3.5_fp, 2.6_fp}
  }
};

inline void set_slater_positions(
  Particles& target,
  std::size_t target_walker,
  std::size_t source_walker
) {
  auto positions{target.pos(target_walker)};
  std::array<fp_t, slater_particle_count> host_positions{};

  for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
    for (auto particle{0uz}; particle < slater_particle_count; ++particle) {
      host_positions[particle] = slater_positions[source_walker][particle][axis];
    }
    xpu::copy_n(positions[axis], host_positions.data(), host_positions.size());
  }
}

} // namespace test

TEST_CASE("Batched Slater walkers match independent calculations", "[slater][walkers]") {
  Particles particles{test::slater_particle_count, test::slater_walker_count};
  for (auto walker{0uz}; walker < test::slater_walker_count; ++walker) {
    test::set_slater_positions(particles, walker, walker);
  }

  SlaterPlaneWave batched{particles, test::slater_box_length};
  batched.initialize_matrices(particles.batch_view(), 2uz);

  std::array<fp_t, test::slater_walker_count> batched_log_determinants{};

  for (auto walker{0uz}; walker < test::slater_walker_count; ++walker) {
    Particles independent_particles{test::slater_particle_count};
    test::set_slater_positions(independent_particles, 0uz, walker);
    SlaterPlaneWave independent{independent_particles, test::slater_box_length};

    batched_log_determinants[walker] = batched.factorize(walker);
    const auto independent_log_determinant{
      independent.log_abs_det(independent_particles.view())
    };

    CAPTURE(walker, batched_log_determinants[walker], independent_log_determinant);
    REQUIRE(std::isfinite(batched_log_determinants[walker]));
    require_near(batched_log_determinants[walker], independent_log_determinant);

    const auto proposal_walker{(walker + 1uz) % test::slater_walker_count};
    test::set_slater_positions(particles, walker, proposal_walker);
    test::set_slater_positions(independent_particles, 0uz, proposal_walker);

    constexpr auto moved_particle{0uz};
    batched.update_trig_cache(moved_particle, particles.view(walker), walker);
    independent.update_trig_cache(moved_particle, independent_particles.view());

    const auto batched_ratio{
      batched.determinant_ratio(
        moved_particle,
        batched.build_row(moved_particle, walker),
        walker
      )
    };
    const auto independent_ratio{
      independent.determinant_ratio(
        moved_particle,
        independent.build_row(moved_particle)
      )
    };

    REQUIRE(std::isfinite(batched_ratio));
    require_near(batched_ratio, independent_ratio);
  }

  REQUIRE(
    std::abs(batched_log_determinants[0uz] - batched_log_determinants[1uz]) >
    test_tolerance
  );
}

TEST_CASE("Batched Slater initialization preserves singular detection", "[slater][walkers]") {
  Particles particles{test::slater_particle_count, test::slater_walker_count};

  for (auto walker{0uz}; walker < test::slater_walker_count; ++walker) {
    auto positions{particles.pos(walker)};

    for (auto axis{idx(Axis::X)}; axis < idx(Axis::NUM); ++axis) {
      xpu::fill_n(positions[axis], particles.count(), 0.5_fp);
    }
  }

  SlaterPlaneWave slater{particles, test::slater_box_length};
  slater.initialize_matrices(particles.batch_view(), 2uz);

  for (auto walker{0uz}; walker < test::slater_walker_count; ++walker) {
    const auto log_determinant{slater.factorize(walker)};
    REQUIRE(log_determinant == -std::numeric_limits<fp_t>::infinity());
  }
}
