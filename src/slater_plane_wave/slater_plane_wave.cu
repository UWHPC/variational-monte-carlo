#include <xpu/xpu.hpp>
#include "slater_plane_wave.hpp"
#include "slater_plane_wave_kernels.hpp"
#include "particles/particles.hpp"
#include "utilities/matrix.hpp"
#include <xpu/soa.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

struct nVectorCandidate {
  int n_x;
  int n_y;
  int n_z;
  int magnitude_squared;
};

} // namespace

void SlaterPlaneWave::initialize() {
  const auto num_orbitals{this->num_orbitals()};
  const auto n_max{
    scast<int>(xpu::ceil(xpu::cbrt(scast<fp_t>(num_orbitals)))) + 2
  };
  const auto side{scast<std::size_t>(2 * n_max + 1)};

  std::vector<nVectorCandidate> candidates{};
  candidates.reserve(side * side * side);

  for (auto n_x = -n_max; n_x <= n_max; ++n_x) {
    for (auto n_y = -n_max; n_y <= n_max; ++n_y) {
      for (auto n_z = -n_max; n_z <= n_max; ++n_z) {
        if (!is_canonical(n_x, n_y, n_z)) { continue; }

        const auto magnitude_squared{
          n_x * n_x +
          n_y * n_y +
          n_z * n_z
        };
        candidates.emplace_back(
          n_x, n_y, n_z,
          magnitude_squared
        );
      }
    }
  }

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const nVectorCandidate& a, const nVectorCandidate& b) {
      return
        std::tie(a.magnitude_squared, a.n_x, a.n_y, a.n_z) <
        std::tie(b.magnitude_squared, b.n_x, b.n_y, b.n_z);
    }
  );

  std::vector<int> n_x(num_orbitals);
  std::vector<int> n_y(num_orbitals);
  std::vector<int> n_z(num_orbitals);
  std::vector<fp_t> k_x(num_orbitals);
  std::vector<fp_t> k_y(num_orbitals);
  std::vector<fp_t> k_z(num_orbitals);
  std::vector<std::size_t> orbital_k_index(num_orbitals);
  std::vector<std::uint8_t> orbital_type(num_orbitals);

  const auto two_pi_inv_L{
    2.0_fp * std::numbers::pi_v<fp_t> / this->box_length()
  };

  auto orbital{0uz};
  auto k_idx{0uz};

  for (const auto& candidate : candidates) {
    if (orbital >= num_orbitals) { break; }

    n_x[k_idx] = candidate.n_x;
    n_y[k_idx] = candidate.n_y;
    n_z[k_idx] = candidate.n_z;

    k_x[k_idx] = two_pi_inv_L * scast<fp_t>(candidate.n_x);
    k_y[k_idx] = two_pi_inv_L * scast<fp_t>(candidate.n_y);
    k_z[k_idx] = two_pi_inv_L * scast<fp_t>(candidate.n_z);

    orbital_k_index[orbital] = k_idx;
    orbital_type[orbital] = 0u;
    ++orbital;

    if (candidate.magnitude_squared != 0 && orbital < num_orbitals) {
      orbital_k_index[orbital] = k_idx;
      orbital_type[orbital] = 1u;
      ++orbital;
    }

    ++k_idx;
  }

  num_unique_k_ = k_idx;
  trig_row_stride_ = xpu::handle_pad<fp_t>(this->num_unique_k());

  auto n_vector{this->n_vector()};
  auto k_vector{this->k_vector()};

  xpu::copy_n(n_vector[idx(Axis::X)], n_x.data(), this->num_unique_k());
  xpu::copy_n(n_vector[idx(Axis::Y)], n_y.data(), this->num_unique_k());
  xpu::copy_n(n_vector[idx(Axis::Z)], n_z.data(), this->num_unique_k());

  xpu::copy_n(k_vector[idx(Axis::X)], k_x.data(), this->num_unique_k());
  xpu::copy_n(k_vector[idx(Axis::Y)], k_y.data(), this->num_unique_k());
  xpu::copy_n(k_vector[idx(Axis::Z)], k_z.data(), this->num_unique_k());

  xpu::copy_n(this->orbital_k_index(), orbital_k_index.data(), num_orbitals);
  xpu::copy_n(this->orbital_type(), orbital_type.data(), num_orbitals);

  trig_cache_ = xpu::soa_batch<fp_t, NUM_TRIG_ARRAYS>(
    this->walker_count(),
    this->num_orbitals() * this->trig_row_stride()
  );
  trig_scratch_ = xpu::soa_batch<fp_t, NUM_SCRATCH_TRIG>(
    this->walker_count(),
    this->num_unique_k()
  );
}

SlaterPlaneWave::SlaterPlaneWave(const Particles& particles, fp_t box_lengthL)
  : num_orbitals_{particles.count()}
  , matrix_row_stride_{xpu::handle_pad<fp_t>(particles.count())}
  , matrix_size_{matrix_row_stride_ * particles.count()}
  , num_walkers_{particles.walker_count()}
  , box_length_{box_lengthL}
  , orbital_k_index_(particles.count())
  , orbital_type_(particles.count())
  , int_vec_{particles.count()}
  , k_vectors_{particles.count()}
  , walker_vectors_{particles.walker_count(), particles.count()}
  , trig_cache_{particles.walker_count(), 0uz}
  , trig_scratch_{particles.walker_count(), 0uz}
  , matrices_{particles.walker_count(), matrix_row_stride_ * particles.count()}
  , reduction_scratch_{particles.walker_count()}
  , lu_factorization_{particles.count(), matrix_row_stride_}
{
  this->initialize();
};

void SlaterPlaneWave::restore_trig_row(
  std::size_t particle,
  std::size_t walker
) {
  const auto row_offset{
    scast<std::ptrdiff_t>(particle * this->trig_row_stride())
  };

  const auto sin_start{this->sin_cache(walker) + row_offset};
  const auto cos_start{this->cos_cache(walker) + row_offset};

  xpu::copy_n(
    sin_start,
    trig_scratch_.view<1uz, SIN_SAVED>(walker)[0uz],
    this->num_unique_k()
  );
  xpu::copy_n(
    cos_start,
    trig_scratch_.view<1uz, COS_SAVED>(walker)[0uz],
    this->num_unique_k()
  );
}

void SlaterPlaneWave::save_trig_row(
  std::size_t particle,
  std::size_t walker
) {
  const auto row_offset{
    scast<std::ptrdiff_t>(particle * this->trig_row_stride())
  };
  const auto sin_start{this->sin_cache(walker) + row_offset};
  const auto cos_start{this->cos_cache(walker) + row_offset};

  xpu::copy_n(
    trig_scratch_.view<1uz, SIN_SAVED>(walker)[0uz],
    sin_start,
    this->num_unique_k()
  );
  xpu::copy_n(
    trig_scratch_.view<1uz, COS_SAVED>(walker)[0uz],
    cos_start,
    this->num_unique_k()
  );
}

void SlaterPlaneWave::update_trig_cache(
  std::size_t particle,
  Particles::View particles,
  std::size_t walker
) {
  this->save_trig_row(particle, walker);
  kernel::slater::update_trig_cache(this->view(walker), particle, particles);
}

void SlaterPlaneWave::accept_move(
  std::size_t particle,
  const fp_t* new_row,
  fp_t ratio,
  std::size_t walker
) noexcept {
  const auto inv_ratio{1.0_fp / ratio};
  if (!std::isfinite(inv_ratio)) { return; }
  auto slater{this->view(walker)};

  const auto particle_offset{this->matrix_row_stride() * particle};
  xpu::copy_n(
    this->inv_d_col(walker),
    &this->inv_determinant(walker)[particle_offset],
    this->num_orbitals()
  );
  xpu::zero_n(this->solution(walker), this->num_orbitals());

  kernel::slater::k_compute_sk(slater, particle, new_row);

  kernel::slater::k_update_inverse(slater, particle, inv_ratio);

  xpu::copy_n(
    &this->determinant(walker)[particle_offset],
    new_row,
    this->num_orbitals()
  );
}

fp_t* SlaterPlaneWave::build_row(
  std::size_t particle,
  std::size_t walker
) noexcept {
  kernel::slater::build_row(this->view(walker), particle);

  return this->new_row(walker);
}

fp_t SlaterPlaneWave::determinant_ratio(
  std::size_t particle,
  const fp_t* new_row,
  std::size_t walker
) noexcept {
  return kernel::slater::determinant_ratio(
    this->view(walker), particle, new_row
  );
}

void SlaterPlaneWave::add_derivatives(
  Particles::View particles,
  std::size_t walker
) noexcept {
  kernel::slater::add_derivatives(this->view(walker), particles);
}

void SlaterPlaneWave::initialize_matrices(
  Particles::BatchView particles,
  std::size_t num_threads
) {
  kernel::slater::initialize_matrices(
    this->batch_view(),
    particles,
    num_threads
  );
}

fp_t SlaterPlaneWave::log_abs_det(
  Particles::View particles,
  std::size_t walker
) {
  auto slater{this->view(walker)};

  kernel::slater::build_trig_cache(slater, particles);
  kernel::slater::build_determinant(slater);

  xpu::copy_n(
    this->lower_upper(walker),
    this->determinant(walker),
    this->matrix_size()
  );

  return this->factorize(walker);
}

fp_t SlaterPlaneWave::factorize(std::size_t walker) {
  auto slater{this->view(walker)};

  const auto factorization_status{
    lu_factorization_.factorize(slater.lower_upper)
  };
  if (factorization_status == xpu::linalg::status::singular) {
    return -std::numeric_limits<fp_t>::infinity();
  }

  const auto log_abs_det{
    kernel::slater::compute_log_abs_det(slater)
  };
  if (!std::isfinite(log_abs_det)) {
    return -std::numeric_limits<fp_t>::infinity();
  }

  lu_factorization_.invert(
    slater.lower_upper,
    slater.inv_determinant
  );
  xpu::linalg::transpose_square(
    slater.inv_determinant,
    slater.num_orbitals,
    slater.matrix_row_stride
  );

  return log_abs_det;
}
