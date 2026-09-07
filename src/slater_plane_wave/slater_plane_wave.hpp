#pragma once

#include "../particles/particles.hpp"
#include <xpu/buffer.hpp>
#include <xpu/linear_algebra.hpp>
#include <xpu/soa.hpp>
#include "../utilities/macros.hpp"
#include <xpu/math.hpp>

#include <cstddef>
#include <cstdlib>
#include <cstring>

class SlaterPlaneWave {
private:
  std::size_t num_orbitals_;
  std::size_t num_unique_k_;
  std::size_t trig_row_stride_;
  std::size_t matrix_row_stride_;
  std::size_t matrix_size_;
  std::size_t num_walkers_;
  fp_t box_length_;

  enum OrbKIndex : std::size_t { K, NUM_ORB_K };
  xpu::soa<std::size_t, NUM_ORB_K> orbital_k_index_;
  
  enum OrbType : std::size_t { O, NUM_ORB_TYPE };
  xpu::soa<std::uint8_t, NUM_ORB_TYPE> orbital_type_;

  enum IntVectorIndex : std::size_t { N_X, N_Y, N_Z, NUM_INT_VECTORS };
  xpu::soa<int, NUM_INT_VECTORS> int_vec_;

  enum KVectorIndex : std::size_t {
    K_X,
    K_Y,
    K_Z,
    NUM_K_VECTORS
  };
  xpu::soa<fp_t, NUM_K_VECTORS> k_vectors_;

  enum WalkerVectorIndex : std::size_t {
    SOLUTION,
    NEW_ROW,
    INV_D_COL,
    NUM_WALKER_VECTORS
  };
  xpu::soa_batch<fp_t, NUM_WALKER_VECTORS> walker_vectors_;

  enum TrigIndex : std::size_t { SIN_CACHE, COS_CACHE, NUM_TRIG_ARRAYS };
  xpu::soa_batch<fp_t, NUM_TRIG_ARRAYS> trig_cache_;

  enum ScratchTrigIndex : std::size_t { SIN_SAVED, COS_SAVED, NUM_SCRATCH_TRIG };
  xpu::soa_batch<fp_t, NUM_SCRATCH_TRIG> trig_scratch_;

  enum MatrixIndex : std::size_t { D, INV_D, LU, NUM_MATRIX };
  xpu::soa_batch<fp_t, NUM_MATRIX> matrices_;

  xpu::buffer<fp_t> reduction_scratch_;
  xpu::linalg::lu_factorization<fp_t> lu_factorization_;

public:
  struct View {
    std::size_t num_orbitals{};
    std::size_t num_unique_k{};
    std::size_t trig_row_stride{};
    std::size_t matrix_row_stride{};

    xpu::soa_view<fp_t, idx(Axis::NUM)> k_vector{nullptr, 0uz};
    const std::size_t* orbital_k_index{};
    const std::uint8_t* orbital_type{};

    fp_t* determinant{};
    fp_t* inv_determinant{};
    fp_t* lower_upper{};
    fp_t* reduction_scratch{};
    fp_t* solution{};
    fp_t* new_row{};
    fp_t* inv_d_col{};

    fp_t* sin_cache{};
    fp_t* cos_cache{};
    fp_t* sin_saved{};
    fp_t* cos_saved{};
  };

  struct BatchView {
    std::size_t num_orbitals{};
    std::size_t num_unique_k{};
    std::size_t trig_row_stride{};
    std::size_t matrix_row_stride{};

    xpu::soa_view<fp_t, idx(Axis::NUM)> k_vector{nullptr, 0uz};
    const std::size_t* orbital_k_index{};
    const std::uint8_t* orbital_type{};

    xpu::soa_batch_view<fp_t, NUM_MATRIX> matrices{
      nullptr, 0uz, 0uz, 0uz
    };
    xpu::soa_batch_view<fp_t, NUM_WALKER_VECTORS> walker_vectors{
      nullptr, 0uz, 0uz, 0uz
    };
    xpu::soa_batch_view<fp_t, NUM_TRIG_ARRAYS> trig_cache{
      nullptr, 0uz, 0uz, 0uz
    };
    xpu::soa_batch_view<fp_t, NUM_SCRATCH_TRIG> trig_scratch{
      nullptr, 0uz, 0uz, 0uz
    };

    fp_t* reduction_scratch{};

    [[nodiscard]] CUDA_CALLABLE
    std::size_t walker_count() const noexcept {
      return matrices.batch_count();
    }

    [[nodiscard]] CUDA_CALLABLE
    View view(std::size_t walker) noexcept {
      auto matrix{matrices.view(walker)};
      auto vectors{walker_vectors.view(walker)};
      auto cache{trig_cache.view(walker)};
      auto scratch{trig_scratch.view(walker)};

      return {
        num_orbitals,
        num_unique_k,
        trig_row_stride,
        matrix_row_stride,
        k_vector,
        orbital_k_index,
        orbital_type,
        matrix[D],
        matrix[INV_D],
        matrix[LU],
        reduction_scratch + walker,
        vectors[SOLUTION],
        vectors[NEW_ROW],
        vectors[INV_D_COL],
        cache[SIN_CACHE],
        cache[COS_CACHE],
        scratch[SIN_SAVED],
        scratch[COS_SAVED]
      };
    }
  };

  explicit SlaterPlaneWave(const Particles& particles, fp_t box_length);

  SlaterPlaneWave(const SlaterPlaneWave&) = delete;
  SlaterPlaneWave& operator=(const SlaterPlaneWave&) = delete;

  SlaterPlaneWave(SlaterPlaneWave&&) = delete;
  SlaterPlaneWave& operator=(SlaterPlaneWave&&) = delete;

  [[nodiscard]] std::size_t num_orbitals() const noexcept { return num_orbitals_; }
  [[nodiscard]] std::size_t num_unique_k() const noexcept { return num_unique_k_; }
  [[nodiscard]] std::size_t trig_row_stride() const noexcept { return trig_row_stride_; }
  [[nodiscard]] std::size_t matrix_row_stride() const noexcept { return matrix_row_stride_; }
  [[nodiscard]] std::size_t matrix_size() const noexcept { return matrix_size_; }
  [[nodiscard]] std::size_t walker_count() const noexcept { return num_walkers_; }
  [[nodiscard]] fp_t box_length() const noexcept { return box_length_; }

  [[nodiscard]]       std::size_t* orbital_k_index()       noexcept { return orbital_k_index_[K]; }
  [[nodiscard]] const std::size_t* orbital_k_index() const noexcept { return orbital_k_index_[K]; }

  [[nodiscard]]       std::uint8_t* orbital_type()       noexcept { return orbital_type_[O]; }
  [[nodiscard]] const std::uint8_t* orbital_type() const noexcept { return orbital_type_[O]; }

  [[nodiscard]] fp_t* determinant(std::size_t walker = 0uz) noexcept {
    return matrices_.view<1uz, D>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* determinant(std::size_t walker = 0uz) const noexcept {
    return matrices_.view<1uz, D>(walker)[0uz];
  }

  [[nodiscard]] fp_t* inv_determinant(std::size_t walker = 0uz) noexcept {
    return matrices_.view<1uz, INV_D>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* inv_determinant(std::size_t walker = 0uz) const noexcept {
    return matrices_.view<1uz, INV_D>(walker)[0uz];
  }

  [[nodiscard]] fp_t* lower_upper(std::size_t walker = 0uz) noexcept {
    return matrices_.view<1uz, LU>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* lower_upper(std::size_t walker = 0uz) const noexcept {
    return matrices_.view<1uz, LU>(walker)[0uz];
  }

  [[nodiscard]]
  xpu::soa_view<int, idx(Axis::NUM)> n_vector() {
    return int_vec_.view<idx(Axis::NUM), N_X>();
  }

  [[nodiscard]]
  xpu::soa_view<const int, idx(Axis::NUM)> n_vector() const {
    return int_vec_.view<idx(Axis::NUM), N_X>();
  }

  [[nodiscard]]
  xpu::soa_view<fp_t, idx(Axis::NUM)> k_vector() {
    return k_vectors_.view<idx(Axis::NUM), K_X>();
  }

  [[nodiscard]]
  xpu::soa_view<const fp_t, idx(Axis::NUM)> k_vector() const {
    return k_vectors_.view<idx(Axis::NUM), K_X>();
  }

  [[nodiscard]] fp_t* solution(std::size_t walker = 0uz) noexcept {
    return walker_vectors_.view<1uz, SOLUTION>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* solution(std::size_t walker = 0uz) const noexcept {
    return walker_vectors_.view<1uz, SOLUTION>(walker)[0uz];
  }

  [[nodiscard]] fp_t* sin_cache(std::size_t walker = 0uz) noexcept {
    return trig_cache_.view<1uz, SIN_CACHE>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* sin_cache(std::size_t walker = 0uz) const noexcept {
    return trig_cache_.view<1uz, SIN_CACHE>(walker)[0uz];
  }

  [[nodiscard]] fp_t* cos_cache(std::size_t walker = 0uz) noexcept {
    return trig_cache_.view<1uz, COS_CACHE>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* cos_cache(std::size_t walker = 0uz) const noexcept {
    return trig_cache_.view<1uz, COS_CACHE>(walker)[0uz];
  }

  void restore_trig_row(std::size_t particle, std::size_t walker = 0uz);
  void update_trig_cache(
    std::size_t particle,
    Particles::View particles,
    std::size_t walker = 0uz
  );

  fp_t log_abs_det(Particles::View particles, std::size_t walker = 0uz);

  fp_t* build_row(std::size_t particle, std::size_t walker = 0uz) noexcept;

  [[nodiscard]] fp_t determinant_ratio(
    std::size_t particle,
    const fp_t* new_row,
    std::size_t walker = 0uz
  ) noexcept;

  void accept_move(
    std::size_t particle,
    const fp_t* new_row,
    fp_t ratio,
    std::size_t walker = 0uz
  ) noexcept;

  void add_derivatives(
    Particles::View particles,
    std::size_t walker = 0uz
  ) noexcept;

  [[nodiscard]]
  BatchView batch_view() noexcept {
    return {
      this->num_orbitals(),
      this->num_unique_k(),
      this->trig_row_stride(),
      this->matrix_row_stride(),
      this->k_vector(),
      this->orbital_k_index(),
      this->orbital_type(),
      matrices_.view(),
      walker_vectors_.view(),
      trig_cache_.view(),
      trig_scratch_.view(),
      reduction_scratch_.data()
    };
  }

  [[nodiscard]]
  View view(std::size_t walker = 0uz) noexcept {
    return this->batch_view().view(walker);
  }

private:
  void initialize();
  void save_trig_row(std::size_t particle, std::size_t walker);

  [[nodiscard]] fp_t* new_row(std::size_t walker) noexcept {
    return walker_vectors_.view<1uz, NEW_ROW>(walker)[0uz];
  }
  [[nodiscard]] fp_t* inv_d_col(std::size_t walker) noexcept {
    return walker_vectors_.view<1uz, INV_D_COL>(walker)[0uz];
  }
  [[nodiscard]] fp_t* reduction_scratch(std::size_t walker) noexcept {
    return reduction_scratch_.data() + walker;
  }
};
