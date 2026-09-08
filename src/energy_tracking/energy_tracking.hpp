#pragma once

#include "../utilities/components.hpp"
#include "../particles/particles.hpp"
#include <xpu/buffer.hpp>
#include <xpu/soa.hpp>
#include <xpu/math.hpp>
#include <cstddef>

class EnergyTracker {
private:
  fp_t box_length_;

  static constexpr fp_t EWALD_RECIPROCAL_TOLERANCE{1.0e-6_fp};
  fp_t ewald_alpha_;
  fp_t ewald_correction_;
  fp_t ewald_background_;

  std::size_t num_g_vectors_;
  std::size_t num_walkers_;
  std::size_t num_particles_;

  enum class SharedArray : std::size_t {
    G_X,
    G_Y,
    G_Z,
    G_WEIGHTS,
    NUM_ARRAYS
  };
  enum class WalkerArray : std::size_t {
    S_REAL,
    S_IMAG,
    NUM_ARRAYS
  };
  enum class WalkerScalar : std::size_t {
    V_REAL,
    V_RECIP,
    NUM_ARRAYS
  };

  xpu::soa<fp_t, idx(SharedArray::NUM_ARRAYS)> shared_data_;
  xpu::soa_batch<fp_t, idx(WalkerArray::NUM_ARRAYS)> walker_data_;
  xpu::soa<fp_t, idx(WalkerScalar::NUM_ARRAYS)> walker_scalars_;
  mutable xpu::buffer<fp_t> reduction_scratch_;
  xpu::buffer<std::byte> sum_scratch_;

public:
  struct View {
    fp_t box_length{};
    std::size_t num_g_vectors{};
    fp_t ewald_alpha{};
    fp_t ewald_correction{};
    fp_t ewald_background{};
    xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector{nullptr, 0uz};
    const fp_t* g_weights{};
    fp_t* sum_real{};
    fp_t* sum_imag{};
    fp_t* real_energy{};
    fp_t* reciprocal_energy{};
    fp_t* reduction_scratch{};
  };

  struct BatchView {
    fp_t box_length{};
    std::size_t num_g_vectors{};
    fp_t ewald_alpha{};
    fp_t ewald_correction{};
    fp_t ewald_background{};

    xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector{nullptr, 0uz};
    const fp_t* g_weights{};

    xpu::soa_batch_view<fp_t, idx(WalkerArray::NUM_ARRAYS)> walker_data{
      nullptr, 0uz, 0uz, 0uz
    };
    xpu::soa_view<fp_t, idx(WalkerScalar::NUM_ARRAYS)> walker_scalars{
      nullptr, 0uz
    };

    fp_t* reduction_scratch{};

    [[nodiscard]] CUDA_CALLABLE
    std::size_t walker_count() const noexcept {
      return walker_data.batch_count();
    }

    [[nodiscard]] CUDA_CALLABLE
    View view(std::size_t walker) noexcept {
      auto data{walker_data.view(walker)};

      return {
        box_length,
        num_g_vectors,
        ewald_alpha,
        ewald_correction,
        ewald_background,
        g_vector,
        g_weights,
        data[idx(WalkerArray::S_REAL)],
        data[idx(WalkerArray::S_IMAG)],
        walker_scalars[idx(WalkerScalar::V_REAL)] + walker,
        walker_scalars[idx(WalkerScalar::V_RECIP)] + walker,
        reduction_scratch + walker
      };
    }
  };

  struct InitializationView {
    View energy{};
    Particles::View particles{};
    fp_t* reciprocal_partials{};
    fp_t* real_partials{};
  };

  struct InitializationBatchView {
    BatchView energy{};
    Particles::BatchView particles{};

    fp_t* reciprocal_partials{};
    fp_t* real_partials{};
    std::size_t reciprocal_partial_count{};
    std::size_t real_partial_count{};

    [[nodiscard]] CUDA_CALLABLE
    InitializationView view(std::size_t walker) noexcept {
      return {
        energy.view(walker),
        particles.view(walker),
        reciprocal_partials + walker * reciprocal_partial_count,
        real_partials + walker * real_partial_count
      };
    }

    [[nodiscard]] CUDA_CALLABLE
    InitializationView operator[](std::size_t walker) noexcept {
      return this->view(walker);
    }
  };

private:
  xpu::buffer<fp_t> reciprocal_partials_;
  xpu::buffer<fp_t> real_partials_;
  std::size_t reciprocal_partial_count_;
  std::size_t real_partial_count_;

  InitializationView initialization_view(Particles::View particles, std::size_t walker) noexcept;

  [[nodiscard]]
  InitializationBatchView initialization_batch_view(
    Particles::BatchView particles
  ) noexcept {
    return {
      this->batch_view(),
      particles,
      reciprocal_partials_.data(),
      real_partials_.data(),
      reciprocal_partial_count_,
      real_partial_count_
    };
  }

  void initialize_reduction_storage();
  void validate_initialization(const Particles& particles, std::size_t num_threads) const;

public:
  explicit EnergyTracker(
    fp_t box_length,
    std::size_t num_particles,
    std::size_t num_walkers = 1uz
  );
  explicit EnergyTracker(fp_t box_length, const Particles& particles)
    : EnergyTracker{
        box_length,
        particles.count(),
        particles.walker_count()
      }
  { }

  [[nodiscard]] std::size_t num_g_vectors() const noexcept {
    return num_g_vectors_;
  }
  [[nodiscard]] std::size_t walker_count() const noexcept {
    return num_walkers_;
  }

  void initialize(Particles& particles, std::size_t num_threads = 1uz);

  void initialize_reciprocal_energy(std::size_t walker = 0uz) noexcept;
  void initialize_real_energy(
    Particles::View particles,
    std::size_t walker = 0uz
  ) noexcept;

  void initialize_structure_factors(
    Particles::View particles,
    std::size_t walker = 0uz
  ) noexcept;

  void update_structure_factors(
    xpu::array<fp_t, idx(Axis::NUM)> old_pos,
    xpu::array<fp_t, idx(Axis::NUM)> new_pos,
    std::size_t walker = 0uz
  ) noexcept;

  void update_real_energy(
    std::size_t moved,
    xpu::array<fp_t, idx(Axis::NUM)> old_pos,
    Particles::View particles,
    std::size_t walker = 0uz
  ) noexcept;

  fp_t eval_total_energy(
    Particles::View particles,
    std::size_t walker = 0uz
  ) noexcept {
    return kinetic_energy(particles, walker) + potential_energy(walker);
  }

  [[nodiscard]]
  BatchView batch_view() noexcept {
    return {
      box_length_,
      this->num_g_vectors(),
      ewald_alpha_,
      ewald_correction_,
      ewald_background_,
      this->g_vector(),
      this->g_weights(),
      walker_data_.view(),
      walker_scalars_.view(),
      reduction_scratch_.data()
    };
  }

  [[nodiscard]]
  View view(std::size_t walker = 0uz) noexcept {
    return this->batch_view().view(walker);
  }

  void accept_move(
    fp_t real_energy_delta,
    fp_t reciprocal_energy,
    std::size_t walker = 0uz
  ) noexcept;

private:
  [[nodiscard]]
  xpu::soa_view<fp_t, idx(Axis::NUM)> g_vector() {
    return shared_data_.view<idx(Axis::NUM), idx(SharedArray::G_X)>();
  }

  [[nodiscard]]
  xpu::soa_view<const fp_t, idx(Axis::NUM)> g_vector() const {
    return shared_data_.view<idx(Axis::NUM), idx(SharedArray::G_X)>();
  }

  [[nodiscard]] fp_t* g_weights() noexcept {
    return shared_data_[idx(SharedArray::G_WEIGHTS)];
  }
  [[nodiscard]] fp_t const* g_weights() const noexcept {
    return shared_data_[idx(SharedArray::G_WEIGHTS)];
  }

  [[nodiscard]] fp_t* sum_real(std::size_t walker) noexcept {
    return walker_data_.view<1uz, idx(WalkerArray::S_REAL)>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* sum_real(std::size_t walker) const noexcept {
    return walker_data_.view<1uz, idx(WalkerArray::S_REAL)>(walker)[0uz];
  }

  [[nodiscard]] fp_t* sum_imag(std::size_t walker) noexcept {
    return walker_data_.view<1uz, idx(WalkerArray::S_IMAG)>(walker)[0uz];
  }
  [[nodiscard]] fp_t const* sum_imag(std::size_t walker) const noexcept {
    return walker_data_.view<1uz, idx(WalkerArray::S_IMAG)>(walker)[0uz];
  }

  [[nodiscard]] fp_t* reduction_scratch(std::size_t walker) const noexcept {
    return reduction_scratch_.data() + walker;
  }

  fp_t kinetic_energy(
    Particles::View particles,
    std::size_t walker
  ) noexcept;
  fp_t potential_energy(std::size_t walker) const noexcept;
};
