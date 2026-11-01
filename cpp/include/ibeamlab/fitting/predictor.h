#pragma once
#include <ibeamlab/fitting/problem.h>
#include <ibeamlab/inference/models.h>
namespace ibeamlab::fitting {
/** @brief Batch spectrum provider. One instance belongs to one fitting worker. */
class IBEAMLAB_API Predictor {
  public:
    virtual ~Predictor() = default;
    virtual bool can_vary(const core::ParameterTarget &) const { return true; }
    virtual void predict(std::span<const core::Experiment>, std::span<simulation::Result>) = 0;
    virtual void request_stop() noexcept {}
    virtual bool stopped() const noexcept { return false; }
};
/** @brief Non-owning adapter retaining true ONNX population execution and outputs. */
class IBEAMLAB_API ForwardPredictor final : public Predictor {
    inference::ForwardModel &model_;

  public:
    explicit ForwardPredictor(inference::ForwardModel &model) : model_(model) {}
    bool can_vary(const core::ParameterTarget &) const override;
    void predict(std::span<const core::Experiment>, std::span<simulation::Result>) override;
};
/** @brief Non-owning adapter for native or Python-backed simulation providers. */
class IBEAMLAB_API SimulatorPredictor final : public Predictor {
    simulation::Simulator &simulator_;

  public:
    explicit SimulatorPredictor(simulation::Simulator &simulator) : simulator_(simulator) {}
    void predict(std::span<const core::Experiment>, std::span<simulation::Result>) override;
    void request_stop() noexcept override { simulator_.request_stop(); }
    bool stopped() const noexcept override { return simulator_.stopped(); }
};
} // namespace ibeamlab::fitting
