#pragma once
#include <ibeamlab/analysis/plan.h>
namespace ibeamlab::analysis {
/** Combines measurement providers while retaining population batching.
 * Mutable scratch belongs to one caller. Cancellation may be requested externally. */
class IBEAMLAB_API Predictor final : public fitting::Predictor {
    Plan plan_;
    struct Workspace {
        std::vector<core::Experiment> inputs;
        std::vector<simulation::Result> outputs;
    };
    // Distinct detector layouts must not shrink/destroy each other's buffers.
    std::vector<Workspace> workspaces_;

  public:
    explicit Predictor(Plan);
    bool can_vary(const core::ParameterTarget &) const override;
    bool can_vary_layer_properties() const override;
    void predict(std::span<const core::Experiment>, std::span<simulation::Result>) override;
    void request_stop() noexcept override;
    void reset_stop() noexcept override;
    bool stopped() const noexcept override;
};
} // namespace ibeamlab::analysis
