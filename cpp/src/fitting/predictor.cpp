#include <algorithm>
#include <ibeamlab/fitting/predictor.h>
#include <type_traits>
namespace ibeamlab::fitting {
bool ForwardPredictor::can_vary(const core::ParameterTarget &target) const {
    const auto &f = std::get<model::ForwardModelMetadata>(model_.metadata().model);
    if (model_.metadata().layer_input() &&
        (std::holds_alternative<core::LayerThickness>(target) ||
         std::holds_alternative<core::SpeciesConcentration>(target)))
        return true;
    if (const auto *parameters = std::get_if<model::ParameterInputSchema>(&f.input_preparation))
        for (const auto &p : parameters->parameters)
            if (core::parameter_key(p.target) == core::parameter_key(target))
                return true;
    return std::visit(
        [&](const auto &t) {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, core::CalibrationLinear> ||
                          std::is_same_v<T, core::CalibrationOffset> ||
                          std::is_same_v<T, core::CalibrationQuadratic> ||
                          std::is_same_v<T, core::ParticlesSr>) {
                for (const auto &pipeline : f.output_preparation)
                    if (pipeline.measurement == t.measurement)
                        for (const auto &step : pipeline.steps) {
                            if constexpr (std::is_same_v<T, core::ParticlesSr>) {
                                if (std::holds_alternative<model::ExposureScale>(step))
                                    return true;
                            } else if (std::holds_alternative<model::Rebin>(step))
                                return true;
                        }
            }
            return false;
        },
        target);
}
void ForwardPredictor::predict(std::span<const core::Experiment> inputs,
                               std::span<simulation::Result> outputs) {
    model_.predict_batch(inputs, outputs);
}
void SimulatorPredictor::predict(std::span<const core::Experiment> inputs,
                                 std::span<simulation::Result> outputs) {
    if (inputs.size() != outputs.size())
        throw std::invalid_argument("predictor output dimensions differ");
    auto results = simulator_.simulate_batch(inputs);
    if (results.size() != inputs.size() && !simulator_.stopped())
        throw std::runtime_error("simulator batch result count differs");
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (i < results.size())
            outputs[i] = std::move(results[i]);
        else {
            outputs[i].spectra.clear();
            outputs[i].failure = simulation::Failure{"", "simulation cancelled"};
        }
    }
}
} // namespace ibeamlab::fitting
