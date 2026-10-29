#include <cmath>
#include <ibeamlab/core/parameter.h>
#include <unordered_set>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ibeamlab::core {
namespace {
const Measurement &measurement(const ExperimentalSetup &setup, const std::string &label) {
    const auto found = std::find_if(setup.measurements.begin(), setup.measurements.end(),
                                    [&](const auto &value) { return value.label == label; });
    if (found == setup.measurements.end())
        throw std::invalid_argument("unknown measurement: " + label);
    return *found;
}
Measurement &measurement(ExperimentalSetup &setup, const std::string &label) {
    return const_cast<Measurement &>(measurement(std::as_const(setup), label));
}
const Species &species(const Experiment &input, const SpeciesConcentration &target) {
    const auto &values = input.sample.layers.at(target.layer).species;
    const auto found = std::find_if(values.begin(), values.end(), [&](const auto &value) {
        return value.element == target.element;
    });
    if (found == values.end())
        throw std::invalid_argument("unknown species target: " + target.element);
    return *found;
}
} // namespace

std::string parameter_key(const ParameterTarget &target) {
    return std::visit(
        [](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, LayerThickness>)
                return "layer.thickness." + std::to_string(value.layer);
            else if constexpr (std::is_same_v<T, SpeciesConcentration>)
                return "layer.species." + std::to_string(value.layer) + "." + value.element;
            else if constexpr (std::is_same_v<T, BeamEnergy>)
                return "beam.energy." + value.measurement;
            else if constexpr (std::is_same_v<T, BeamSpread>)
                return "beam.spread." + value.measurement;
            else if constexpr (std::is_same_v<T, CalibrationLinear>)
                return "calibration.linear." + value.measurement;
            else if constexpr (std::is_same_v<T, CalibrationOffset>)
                return "calibration.offset." + value.measurement;
            else if constexpr (std::is_same_v<T, CalibrationQuadratic>)
                return "calibration.quadratic." + value.measurement;
            else if constexpr (std::is_same_v<T, DetectorResolution>)
                return "detector.resolution." + value.measurement;
            else
                return "acquisition.particles_sr." + value.measurement;
        },
        target);
}

double read_parameter(const Experiment &input, const ParameterTarget &target) {
    return std::visit(
        [&](const auto &value) -> double {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, LayerThickness>)
                return input.sample.layers.at(value.layer).thickness;
            else if constexpr (std::is_same_v<T, SpeciesConcentration>)
                return species(input, value).concentration;
            else {
                const auto &item = measurement(input.setup, value.measurement);
                if constexpr (std::is_same_v<T, BeamEnergy>)
                    return item.beam.energy;
                else if constexpr (std::is_same_v<T, BeamSpread>)
                    return item.beam.spread;
                else if constexpr (std::is_same_v<T, CalibrationLinear>)
                    return item.detector.calibrationLinear;
                else if constexpr (std::is_same_v<T, CalibrationOffset>)
                    return item.detector.calibrationOffset;
                else if constexpr (std::is_same_v<T, CalibrationQuadratic>)
                    return item.detector.calibrationQuadratic;
                else if constexpr (std::is_same_v<T, DetectorResolution>)
                    return item.detector.resolution;
                else
                    return item.acquisition.particlesSr;
            }
        },
        target);
}

void write_parameter(Experiment &input, const ParameterTarget &target, double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("parameter value must be finite");
    std::visit(
        [&](const auto &location) {
            using T = std::decay_t<decltype(location)>;
            if constexpr (std::is_same_v<T, LayerThickness>)
                input.sample.layers.at(location.layer).thickness = value;
            else if constexpr (std::is_same_v<T, SpeciesConcentration>) {
                auto &values = input.sample.layers.at(location.layer).species;
                const auto found =
                    std::find_if(values.begin(), values.end(), [&](const auto &item) {
                        return item.element == location.element;
                    });
                if (found == values.end())
                    throw std::invalid_argument("unknown species target: " + location.element);
                found->concentration = value;
            } else {
                auto &item = measurement(input.setup, location.measurement);
                if constexpr (std::is_same_v<T, BeamEnergy>)
                    item.beam.energy = value;
                else if constexpr (std::is_same_v<T, BeamSpread>)
                    item.beam.spread = value;
                else if constexpr (std::is_same_v<T, CalibrationLinear>)
                    item.detector.calibrationLinear = value;
                else if constexpr (std::is_same_v<T, CalibrationOffset>)
                    item.detector.calibrationOffset = value;
                else if constexpr (std::is_same_v<T, CalibrationQuadratic>)
                    item.detector.calibrationQuadratic = value;
                else if constexpr (std::is_same_v<T, DetectorResolution>)
                    item.detector.resolution = value;
                else
                    item.acquisition.particlesSr = value;
            }
        },
        target);
}

} // namespace ibeamlab::core

namespace ibeamlab::core {
void ParameterBounds::validate() const {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum)
        throw std::invalid_argument("parameter bounds must be finite and ordered");
}
bool ParameterBounds::contains(double value) const {
    validate();
    return std::isfinite(value) && value >= minimum && value <= maximum;
}
void apply(Experiment &experiment, std::span<const ParameterUpdate> updates) {
    auto candidate = experiment;
    std::unordered_set<std::string> targets;
    for (const auto &update : updates) {
        if (!targets.insert(parameter_key(update.target)).second)
            throw std::invalid_argument("duplicate parameter target: " +
                                        parameter_key(update.target));
        write_parameter(candidate, update.target, update.value);
    }
    candidate.validate();
    experiment = std::move(candidate);
}
} // namespace ibeamlab::core

namespace ibeamlab::core {
void validate_parameters(const Experiment &experiment, std::span<const ParameterSpec> parameters) {
    std::unordered_set<std::string> names, targets;
    for (const auto &parameter : parameters) {
        if (parameter.name.empty() || !names.insert(parameter.name).second ||
            !targets.insert(parameter_key(parameter.target)).second)
            throw std::invalid_argument("duplicate or empty parameter specification");
        read_parameter(experiment, parameter.target);
    }
}
} // namespace ibeamlab::core
