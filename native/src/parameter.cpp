#include <ibeamlab/parameter.h>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ibeamlab::parameter {
namespace {
const sample::Detector& detector(const sample::ExperimentalSetup& setup, const std::string& label) {
    const auto found = std::find_if(setup.detectors.begin(), setup.detectors.end(),
        [&](const auto& value) { return value.label == label; });
    if (found == setup.detectors.end()) throw std::invalid_argument("unknown detector: " + label);
    return *found;
}
sample::Detector& detector(sample::ExperimentalSetup& setup, const std::string& label) {
    return const_cast<sample::Detector&>(detector(std::as_const(setup), label));
}
const sample::Species& species(const simulator::SimulationInput& input,
                               const SpeciesConcentration& target) {
    const auto& values = input.sample.layers.at(target.layer).species;
    const auto found = std::find_if(values.begin(), values.end(),
        [&](const auto& value) { return value.element == target.element; });
    if (found == values.end()) throw std::invalid_argument("unknown species target: " + target.element);
    return *found;
}
} // namespace

std::string key(const Target& target) {
    return std::visit([](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, LayerThickness>) return "layer.thickness." + std::to_string(value.layer);
        else if constexpr (std::is_same_v<T, SpeciesConcentration>) return "layer.species." + std::to_string(value.layer) + "." + value.element;
        else if constexpr (std::is_same_v<T, BeamEnergy>) return "beam.energy." + value.detector;
        else if constexpr (std::is_same_v<T, BeamSpread>) return "beam.spread." + value.detector;
        else if constexpr (std::is_same_v<T, CalibrationLinear>) return "calibration.linear." + value.detector;
        else if constexpr (std::is_same_v<T, CalibrationOffset>) return "calibration.offset." + value.detector;
        else if constexpr (std::is_same_v<T, CalibrationQuadratic>) return "calibration.quadratic." + value.detector;
        else if constexpr (std::is_same_v<T, DetectorResolution>) return "detector.resolution." + value.detector;
        else return "detector.particles_sr." + value.detector;
    }, target);
}

double read(const simulator::SimulationInput& input, const Target& target) {
    return std::visit([&](const auto& value) -> double {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, LayerThickness>) return input.sample.layers.at(value.layer).thickness;
        else if constexpr (std::is_same_v<T, SpeciesConcentration>) return species(input, value).concentration;
        else {
            const auto& item = detector(input.setup, value.detector);
            if constexpr (std::is_same_v<T, BeamEnergy>) return item.beam.energy;
            else if constexpr (std::is_same_v<T, BeamSpread>) return item.beam.spread;
            else if constexpr (std::is_same_v<T, CalibrationLinear>) return item.calibrationLinear;
            else if constexpr (std::is_same_v<T, CalibrationOffset>) return item.calibrationOffset;
            else if constexpr (std::is_same_v<T, CalibrationQuadratic>) return item.calibrationQuadratic;
            else if constexpr (std::is_same_v<T, DetectorResolution>) return item.resolution;
            else return item.particlesSr;
        }
    }, target);
}

void write(simulator::SimulationInput& input, const Target& target, double value) {
    std::visit([&](const auto& location) {
        using T = std::decay_t<decltype(location)>;
        if constexpr (std::is_same_v<T, LayerThickness>) input.sample.layers.at(location.layer).thickness = value;
        else if constexpr (std::is_same_v<T, SpeciesConcentration>) {
            auto& values = input.sample.layers.at(location.layer).species;
            const auto found = std::find_if(values.begin(), values.end(),
                [&](const auto& item) { return item.element == location.element; });
            if (found == values.end()) throw std::invalid_argument("unknown species target: " + location.element);
            found->concentration = value;
        } else {
            auto& item = detector(input.setup, location.detector);
            if constexpr (std::is_same_v<T, BeamEnergy>) item.beam.energy = value;
            else if constexpr (std::is_same_v<T, BeamSpread>) item.beam.spread = value;
            else if constexpr (std::is_same_v<T, CalibrationLinear>) item.calibrationLinear = value;
            else if constexpr (std::is_same_v<T, CalibrationOffset>) item.calibrationOffset = value;
            else if constexpr (std::is_same_v<T, CalibrationQuadratic>) item.calibrationQuadratic = value;
            else if constexpr (std::is_same_v<T, DetectorResolution>) item.resolution = value;
            else item.particlesSr = value;
        }
    }, target);
}

} // namespace ibeamlab::parameter
