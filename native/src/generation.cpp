#include <ibeamlab/generation.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace ibeamlab::generation {
namespace {
const sample::Detector& findDetector(const sample::ExperimentalSetup& setup, const std::string& label) {
    const auto it = std::find_if(setup.detectors.begin(), setup.detectors.end(),
        [&](const auto& detector) { return detector.label == label; });
    if (it == setup.detectors.end()) throw std::invalid_argument("unknown detector: " + label);
    return *it;
}
sample::Detector& findDetector(sample::ExperimentalSetup& setup, const std::string& label) {
    return const_cast<sample::Detector&>(findDetector(std::as_const(setup), label));
}
std::string targetKey(const ParameterTarget& target) {
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
void apply(simulator::SimulationInput& input, const ParameterTarget& target, double value) {
    std::visit([&](const auto& item) {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, LayerThickness>) input.sample.layers.at(item.layer).thickness = value;
        else if constexpr (std::is_same_v<T, SpeciesConcentration>) {
            auto& species = input.sample.layers.at(item.layer).species;
            const auto it = std::find_if(species.begin(), species.end(), [&](const auto& s){ return s.element == item.element; });
            if (it == species.end()) throw std::invalid_argument("unknown species target: " + item.element);
            it->concentration = value;
        } else {
            auto& detector = findDetector(input.setup, item.detector);
            if constexpr (std::is_same_v<T, BeamEnergy>) detector.beam.energy = value;
            else if constexpr (std::is_same_v<T, BeamSpread>) detector.beam.spread = value;
            else if constexpr (std::is_same_v<T, CalibrationLinear>) detector.calibrationLinear = value;
            else if constexpr (std::is_same_v<T, CalibrationOffset>) detector.calibrationOffset = value;
            else if constexpr (std::is_same_v<T, CalibrationQuadratic>) detector.calibrationQuadratic = value;
            else if constexpr (std::is_same_v<T, DetectorResolution>) detector.resolution = value;
            else detector.particlesSr = value;
        }
    }, target);
}
}

void GenerationConfig::validate() const {
    sample.validate(); setup.validate();
    if (methods.empty()) throw std::invalid_argument("generation config requires methods");
    if (methods.size() != setup.detectors.size()) throw std::invalid_argument("methods and detectors must align");
    std::unordered_set<std::string> labels, names, targets;
    for (std::size_t i = 0; i < methods.size(); ++i) {
        const auto& method = methods[i];
        if (method.label.empty() || method.ibaMethod.empty() || method.referenceFile.empty())
            throw std::invalid_argument("method fields cannot be empty");
        if (method.label != setup.detectors[i].label) throw std::invalid_argument("method and detector ordering differs");
        if (!labels.insert(method.label).second) throw std::invalid_argument("method labels must be unique");
    }
    for (const auto& parameter : parameters) {
        if (parameter.name.empty() || !names.insert(parameter.name).second) throw std::invalid_argument("parameter names must be non-empty and unique");
        if (!std::isfinite(parameter.lowerBound) || !std::isfinite(parameter.upperBound) || parameter.lowerBound > parameter.upperBound)
            throw std::invalid_argument("invalid parameter bounds: " + parameter.name);
        if (parameter.fixedValue && (!std::isfinite(*parameter.fixedValue) || *parameter.fixedValue < parameter.lowerBound || *parameter.fixedValue > parameter.upperBound))
            throw std::invalid_argument("fixed parameter is outside bounds: " + parameter.name);
        if (!targets.insert(targetKey(parameter.target)).second) throw std::invalid_argument("parameter targets must be unique");
        simulator::SimulationInput probe{sample, setup}; apply(probe, parameter.target, parameter.fixedValue.value_or(parameter.lowerBound));
    }
}

std::vector<std::string> GenerationConfig::openParameterNames() const {
    std::vector<std::string> result; for (const auto& parameter : parameters) if (!parameter.fixedValue) result.push_back(parameter.name); return result;
}
std::vector<double> GenerationConfig::fixedParameterValues() const {
    std::vector<double> result; for (const auto& parameter : parameters) if (parameter.fixedValue) result.push_back(*parameter.fixedValue); return result;
}
simulator::SimulationInput GenerationConfig::materialize(const std::vector<double>& values) const {
    simulator::SimulationInput result{sample, setup}; std::size_t open = 0;
    for (const auto& parameter : parameters) {
        const double value = parameter.fixedValue.value_or(open < values.size() ? values[open++] : std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(value) || value < parameter.lowerBound || value > parameter.upperBound) throw std::invalid_argument("parameter outside bounds: " + parameter.name);
        apply(result, parameter.target, value);
    }
    if (open != values.size()) throw std::invalid_argument("open parameter row has wrong length");
    result.sample.validate(); result.setup.validate(); return result;
}
std::vector<std::vector<double>> sampleParameters(const GenerationConfig& config, std::size_t count,
    std::uint64_t seed, SamplingMethod method) {
    config.validate(); if (method != SamplingMethod::Uniform) throw std::invalid_argument("unsupported sampling method");
    std::mt19937_64 random(seed); std::vector<std::vector<double>> rows(count);
    for (auto& row : rows) for (const auto& parameter : config.parameters) if (!parameter.fixedValue) {
        std::uniform_real_distribution<double> distribution(parameter.lowerBound, parameter.upperBound);
        row.push_back(distribution(random));
    }
    return rows;
}

} // namespace ibeamlab::generation
