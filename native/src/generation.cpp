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

struct OpenConcentration {
    std::size_t column;
    const ParameterSpec* parameter;
};

void normalizeConcentrations(std::vector<double>& row,
    const std::vector<OpenConcentration>& parameters, double total) {
    constexpr double tolerance = 1e-12;
    double minimumSum = 0.0;
    double maximumSum = 0.0;
    for (const auto& item : parameters) {
        minimumSum += item.parameter->lowerBound;
        maximumSum += item.parameter->upperBound;
    }
    if (total < minimumSum - tolerance || total > maximumSum + tolerance)
        throw std::invalid_argument("concentration bounds cannot sum to the remaining layer fraction");

    std::vector<double> normalized(parameters.size());
    std::vector<double> weights(parameters.size());
    double remaining = total - minimumSum;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const auto& parameter = *parameters[i].parameter;
        normalized[i] = parameter.lowerBound;
        weights[i] = std::max(0.0, row[parameters[i].column] - parameter.lowerBound);
    }

    while (remaining > tolerance) {
        double weightSum = 0.0;
        std::size_t active = 0;
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            if (normalized[i] < parameters[i].parameter->upperBound - tolerance) {
                weightSum += weights[i];
                ++active;
            }
        }
        if (active == 0)
            throw std::invalid_argument("concentration upper bounds leave an unallocated fraction");

        const double amount = remaining;
        double allocated = 0.0;
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            const double capacity = parameters[i].parameter->upperBound - normalized[i];
            if (capacity <= tolerance)
                continue;
            const double share = weightSum > tolerance
                ? amount * weights[i] / weightSum
                : amount / static_cast<double>(active);
            const double addition = std::min(capacity, share);
            normalized[i] += addition;
            allocated += addition;
        }
        if (allocated <= tolerance)
            throw std::invalid_argument("failed to normalize layer concentrations");
        remaining -= allocated;
    }

    for (std::size_t i = 0; i < parameters.size(); ++i)
        row[parameters[i].column] = normalized[i];
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
        double value = std::numeric_limits<double>::quiet_NaN();
        if (parameter.fixedValue) {
            value = *parameter.fixedValue;
        } else if (open < values.size()) {
            value = values[open++];
        }
        if (!std::isfinite(value) || value < parameter.lowerBound || value > parameter.upperBound) throw std::invalid_argument("parameter outside bounds: " + parameter.name);
        apply(result, parameter.target, value);
    }
    if (open != values.size()) throw std::invalid_argument("open parameter row has wrong length");
    result.sample.validate(); result.setup.validate(); return result;
}
std::vector<std::vector<double>> sampleParameters(const GenerationConfig& config, std::size_t count,
    std::uint64_t seed, SamplingMethod method) {
    config.validate();
    if (method != SamplingMethod::Uniform)
        throw std::invalid_argument("unsupported sampling method");

    std::vector<std::vector<OpenConcentration>> concentrationGroups(config.sample.layers.size());
    std::size_t column = 0;
    for (const auto& parameter : config.parameters) {
        if (parameter.fixedValue)
            continue;
        if (const auto* target = std::get_if<SpeciesConcentration>(&parameter.target))
            concentrationGroups.at(target->layer).push_back({column, &parameter});
        ++column;
    }

    std::mt19937_64 random(seed);
    std::vector<std::vector<double>> rows(count);
    for (auto& row : rows) {
        for (const auto& parameter : config.parameters) {
            if (parameter.fixedValue)
                continue;
            std::uniform_real_distribution<double> distribution(parameter.lowerBound,
                                                                  parameter.upperBound);
            row.push_back(distribution(random));
        }

        for (std::size_t layer = 0; layer < concentrationGroups.size(); ++layer) {
            const auto& open = concentrationGroups[layer];
            if (open.empty())
                continue;
            double fixedSum = 0.0;
            for (const auto& species : config.sample.layers[layer].species) {
                const ParameterSpec* concentrationParameter = nullptr;
                for (const auto& parameter : config.parameters) {
                    const auto* target = std::get_if<SpeciesConcentration>(&parameter.target);
                    if (target && target->layer == layer && target->element == species.element) {
                        concentrationParameter = &parameter;
                        break;
                    }
                }
                if (!concentrationParameter)
                    fixedSum += species.concentration;
                else if (concentrationParameter->fixedValue)
                    fixedSum += *concentrationParameter->fixedValue;
            }
            const double remaining = 1.0 - fixedSum;
            if (remaining < -1e-12)
                throw std::invalid_argument("fixed layer concentrations exceed one");
            normalizeConcentrations(row, open, std::max(0.0, remaining));
        }
    }
    return rows;
}

} // namespace ibeamlab::generation
