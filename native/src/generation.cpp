#include <ibeamlab/generation.h>
#include <ibeamlab/sample_toml.h>

#include <toml++/toml.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace ibeamlab::generation {
namespace {
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
        if (!targets.insert(parameter::key(parameter.target)).second) throw std::invalid_argument("parameter targets must be unique");
        simulator::SimulationInput probe{sample, setup}; parameter::write(probe, parameter.target, parameter.fixedValue.value_or(parameter.lowerBound));
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
        parameter::write(result, parameter.target, value);
    }
    if (open != values.size()) throw std::invalid_argument("open parameter row has wrong length");
    result.sample.validate(); result.setup.validate(); return result;
}
std::string generationConfigToToml(const GenerationConfig& config) {
    config.validate();
    auto sampleTable = toml::parse(sample::toToml(config.sample));
    sampleTable.erase("format");
    sampleTable.erase("format_version");
    auto setupTable = toml::parse(sample::toToml(config.setup));
    setupTable.erase("format");
    setupTable.erase("format_version");

    toml::array methods;
    for (const auto& method : config.methods)
        methods.push_back(toml::table{{"label", method.label}, {"iba_method", method.ibaMethod},
            {"reference_file", method.referenceFile.generic_string()}});

    toml::array parameters;
    for (const auto& parameter : config.parameters) {
        toml::table target;
        std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, LayerThickness>) {
                target.insert("type", "layer_thickness");
                target.insert("layer", static_cast<std::int64_t>(value.layer));
            } else if constexpr (std::is_same_v<T, SpeciesConcentration>) {
                target.insert("type", "species_concentration");
                target.insert("layer", static_cast<std::int64_t>(value.layer));
                target.insert("element", value.element);
            } else {
                if constexpr (std::is_same_v<T, BeamEnergy>) target.insert("type", "beam_energy");
                else if constexpr (std::is_same_v<T, BeamSpread>) target.insert("type", "beam_spread");
                else if constexpr (std::is_same_v<T, CalibrationLinear>) target.insert("type", "calibration_linear");
                else if constexpr (std::is_same_v<T, CalibrationOffset>) target.insert("type", "calibration_offset");
                else if constexpr (std::is_same_v<T, CalibrationQuadratic>) target.insert("type", "calibration_quadratic");
                else if constexpr (std::is_same_v<T, DetectorResolution>) target.insert("type", "detector_resolution");
                else target.insert("type", "particles_sr");
                target.insert("detector", value.detector);
            }
        }, parameter.target);
        toml::table item{{"name", parameter.name}, {"lower_bound", parameter.lowerBound},
            {"upper_bound", parameter.upperBound}, {"unit", parameter.unit},
            {"open", !parameter.fixedValue.has_value()}, {"target", std::move(target)}};
        if (parameter.fixedValue)
            item.insert("fixed_value", *parameter.fixedValue);
        parameters.push_back(std::move(item));
    }

    toml::table root{{"format_version", 1}, {"sample", std::move(sampleTable)},
        {"setup", std::move(setupTable)}, {"methods", std::move(methods)},
        {"parameters", std::move(parameters)},
        {"concentration_sampling", "normalized_uniform_per_layer"}};
    std::ostringstream output;
    output << root;
    return output.str();
}

} // namespace ibeamlab::generation
