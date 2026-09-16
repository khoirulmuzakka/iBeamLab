#pragma once

#include <ibeamlab/sample.h>
#include <ibeamlab/simulator.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ibeamlab::generation {

struct LayerThickness { std::size_t layer{}; };
struct SpeciesConcentration { std::size_t layer{}; std::string element; };
struct BeamEnergy { std::string detector; };
struct BeamSpread { std::string detector; };
struct CalibrationLinear { std::string detector; };
struct CalibrationOffset { std::string detector; };
struct CalibrationQuadratic { std::string detector; };
struct DetectorResolution { std::string detector; };
struct ParticlesSr { std::string detector; };
using ParameterTarget = std::variant<LayerThickness, SpeciesConcentration, BeamEnergy,
    BeamSpread, CalibrationLinear, CalibrationOffset, CalibrationQuadratic,
    DetectorResolution, ParticlesSr>;

struct ParameterSpec {
    std::string name;
    ParameterTarget target;
    double lowerBound{};
    double upperBound{};
    std::optional<double> fixedValue;
    std::string unit;
};

struct MethodConfig {
    std::string label;
    std::string ibaMethod;
    std::filesystem::path referenceFile;
};

struct GenerationConfig {
    sample::SampleModel sample;
    sample::ExperimentalSetup setup;
    std::vector<MethodConfig> methods;
    std::vector<ParameterSpec> parameters;

    void validate() const;
    std::vector<std::string> openParameterNames() const;
    std::vector<double> fixedParameterValues() const;
    simulator::SimulationInput materialize(const std::vector<double>& openValues) const;
};

enum class SamplingMethod { Uniform };
std::vector<std::vector<double>> sampleParameters(const GenerationConfig& config,
    std::size_t count, std::uint64_t seed, SamplingMethod method = SamplingMethod::Uniform);
std::string generationConfigToToml(const GenerationConfig& config);

} // namespace ibeamlab::generation
