#pragma once

/**
 * @file generation.h
 * @brief Maps named parameter rows onto concrete simulation inputs.
 *
 * This module defines parameter targets and validates/materializes externally
 * sampled values. Distribution design and random sampling belong to Python.
 */

#include <ibeamlab/parameter.h>
#include <ibeamlab/sample.h>
#include <ibeamlab/simulator.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ibeamlab::generation {

/** @brief Targets the areal thickness of a zero-based sample layer. */
using parameter::LayerThickness;
/** @brief Targets one element concentration in a zero-based sample layer. */
using parameter::SpeciesConcentration;
/** @brief Targets incident beam energy for a labeled detector. */
using parameter::BeamEnergy;
/** @brief Targets incident beam energy spread for a labeled detector. */
using parameter::BeamSpread;
/** @brief Targets the linear calibration coefficient. */
using parameter::CalibrationLinear;
/** @brief Targets the calibration offset. */
using parameter::CalibrationOffset;
/** @brief Targets the quadratic calibration coefficient. */
using parameter::CalibrationQuadratic;
/** @brief Targets detector energy resolution. */
using parameter::DetectorResolution;
/** @brief Targets the particles-times-solid-angle normalization. */
using parameter::ParticlesSr;
/** @brief Type-safe destination for a generated physical value. */
using ParameterTarget = parameter::Target;

/** @brief Declares one fixed or externally sampled physical parameter. */
struct ParameterSpec {
    std::string name;
    ParameterTarget target;
    double lowerBound{};
    double upperBound{};
    std::optional<double> fixedValue;
    std::string unit;
};

/** @brief Associates an output label with an IBA method and reference file. */
struct MethodConfig {
    std::string label;
    std::string ibaMethod;
    std::filesystem::path referenceFile;
};

/**
 * @brief Fixed simulation schema used to interpret parameter rows.
 *
 * The baseline sample/setup contain all layers, species, and detectors. Each
 * ParameterSpec replaces one baseline value with either a fixed value or a
 * value from an externally supplied row.
 */
struct GenerationConfig {
    sample::SampleModel sample;
    sample::ExperimentalSetup setup;
    std::vector<MethodConfig> methods;
    std::vector<ParameterSpec> parameters;

    /** @brief Validates the complete schema and every parameter target. */
    void validate() const;
    /** @brief Returns open parameter names in required row-column order. */
    std::vector<std::string> openParameterNames() const;
    /** @brief Returns fixed parameter values in declaration order. */
    std::vector<double> fixedParameterValues() const;
    /** @brief Builds and validates one concrete simulator input. */
    simulator::SimulationInput materialize(const std::vector<double>& openValues) const;
};

/** @brief Serializes the reproducible generation schema as TOML. */
std::string generationConfigToToml(const GenerationConfig& config);

} // namespace ibeamlab::generation
