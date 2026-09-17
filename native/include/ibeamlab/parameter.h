#pragma once

/**
 * @file parameter.h
 * @brief Physical parameter locations shared by generation and ONNX models.
 */

#include <ibeamlab/simulator.h>

#include <cstddef>
#include <string>
#include <variant>

namespace ibeamlab::parameter {

struct LayerThickness { std::size_t layer{}; };
struct SpeciesConcentration { std::size_t layer{}; std::string element; };
struct BeamEnergy { std::string detector; };
struct BeamSpread { std::string detector; };
struct CalibrationLinear { std::string detector; };
struct CalibrationOffset { std::string detector; };
struct CalibrationQuadratic { std::string detector; };
struct DetectorResolution { std::string detector; };
struct ParticlesSr { std::string detector; };

using Target = std::variant<LayerThickness, SpeciesConcentration, BeamEnergy, BeamSpread,
    CalibrationLinear, CalibrationOffset, CalibrationQuadratic, DetectorResolution, ParticlesSr>;

/** @brief Returns a stable identity used for duplicate-target validation. */
std::string key(const Target& target);

/** @brief Reads one physical value from a fully materialized input. */
double read(const simulator::SimulationInput& input, const Target& target);

/** @brief Replaces one physical value in a simulation input. */
void write(simulator::SimulationInput& input, const Target& target, double value);

} // namespace ibeamlab::parameter
