#pragma once

/**
 * @file parameter.h
 * @brief Physical parameter locations shared by generation, fitting, and inference.
 */

#include <ibeamlab/core/experiment.h>
#include <ibeamlab/export.h>
#include <span>

#include <cstddef>
#include <string>
#include <variant>

namespace ibeamlab::core {

/** @brief Areal thickness of a zero-based layer, in 1e15 atoms/cm2. */
struct LayerThickness {
    std::size_t layer{};
};
/** @brief Atomic fraction of the named element in a zero-based layer. */
struct SpeciesConcentration {
    std::size_t layer{};
    std::string element;
};
/** @brief Incident beam energy in keV, selected by measurement label. */
struct BeamEnergy {
    std::string measurement;
};
/** @brief Beam energy spread in keV FWHM, selected by measurement label. */
struct BeamSpread {
    std::string measurement;
};
/** @brief Linear calibration coefficient in keV/channel for a labelled measurement. */
struct CalibrationLinear {
    std::string measurement;
};
/** @brief Calibration offset in keV for a labelled measurement. */
struct CalibrationOffset {
    std::string measurement;
};
/** @brief Quadratic calibration coefficient in keV/channel squared. */
struct CalibrationQuadratic {
    std::string measurement;
};
/** @brief Detector energy resolution in keV FWHM for a labelled measurement. */
struct DetectorResolution {
    std::string measurement;
};
/** @brief Incident particle count times solid angle for a labelled measurement. */
struct ParticlesSr {
    std::string measurement;
};

using ParameterTarget =
    std::variant<LayerThickness, SpeciesConcentration, BeamEnergy, BeamSpread, CalibrationLinear,
                 CalibrationOffset, CalibrationQuadratic, DetectorResolution, ParticlesSr>;

/** @brief Returns a stable identity used for duplicate-target validation. */
IBEAMLAB_API std::string parameter_key(const ParameterTarget &target);

/** @brief Reads one physical value from a fully materialized input. */
IBEAMLAB_API double read_parameter(const Experiment &input, const ParameterTarget &target);

/** @brief Assigns a finite value without allocating experiment copies.
 * May temporarily invalidate physical constraints, including composition sums.
 * Validate the experiment after the complete set of writes.
 */
IBEAMLAB_API void write_parameter(Experiment &input, const ParameterTarget &target,
                                       double value);

/** @brief Ordered scientific input column; values use core's physical units. */
struct ParameterSpec {
    std::string name;
    ParameterTarget target;
    std::string unit;
};
IBEAMLAB_API void validate_parameters(const Experiment &, std::span<const ParameterSpec>);

/** @brief Inclusive finite interval in the target parameter's physical units. */
struct ParameterBounds {
    double minimum;
    double maximum;
    IBEAMLAB_API void validate() const;
    IBEAMLAB_API bool contains(double value) const;
};
/** @brief One physical assignment; apply() validates the complete set atomically. */
struct ParameterUpdate {
    ParameterTarget target;
    double value;
};
// Convenience operation: copies the experiment, validates all updates, then commits.
// Allocates storage for the copy; failed updates leave the original unchanged.
IBEAMLAB_API void apply(Experiment &experiment, std::span<const ParameterUpdate> updates);
} // namespace ibeamlab::core
