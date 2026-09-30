#pragma once

/**
 * @file sample_toml.h
 * @brief Versioned TOML serialization for sample and setup value types.
 *
 * This module provides stable interchange and metadata serialization; it does
 * not perform simulation or sampling.
 */
#include <ibeamlab/sample.h>
#include <ibeamlab/export.h>
#include <string>
namespace ibeamlab::sample {
/** @brief Serializes a sample model to versioned TOML. */
IBEAMLAB_API std::string toToml(const SampleModel& sample);
/** @brief Parses and validates a sample model from TOML. */
IBEAMLAB_API SampleModel sampleModelFromToml(const std::string& text);
/** @brief Serializes an experimental setup to versioned TOML. */
IBEAMLAB_API std::string toToml(const ExperimentalSetup& setup);
/** @brief Parses and validates an experimental setup from TOML. */
IBEAMLAB_API ExperimentalSetup experimentalSetupFromToml(const std::string& text);
}
