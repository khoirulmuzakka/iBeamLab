#pragma once

/**
 * @file sample_toml.h
 * @brief Versioned TOML serialization for sample and setup value types.
 *
 * This module provides stable interchange and metadata serialization; it does
 * not perform simulation or sampling.
 */
#include <ibeamlab/sample.h>
#include <string>
namespace ibeamlab::sample {
/** @brief Serializes a sample model to versioned TOML. */
std::string toToml(const SampleModel& sample);
/** @brief Parses and validates a sample model from TOML. */
SampleModel sampleModelFromToml(const std::string& text);
/** @brief Serializes an experimental setup to versioned TOML. */
std::string toToml(const ExperimentalSetup& setup);
/** @brief Parses and validates an experimental setup from TOML. */
ExperimentalSetup experimentalSetupFromToml(const std::string& text);
}
