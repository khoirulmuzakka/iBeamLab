#pragma once

/**
 * @file sample.h
 * @brief Dependency-light physical sample and experimental setup value types.
 *
 * These types form the shared C++/Python data contract used by simulators,
 * dataset generation, and native consumers.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace ibeamlab::sample {

/** @brief Isotopic composition entry for a chemical species. */
struct Isotope {
    std::int32_t massNumber{0}; // 0 selects natural abundance.
    double exactMass{0.0};     // Atomic mass units; 0 means unspecified.
    double fraction{1.0};
};

/** @brief Element and concentration, optionally with explicit isotope fractions. */
struct Species {
    std::string element;
    double concentration{0.0};
    std::vector<Isotope> isotopes;
};

/** @brief Ordered sample layer with areal thickness and composition. */
struct Layer {
    double thickness{0.0}; // 1e15 atoms/cm^2
    double roughness{0.0};
    double porosityFraction{0.0};
    double poreDiameter{0.0};
    std::vector<Species> species;
};

/** @brief Surface-to-depth ordered stack of sample layers. */
struct SampleModel {
    std::vector<Layer> layers;
    /** @brief Validates layer dimensions, species, and normalized compositions. */
    void validate() const;
};

/** @brief Incident particle identity, energy, and energy spread. */
struct Beam {
    std::string particle;
    double energy{0.0}; // keV
    double spread{0.0}; // keV FWHM
};

/** @brief Beam/detector settings for one labeled measurement method. */
struct Detector {
    std::string label;
    Beam beam;
    double calibrationLinear{1.0};
    double particlesSr{0.0};
    double calibrationOffset{0.0};
    double calibrationQuadratic{0.0};
    double resolution{0.0}; // keV FWHM
    double realTime{0.0};   // seconds
    double liveTime{0.0};   // seconds
    std::string info;
};

/** @brief Ordered detector configurations used to produce method spectra. */
struct ExperimentalSetup {
    std::vector<Detector> detectors;
    /** @brief Validates detector labels, beam values, and calibration settings. */
    void validate() const;
};

} // namespace ibeamlab::sample
