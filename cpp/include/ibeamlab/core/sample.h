#pragma once

/**
 * @file sample.h
 * @brief Dependency-light physical sample and experimental setup value types.
 *
 * These types form the shared scientific data contract used by simulators,
 * dataset generation, and native consumers.
 */

#include <cstdint>
#include <ibeamlab/export.h>
#include <string>
#include <vector>

namespace ibeamlab::core {

/** @brief Isotopic composition entry for a chemical species. */
struct Isotope {
    std::int32_t massNumber{0}; // 0 selects natural abundance.
    double exactMass{0.0};      // Atomic mass units; 0 means unspecified.
    double fraction{1.0};
};

/** @brief Element and atomic concentration fraction, optionally with isotope fractions.
 * Species fractions in a layer sum to one. An empty isotope list uses the
 * backend's natural composition; explicit isotope fractions also sum to one.
 */
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
struct IBEAMLAB_API Sample {
    std::vector<Layer> layers;
    /** @brief Validates layer dimensions, species, and normalized compositions. */
    void validate() const;
};

} // namespace ibeamlab::core
