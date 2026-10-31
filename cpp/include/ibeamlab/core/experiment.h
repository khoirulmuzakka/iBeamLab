#pragma once
#include <ibeamlab/core/sample.h>

namespace ibeamlab::core {
/** @brief Incident particle identity, energy, and energy spread. */
struct Beam {
    std::string particle;
    double energy{0.0}; // keV
    double spread{0.0}; // keV FWHM
};

/** @brief Per-detector measurement geometry, in degrees, using SIMNRA conventions.
 * Incidence alpha is measured from the inward surface normal; exit beta from
 * the outward normal. Theta is the angle between incident and outgoing paths.
 * Beta may exceed 90 degrees for transmission. The three angles need not be
 * coplanar. Default values describe normal incidence and exact backscattering.
 */
struct IBEAMLAB_API Geometry {
    double incidenceAngle{0.0};
    double exitAngle{0.0};
    double scatteringAngle{180.0};
    void validate() const;
    bool operator==(const Geometry &) const = default;
};
/** @brief Detector response and polynomial channel-center calibration.
 * Energy in keV is offset + linear*i + quadratic*i*i for zero-based channel i.
 * Channel edges use half-integer indices and must increase over the used range.
 */
struct Detector {
    double calibrationLinear{1.0};
    double calibrationOffset{0.0};
    double calibrationQuadratic{0.0};
    double resolution{0.0}; // keV FWHM
    std::string info;
    Geometry geometry;
};

/** @brief Exposure and acquisition times for one measurement. */
struct Acquisition {
    double particlesSr{0.0}; ///< Incident particle count times detector solid angle (sr).
    double realTime{0.0}; // seconds
    double liveTime{0.0}; // seconds
};

/** @brief Independently configured measurement of the shared sample. */
struct Measurement {
    std::string label;
    Beam beam;
    Detector detector;
    Acquisition acquisition;
};

/** @brief Ordered measurement configurations; beams may differ between measurements. */
struct IBEAMLAB_API ExperimentalSetup {
    std::vector<Measurement> measurements;
    /** @brief Validates measurement labels, beam values, and detector/acquisition settings. */
    void validate() const;
};

/** @brief Shared sample and independently configured measurements of that sample. */
struct Experiment {
    Sample sample;
    ExperimentalSetup setup;
    void validate() const {
        sample.validate();
        setup.validate();
    }
};
} // namespace ibeamlab::core
