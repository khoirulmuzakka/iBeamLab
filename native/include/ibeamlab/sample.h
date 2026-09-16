#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ibeamlab::sample {

struct Isotope {
    std::int32_t massNumber{0}; // 0 selects natural abundance.
    double exactMass{0.0};     // Atomic mass units; 0 means unspecified.
    double fraction{1.0};
};

struct Species {
    std::string element;
    double concentration{0.0};
    std::vector<Isotope> isotopes;
};

struct Layer {
    double thickness{0.0}; // 1e15 atoms/cm^2
    double roughness{0.0};
    double porosityFraction{0.0};
    double poreDiameter{0.0};
    std::vector<Species> species;
};

struct SampleModel {
    std::vector<Layer> layers;
    void validate() const;
};

struct Beam {
    std::string particle;
    double energy{0.0}; // keV
    double spread{0.0}; // keV FWHM
};

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

struct ExperimentalSetup {
    std::vector<Detector> detectors;
    void validate() const;
};

} // namespace ibeamlab::sample
