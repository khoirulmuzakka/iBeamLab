#include <ibeamlab/sample.h>

#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace ibeamlab::sample {
namespace {
void finiteNonnegative(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
}
}

void SampleModel::validate() const {
    if (layers.empty()) throw std::invalid_argument("sample must contain at least one layer");
    for (const auto& layer : layers) {
        finiteNonnegative(layer.thickness, "layer thickness");
        finiteNonnegative(layer.roughness, "layer roughness");
        finiteNonnegative(layer.porosityFraction, "layer porosity fraction");
        finiteNonnegative(layer.poreDiameter, "layer pore diameter");
        if (layer.porosityFraction > 1.0) throw std::invalid_argument("porosity fraction exceeds one");
        if (layer.species.empty()) throw std::invalid_argument("layer must contain species");
        double sum = 0.0;
        std::unordered_set<std::string> elements;
        for (const auto& species : layer.species) {
            if (species.element.empty() || !elements.insert(species.element).second)
                throw std::invalid_argument("layer element names must be non-empty and unique");
            finiteNonnegative(species.concentration, "species concentration");
            sum += species.concentration;
            double isotopeSum = 0.0;
            for (const auto& isotope : species.isotopes) {
                if (isotope.massNumber < 0) throw std::invalid_argument("isotope mass number cannot be negative");
                finiteNonnegative(isotope.exactMass, "isotope exact mass");
                finiteNonnegative(isotope.fraction, "isotope fraction");
                isotopeSum += isotope.fraction;
            }
            if (!species.isotopes.empty() && std::abs(isotopeSum - 1.0) > 1e-6)
                throw std::invalid_argument("isotope fractions must sum to one");
        }
        if (std::abs(sum - 1.0) > 1e-6) throw std::invalid_argument("species concentrations must sum to one");
    }
}

void ExperimentalSetup::validate() const {
    if (detectors.empty()) throw std::invalid_argument("experimental setup requires a detector");
    std::unordered_set<std::string> labels;
    for (const auto& detector : detectors) {
        if (detector.label.empty() || !labels.insert(detector.label).second)
            throw std::invalid_argument("detector labels must be non-empty and unique");
        if (detector.beam.particle.empty()) throw std::invalid_argument("beam particle cannot be empty");
        if (!std::isfinite(detector.beam.energy) || detector.beam.energy <= 0.0)
            throw std::invalid_argument("beam energy must be finite and positive");
        finiteNonnegative(detector.beam.spread, "beam spread");
        finiteNonnegative(detector.resolution, "detector resolution");
        finiteNonnegative(detector.particlesSr, "particles per steradian");
        finiteNonnegative(detector.realTime, "real time");
        finiteNonnegative(detector.liveTime, "live time");
        if (!std::isfinite(detector.calibrationLinear) ||
            !std::isfinite(detector.calibrationOffset) ||
            !std::isfinite(detector.calibrationQuadratic))
            throw std::invalid_argument("detector calibration must be finite");
    }
}

} // namespace ibeamlab::sample
