#include <ibeamlab/core/experiment.h>

#include <ibeamlab/core/isotopes.h>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace ibeamlab::core {
namespace {
void finiteNonnegative(double value, const char *name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
    }
}
} // namespace

void Sample::validate() const {
    if (layers.empty())
        throw std::invalid_argument("sample must contain at least one layer");
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto &layer = layers[index];
        try {
            finiteNonnegative(layer.thickness, "layer thickness");
            finiteNonnegative(layer.roughness, "layer roughness");
            finiteNonnegative(layer.porosityFraction, "layer porosity fraction");
            finiteNonnegative(layer.poreDiameter, "layer pore diameter");
            if (layer.porosityFraction > 1.0)
                throw std::invalid_argument("porosity fraction exceeds one");
            if (layer.species.empty())
                throw std::invalid_argument("layer must contain species");
            double sum = 0.0;
            std::unordered_set<std::string> elements;
            for (const auto &species : layer.species) {
                if (species.element.empty() || !elements.insert(species.element).second)
                    throw std::invalid_argument("layer element names must be non-empty and unique");
                finiteNonnegative(species.concentration, "species concentration");
                sum += species.concentration;
                validate_isotopes(species.element, species.isotopes);
            }
            if (!std::isfinite(sum) || std::abs(sum - 1.0) > 1e-6)
                throw std::invalid_argument("species concentrations must sum to one");
        } catch (const std::invalid_argument &error) {
            throw std::invalid_argument("sample.layers[" + std::to_string(index) +
                                        "]: " + error.what());
        }
    }
}

void ExperimentalSetup::validate() const {
    if (measurements.empty())
        throw std::invalid_argument("experimental setup requires a measurement");
    std::unordered_set<std::string> labels;
    for (std::size_t index = 0; index < measurements.size(); ++index) {
        const auto &measurement = measurements[index];
        try {
            if (measurement.label.empty() || !labels.insert(measurement.label).second)
                throw std::invalid_argument("measurement labels must be non-empty and unique");
            if (measurement.beam.particle.empty())
                throw std::invalid_argument("beam particle cannot be empty");
            if (!std::isfinite(measurement.beam.energy) || measurement.beam.energy <= 0.0)
                throw std::invalid_argument("beam energy must be finite and positive");
            finiteNonnegative(measurement.beam.spread, "beam spread");
            finiteNonnegative(measurement.detector.resolution, "detector resolution");
            finiteNonnegative(measurement.acquisition.particlesSr, "particles per steradian");
            finiteNonnegative(measurement.acquisition.realTime, "real time");
            finiteNonnegative(measurement.acquisition.liveTime, "live time");
            if (!std::isfinite(measurement.detector.calibrationLinear) ||
                !std::isfinite(measurement.detector.calibrationOffset) ||
                !std::isfinite(measurement.detector.calibrationQuadratic))
                throw std::invalid_argument("detector calibration must be finite");
        } catch (const std::invalid_argument &error) {
            throw std::invalid_argument("setup.measurements[" + std::to_string(index) +
                                        "]: " + error.what());
        }
    }
}

} // namespace ibeamlab::core
