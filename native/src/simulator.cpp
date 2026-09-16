#include <ibeamlab/simulator.h>

#include <cmath>
#include <stdexcept>

namespace ibeamlab::simulator {

std::vector<SimulationResult> DummySimulator::simulateBatch(
    const std::vector<SimulationInput>& inputs, const SimulationOptions&) {
    resetStop();
    std::vector<SimulationResult> output;
    output.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (stopRequested_.load()) break;
        inputs[index].sample.validate();
        inputs[index].setup.validate();
        SimulationResult result;
        for (const auto& detector : inputs[index].setup.detectors) {
            Spectrum spectrum{detector.label, Counts(channels_)};
            double material = 0.0;
            for (const auto& layer : inputs[index].sample.layers) {
                material += layer.thickness;
                for (const auto& species : layer.species)
                    material += 100.0 * species.concentration;
            }
            for (std::size_t channel = 0; channel < channels_; ++channel) {
                const double center = detector.beam.energy /
                    (detector.calibrationLinear > 0.0 ? detector.calibrationLinear : 1.0);
                const double width = std::max(1.0, detector.resolution + detector.beam.spread);
                const double x = (static_cast<double>(channel) - center) / width;
                spectrum.counts[channel] = static_cast<float>(
                    material * detector.particlesSr * 1e-12 * std::exp(-0.5 * x * x));
            }
            result.spectra.push_back(std::move(spectrum));
        }
        output.push_back(std::move(result));
    }
    return output;
}

} // namespace ibeamlab::simulator
