#include <ibeamlab/simulation/simulator.h>
namespace ibeamlab::simulation {
std::vector<Result> Simulator::simulate_batch(std::span<const core::Experiment> inputs,
                                              Progress progress) {
    std::vector<Result> results;
    results.reserve(inputs.size());
    for (const auto &input : inputs) {
        if (stopped())
            break;
        results.emplace_back();
        try {
            simulate(input, results.back());
        } catch (const std::exception &e) {
            results.back().failure = Failure{"", e.what()};
        }
        if (progress)
            progress(results.size(), inputs.size());
    }
    return results;
}
DummySimulator::DummySimulator(std::size_t channels) : channels_(channels) {
    if (channels == 0 || channels > 1000000)
        throw std::invalid_argument("invalid channel count");
}
std::string DummySimulator::provenance() const {
    return "dummy-v1;channels=" + std::to_string(channels_);
}
void DummySimulator::simulate(const core::Experiment &x, Result &output) {
    x.validate();
    output.failure.reset();
    output.spectra.resize(x.setup.measurements.size());
    double inventory = 0;
    for (auto &l : x.sample.layers)
        for (auto &s : l.species) {
            double z = 0;
            for (unsigned char c : s.element)
                z += c;
            inventory += l.thickness * s.concentration * z / 100.;
        }
    for (std::size_t m = 0; m < output.spectra.size(); ++m) {
        if (stopped())
            throw std::runtime_error("simulation cancelled");
        auto &out = output.spectra[m];
        auto &measurement = x.setup.measurements[m];
        out.measurement = measurement.label;
        out.counts.resize(channels_);
        double center = static_cast<double>(channels_) * .45 +
                        std::fmod(measurement.beam.energy, static_cast<double>(channels_) * .1 + 1);
        double width = static_cast<double>(channels_) * .08 + measurement.detector.resolution;
        for (std::size_t i = 0; i < channels_; ++i) {
            double u = (static_cast<double>(i) - center) / width;
            out.counts[i] = inventory * (.01 + std::exp(-.5 * u * u));
        }
    }
}
} // namespace ibeamlab::simulation
