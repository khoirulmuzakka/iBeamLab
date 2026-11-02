#include <algorithm>
#include <ibeamlab/analysis/predictor.h>
#include <stdexcept>
#include <type_traits>
namespace ibeamlab::analysis {
Predictor::Predictor(Plan p) : plan_(std::move(p)) {
    plan_.validate();
    workspaces_.resize(plan_.routes.size());
}
bool Predictor::can_vary(const core::ParameterTarget &target) const {
    std::string measurement;
    std::visit(
        [&](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (!std::is_same_v<T, core::LayerThickness> &&
                          !std::is_same_v<T, core::SpeciesConcentration>)
                measurement = v.measurement;
        },
        target);
    bool found = false;
    for (const auto &r : plan_.routes) {
        if (measurement.empty() || std::find(r.measurements.begin(), r.measurements.end(),
                                             measurement) != r.measurements.end()) {
            found = true;
            if (!r.predictor->can_vary(target))
                return false;
        }
    }
    return found;
}
bool Predictor::can_vary_layer_properties() const {
    return std::all_of(plan_.routes.begin(), plan_.routes.end(),
                       [](const auto &r) { return r.predictor->can_vary_layer_properties(); });
}
void Predictor::predict(std::span<const core::Experiment> source,
                        std::span<simulation::Result> dest) {
    if (source.size() != dest.size())
        throw std::invalid_argument("analysis batch dimensions differ");
    if (source.empty())
        return;
    std::size_t labels = 0;
    for (const auto &r : plan_.routes)
        labels += r.measurements.size();
    for (const auto &e : source) {
        if (e.setup.measurements.size() != labels)
            throw std::invalid_argument("unrouted or missing measurements");
        for (const auto &r : plan_.routes)
            for (const auto &name : r.measurements)
                core::find_measurement(e.setup, name);
    }
    for (std::size_t i = 0; i < source.size(); ++i) {
        dest[i].failure.reset();
        dest[i].elemental_spectra.clear();
        dest[i].spectra.resize(labels);
        for (std::size_t j = 0; j < labels; ++j)
            dest[i].spectra[j].measurement = source[i].setup.measurements[j].label;
    }
    for (std::size_t route = 0; route < plan_.routes.size(); ++route) {
        const auto &r = plan_.routes[route];
        auto &inputs = workspaces_[route].inputs;
        auto &outputs = workspaces_[route].outputs;
        if (inputs.size() < source.size())
            inputs.resize(source.size());
        if (outputs.size() < source.size())
            outputs.resize(source.size());
        for (std::size_t i = 0; i < source.size(); ++i) {
            inputs[i].sample = source[i].sample;
            inputs[i].setup.measurements.resize(r.measurements.size());
            for (std::size_t j = 0; j < r.measurements.size(); ++j)
                inputs[i].setup.measurements[j] =
                    core::find_measurement(source[i].setup, r.measurements[j]);
        }
        r.predictor->predict({inputs.data(), source.size()}, {outputs.data(), source.size()});
        for (std::size_t i = 0; i < source.size(); ++i) {
            if (outputs[i].failure) {
                dest[i].failure = outputs[i].failure;
                continue;
            }
            for (const auto &name : r.measurements) {
                auto found = std::find_if(outputs[i].spectra.begin(), outputs[i].spectra.end(),
                                          [&](const auto &s) { return s.measurement == name; });
                if (found == outputs[i].spectra.end())
                    throw std::runtime_error("routed predictor omitted measurement " + name);
                auto slot = std::find_if(dest[i].spectra.begin(), dest[i].spectra.end(),
                                         [&](const auto &s) { return s.measurement == name; });
                *slot = *found;
                for (const auto &element : outputs[i].elemental_spectra)
                    if (element.measurement == name)
                        dest[i].elemental_spectra.push_back(element);
            }
        }
    }
}
void Predictor::request_stop() noexcept {
    for (auto &r : plan_.routes)
        r.predictor->request_stop();
}
void Predictor::reset_stop() noexcept {
    for (auto &r : plan_.routes)
        r.predictor->reset_stop();
}
bool Predictor::stopped() const noexcept {
    for (auto &r : plan_.routes)
        if (r.predictor->stopped())
            return true;
    return false;
}
} // namespace ibeamlab::analysis
