#include "onnx_model.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <ibeamlab/forward_model.h>
#include <ibeamlab/parameter.h>
#include <ibeamlab/spectrum_processing.h>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
namespace ibeamlab::inference {
struct ForwardModel::Impl {
    detail::OnnxModel model;
    InferenceOptions options;
    mutable std::mutex warningMutex;
    mutable std::set<std::string> warnings;
    void warn(const std::string &message) const {
        std::lock_guard<std::mutex> lock(warningMutex);
        if (warnings.insert(message).second)
            std::cerr << "iBeamLab warning: " << message << '\n';
    }
    explicit Impl(model::ModelPackage p, InferenceOptions o) : model(std::move(p), o), options(o) {
        if (o.correctionThreads < 1)
            throw std::invalid_argument("correction threads must be positive");
        if (model.metadata().modelType != ibeamlab::model::ModelType::Forward)
            throw std::invalid_argument("ForwardModel requires a forward package");
    }
};
ForwardModel::ForwardModel(const std::filesystem::path &p, InferenceOptions o)
    : ForwardModel(model::ModelPackage::open(p), o) {}
ForwardModel::ForwardModel(model::ModelPackage p, InferenceOptions o)
    : impl_(std::make_unique<Impl>(std::move(p), o)) {}
ForwardModel::~ForwardModel() = default;
ForwardModel::ForwardModel(ForwardModel &&) noexcept = default;
ForwardModel &ForwardModel::operator=(ForwardModel &&) noexcept = default;
const model::ModelMetadata &ForwardModel::metadata() const noexcept {
    return impl_->model.metadata();
}
std::vector<ForwardResult>
ForwardModel::predict(const std::vector<simulator::SimulationInput> &inputs) const {
    preprocessing::Matrix rows;
    rows.reserve(inputs.size());
    const auto &m = impl_->model.metadata().forward;
    for (const auto &i : inputs) {
        i.sample.validate();
        i.setup.validate();
        if (m.bareSpectrumCorrections) {
            for (const auto &ref : m.setupTemplate.detectors) {
                const auto d = std::find_if(i.setup.detectors.begin(), i.setup.detectors.end(),
                                            [&](const auto &v) { return v.label == ref.label; });
                if (d == i.setup.detectors.end())
                    throw std::invalid_argument("missing detector: " + ref.label);
                if (d->beam.particle != ref.beam.particle || d->beam.energy != ref.beam.energy ||
                    d->beam.spread != ref.beam.spread || d->resolution != ref.resolution)
                    impl_->warn("beam or detector resolution differs from training setup for " +
                                ref.label + "; prediction uses training values");
            }
            if (i.sample.layers.size() > m.sampleTemplate.layers.size())
                throw std::invalid_argument("sample exceeds trained layer layout");
            for (std::size_t l = 0; l < i.sample.layers.size(); ++l) {
                for (const auto &species : i.sample.layers[l].species) {
                    const auto &supported = m.sampleTemplate.layers[l].species;
                    if (std::none_of(supported.begin(), supported.end(),
                                     [&](const auto &v) { return v.element == species.element; }))
                        impl_->warn("unsupported element " + species.element + " in layer " +
                                    std::to_string(l) + "; ignored by surrogate");
                }
            }
        }
        std::vector<float> row;
        row.reserve(m.inputParameters.size());
        for (const auto &p : m.inputParameters) {
            double v = 0;
            if (m.bareSpectrumCorrections &&
                std::holds_alternative<parameter::SpeciesConcentration>(p.target)) {
                const auto &t = std::get<parameter::SpeciesConcentration>(p.target);
                if (t.layer < i.sample.layers.size()) {
                    const auto &ss = i.sample.layers[t.layer].species;
                    const auto it = std::find_if(ss.begin(), ss.end(), [&](const auto &x) {
                        return x.element == t.element;
                    });
                    if (it != ss.end())
                        v = it->concentration;
                }
            } else if (m.bareSpectrumCorrections &&
                       std::holds_alternative<parameter::LayerThickness>(p.target) &&
                       std::get<parameter::LayerThickness>(p.target).layer >=
                           i.sample.layers.size()) {
                v = 0;
            } else
                v = parameter::read(i, p.target);
            if (!std::isfinite(v) || v < p.lowerBound || v > p.upperBound)
                throw std::invalid_argument("forward parameter outside bounds: " + p.name);
            row.push_back(static_cast<float>(v));
        }
        rows.push_back(std::move(row));
    }
    const auto values = impl_->model.run(rows);
    std::vector<ForwardResult> result(values.size());
    // Never let exceptions escape an OpenMP region. Each row owns its output/error.
    std::vector<std::exception_ptr> errors(values.size());
    [[maybe_unused]] const int workers = static_cast<int>(std::max<std::size_t>(
        1, std::min<std::size_t>(impl_->options.correctionThreads, values.size())));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)                                                          \
    num_threads(workers) if (workers > 1 && m.bareSpectrumCorrections)
#endif
    for (std::int64_t rowIndex = 0; rowIndex < static_cast<std::int64_t>(values.size());
         ++rowIndex) {
        const auto r = static_cast<std::size_t>(rowIndex);
        try {
            std::size_t offset = 0;
            for (const auto &s : m.outputSpectra) {
                simulator::Spectrum spectrum{s.label, {}};
                spectrum.counts.assign(values[r].begin() + static_cast<std::ptrdiff_t>(offset),
                                       values[r].begin() +
                                           static_cast<std::ptrdiff_t>(offset + s.length));
                if (m.bareSpectrumCorrections) {
                    const auto &refs = m.setupTemplate.detectors;
                    const auto &ds = inputs[r].setup.detectors;
                    const auto &ref = *std::find_if(refs.begin(), refs.end(), [&](const auto &v) {
                        return v.label == s.label;
                    });
                    const auto &d = *std::find_if(
                        ds.begin(), ds.end(), [&](const auto &v) { return v.label == s.label; });
                    auto edges = [](const auto &detector, std::size_t count) {
                        std::vector<double> e(count + 1);
                        for (std::size_t c = 0; c <= count; ++c) {
                            const double x = static_cast<double>(c);
                            e[c] = detector.calibrationOffset + detector.calibrationLinear * x +
                                   detector.calibrationQuadratic * x * x;
                            if (!std::isfinite(e[c]) || (c && e[c] <= e[c - 1]))
                                throw std::invalid_argument(
                                    "calibration edges must be finite and increasing");
                        }
                        return e;
                    };
                    const auto oldEdges = edges(ref, s.length);
                    double channel;
                    const bool sameCalibration = d.calibrationLinear == ref.calibrationLinear &&
                                                 d.calibrationOffset == ref.calibrationOffset &&
                                                 d.calibrationQuadratic == ref.calibrationQuadratic;
                    if (sameCalibration)
                        channel = static_cast<double>(s.length);
                    else if (d.calibrationQuadratic == 0) {
                        if (d.calibrationLinear <= 0)
                            throw std::invalid_argument("linear calibration must be positive");
                        channel = (oldEdges.back() - d.calibrationOffset) / d.calibrationLinear;
                    } else {
                        const double disc =
                            d.calibrationLinear * d.calibrationLinear +
                            4 * d.calibrationQuadratic * (oldEdges.back() - d.calibrationOffset);
                        if (disc < 0)
                            throw std::invalid_argument(
                                "calibration does not cover reference energy range");
                        channel = 2 * (oldEdges.back() - d.calibrationOffset) /
                                  (d.calibrationLinear + std::sqrt(disc));
                    }
                    if (!std::isfinite(channel) || channel <= 0 || channel > 10000000)
                        throw std::invalid_argument("invalid corrected channel count");
                    const auto newEdges = edges(d, static_cast<std::size_t>(std::ceil(channel)));
                    auto counts = ibeamlab::spectrum::rebin(
                        oldEdges, newEdges,
                        std::vector<double>(spectrum.counts.begin(), spectrum.counts.end()));
                    for (auto &v : counts)
                        v = std::max(0.0, v * d.particlesSr / ref.particlesSr);
                    if (impl_->options.applyPileupOnInference.value_or(m.applyPileupOnInference))
                        counts = ibeamlab::spectrum::pileup(counts, d.realTime, d.liveTime,
                                                            m.pileupFudgeFactorSeconds);
                    spectrum.counts.resize(counts.size());
                    std::transform(counts.begin(), counts.end(), spectrum.counts.begin(),
                                   [](double v) { return static_cast<float>(v); });
                }
                result[r].spectra.push_back(std::move(spectrum));
                offset += s.length;
            }
        } catch (...) {
            errors[r] = std::current_exception();
        }
    }
    for (const auto &error : errors)
        if (error)
            std::rethrow_exception(error);
    return result;
}
} // namespace ibeamlab::inference
