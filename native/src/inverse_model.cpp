#include "onnx_model.h"
#include <algorithm>
#include <cmath>
#include <ibeamlab/inverse_model.h>
#include <ibeamlab/spectrum_processing.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
namespace ibeamlab::inference {
struct InverseModel::Impl {
    detail::OnnxModel model;
    mutable std::mutex warningMutex;
    mutable std::set<std::string> warnings;
    void warn(const std::string &message) const {
        std::lock_guard<std::mutex> lock(warningMutex);
        if (warnings.insert(message).second)
            std::cerr << "iBeamLab warning: " << message << '\n';
    }
    explicit Impl(model::ModelPackage p, InferenceOptions o) : model(std::move(p), o) {
        if (model.metadata().modelType != ibeamlab::model::ModelType::Inverse)
            throw std::invalid_argument("InverseModel requires an inverse package");
    }
};
InverseModel::InverseModel(const std::filesystem::path &p, InferenceOptions o)
    : InverseModel(model::ModelPackage::open(p), o) {}
InverseModel::InverseModel(model::ModelPackage p, InferenceOptions o)
    : impl_(std::make_unique<Impl>(std::move(p), o)) {}
InverseModel::~InverseModel() = default;
InverseModel::InverseModel(InverseModel &&) noexcept = default;
InverseModel &InverseModel::operator=(InverseModel &&) noexcept = default;
const model::ModelMetadata &InverseModel::metadata() const noexcept {
    return impl_->model.metadata();
}
std::vector<InverseResult> InverseModel::predict(const std::vector<InverseInput> &batch) const {
    const auto &m = metadata().inverse;
    std::vector<std::vector<simulator::Spectrum>> prepared;
    prepared.reserve(batch.size());
    auto edges = [](const sample::Detector &detector, std::size_t count) {
        std::vector<double> result(count + 1);
        for (std::size_t i = 0; i <= count; ++i) {
            const double channel = static_cast<double>(i);
            result[i] = detector.calibrationOffset + detector.calibrationLinear * channel +
                        detector.calibrationQuadratic * channel * channel;
            if (!std::isfinite(result[i]) || (i && result[i] <= result[i - 1]))
                throw std::invalid_argument(
                    "inverse calibration edges must be finite and increasing: " + detector.label);
        }
        return result;
    };
    for (const auto &input : batch) {
        input.setup.validate();
        std::unordered_map<std::string, const simulator::Spectrum *> spectra;
        for (const auto &s : input.spectra)
            if (!spectra.emplace(s.label, &s).second)
                throw std::invalid_argument("duplicate spectrum: " + s.label);
        if (spectra.size() != m.inputSpectra.size())
            throw std::invalid_argument("inverse input must contain exactly the packaged spectra");
        std::vector<simulator::Spectrum> row;
        for (const auto &spec : m.inputSpectra) {
            const auto found = spectra.find(spec.label);
            if (found == spectra.end() || found->second->counts.empty())
                throw std::invalid_argument("missing or empty spectrum: " + spec.label);
            const auto measured =
                std::find_if(input.setup.detectors.begin(), input.setup.detectors.end(),
                             [&](const auto &d) { return d.label == spec.label; });
            if (measured == input.setup.detectors.end())
                throw std::invalid_argument("missing experimental detector: " + spec.label);
            const auto &reference =
                *std::find_if(m.setupTemplate.detectors.begin(), m.setupTemplate.detectors.end(),
                              [&](const auto &d) { return d.label == spec.label; });
            if (measured->particlesSr <= 0 || reference.particlesSr <= 0)
                throw std::invalid_argument("inverse exposure correction requires positive "
                                            "measured and reference ParticlesSr: " +
                                            spec.label);
            if (measured->beam.particle != reference.beam.particle ||
                measured->beam.energy != reference.beam.energy ||
                measured->beam.spread != reference.beam.spread ||
                measured->resolution != reference.resolution)
                impl_->warn("beam or detector resolution differs from inverse training setup for " +
                            spec.label + "; no correction is available");
            std::vector<double> counts(found->second->counts.begin(), found->second->counts.end());
            for (double value : counts)
                if (!std::isfinite(value) || value < 0)
                    throw std::invalid_argument("inverse counts must be finite and nonnegative: " +
                                                spec.label);
            if (m.needPileupSubtraction && !input.pileupAlreadyRemoved)
                counts = spectrum::removePileup(counts, measured->realTime, measured->liveTime,
                                                m.pileupFudgeFactorSeconds);
            const auto oldEdges = edges(*measured, counts.size());
            const auto newEdges = edges(reference, spec.length);
            const double tolerance =
                1e-10 * std::max({1.0, std::abs(newEdges.front()), std::abs(newEdges.back())});
            if (oldEdges.front() > newEdges.front() + tolerance ||
                oldEdges.back() < newEdges.back() - tolerance)
                throw std::invalid_argument(
                    "experimental spectrum does not cover the training energy grid: " + spec.label);
            counts = spectrum::rebin(oldEdges, newEdges, counts);
            const double scale = reference.particlesSr / measured->particlesSr;
            simulator::Spectrum corrected{spec.label, {}};
            corrected.counts.reserve(spec.length);
            for (double value : counts) {
                value *= scale;
                if (!std::isfinite(value) || value < 0 || value > std::numeric_limits<float>::max())
                    throw std::invalid_argument("inverse corrected counts are not representable: " +
                                                spec.label);
                corrected.counts.push_back(static_cast<float>(value));
            }
            row.push_back(std::move(corrected));
        }
        prepared.push_back(std::move(row));
    }
    return predictPrepared(prepared);
}

std::vector<InverseResult>
InverseModel::predictPrepared(const std::vector<std::vector<simulator::Spectrum>> &batch) const {
    const auto &m = impl_->model.metadata().inverse;
    preprocessing::Matrix rows;
    for (const auto &sample : batch) {
        std::unordered_map<std::string, const simulator::Spectrum *> byLabel;
        for (const auto &s : sample)
            if (!byLabel.emplace(s.label, &s).second)
                throw std::invalid_argument("duplicate spectrum: " + s.label);
        if (byLabel.size() != m.inputSpectra.size())
            throw std::invalid_argument("inverse input must contain exactly the packaged spectra");
        std::vector<float> row;
        for (const auto &spec : m.inputSpectra) {
            const auto found = byLabel.find(spec.label);
            if (found == byLabel.end() || found->second->counts.size() != spec.length)
                throw std::invalid_argument("missing or incorrectly sized spectrum: " + spec.label);
            for (float value : found->second->counts)
                if (!std::isfinite(value) || value < 0)
                    throw std::invalid_argument("inverse counts must be finite and nonnegative: " +
                                                spec.label);
            row.insert(row.end(), found->second->counts.begin(), found->second->counts.end());
        }
        rows.push_back(std::move(row));
    }
    const auto prediction = impl_->model.runPrediction(rows);
    const auto &values = prediction.values;
    std::vector<InverseResult> results;
    results.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto &row = values[index];
        EdpMap edp{m.outputEdp.elements, m.outputEdp.unit, {}};
        edp.uncertaintyPredicted = m.uncertaintyPredicted;
        bool padding = false;
        for (std::size_t layer = 0; layer < m.outputEdp.maxLayers; ++layer) {
            auto first = row.begin() + layer * edp.elements.size();
            std::vector<float> densities(first, first + edp.elements.size());
            for (float value : densities)
                if (!std::isfinite(value) || value < 0)
                    throw std::runtime_error("inverse EDP must be finite and nonnegative");
            const bool zero = std::all_of(densities.begin(), densities.end(),
                                          [](float value) { return value == 0; });
            if (padding && !zero)
                throw std::runtime_error("inverse EDP zero layers must be trailing padding");
            padding = padding || zero;
            std::vector<float> probability(edp.elements.size()), deviation(edp.elements.size(), 0);
            const auto offset = layer * edp.elements.size();
            for (std::size_t j = 0; j < edp.elements.size(); ++j) {
                probability[j] = prediction.presenceProbability.empty()
                    ? (densities[j] > 0 ? 1.0f : 0.0f)
                    : prediction.presenceProbability[index][offset + j];
                if (!prediction.posteriorStd.empty())
                    deviation[j] = prediction.posteriorStd[index][offset + j];
            }
            edp.values.push_back(std::move(densities));
            edp.presenceProbability.push_back(std::move(probability));
            edp.posteriorStd.push_back(std::move(deviation));
        }
        edp.validate();
        results.push_back({std::move(edp)});
    }
    return results;
}

void EdpMap::validate() const {
    if (unit != "1e15 atoms/cm2" || elements.empty() || values.empty() ||
        presenceProbability.size() != values.size() || posteriorStd.size() != values.size())
        throw std::invalid_argument("EDP requires three matrices with matching dimensions");
    std::unordered_set<std::string> names;
    for (const auto &element : elements)
        if (element.empty() || !names.insert(element).second)
            throw std::invalid_argument("EDP elements must be nonempty and unique");
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i].size() != elements.size() || presenceProbability[i].size() != elements.size() ||
            posteriorStd[i].size() != elements.size())
            throw std::invalid_argument("EDP matrix dimensions do not match elements");
        for (std::size_t j = 0; j < elements.size(); ++j) {
            const auto mean = values[i][j], probability = presenceProbability[i][j], std = posteriorStd[i][j];
            if (!std::isfinite(mean) || mean < 0 || !std::isfinite(probability) ||
                probability < 0 || probability > 1 || !std::isfinite(std) || std < 0)
                throw std::invalid_argument("invalid EDP density, probability, or standard deviation");
        }
    }
}

sample::SampleModel EdpMap::toSample(const sample::SampleModel &sampleTemplate) const {
    validate();
    sampleTemplate.validate();
    if (unit != "1e15 atoms/cm2" || elements.empty() || values.empty() ||
        sampleTemplate.layers.size() != values.size())
        throw std::invalid_argument("EDP unit or template dimensions do not match");
    std::unordered_set<std::string> names;
    for (const auto &element : elements)
        if (element.empty() || !names.insert(element).second)
            throw std::invalid_argument("EDP elements must be nonempty and unique");
    auto result = sampleTemplate;
    bool padding = false;
    std::size_t active = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto &row = values[i];
        if (row.size() != elements.size())
            throw std::invalid_argument("EDP row dimensions do not match elements");
        for (float value : row)
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("EDP must be finite and nonnegative");
        auto &layer = result.layers[i];
        if (layer.species.size() != elements.size())
            throw std::invalid_argument("EDP elements do not match template species");
        for (const auto &element : elements)
            if (std::none_of(layer.species.begin(), layer.species.end(),
                             [&](const auto &species) { return species.element == element; }))
                throw std::invalid_argument("EDP element missing from template: " + element);
        const double thickness = std::accumulate(row.begin(), row.end(), 0.0);
        if (thickness == 0) {
            padding = true;
            continue;
        }
        if (padding)
            throw std::invalid_argument("EDP zero layers must be trailing padding");
        layer.thickness = thickness;
        for (std::size_t j = 0; j < elements.size(); ++j) {
            auto species = std::find_if(layer.species.begin(), layer.species.end(),
                                        [&](const auto &v) { return v.element == elements[j]; });
            species->concentration = row[j] / thickness;
        }
        ++active;
    }
    if (!active)
        throw std::invalid_argument("cannot reconstruct an entirely empty EDP");
    result.layers.resize(active);
    result.validate();
    return result;
}
} // namespace ibeamlab::inference
