#include "forward_plan.h"
#include "session.h"
#include <ibeamlab/inference/models.h>
#include <ibeamlab/processing/pileup.h>
#include <ibeamlab/processing/rebin.h>
#include <ibeamlab/processing/transforms.h>
#include <algorithm>
#include <limits>
namespace ibeamlab::inference {
namespace {
const core::Spectrum &find_spectrum(std::span<const core::Spectrum> spectra,
                                    std::string_view label) {
    const core::Spectrum *found = nullptr;
    for (const auto &s : spectra)
        if (s.measurement == label) {
            if (found)
                throw std::invalid_argument("duplicate spectrum label");
            found = &s;
        }
    if (!found)
        throw std::invalid_argument("missing spectrum: " + std::string(label));
    return *found;
}
std::span<const model::SpectrumOperation> operations(const model::SpectrumPipeline &pipeline,
                                                     std::string_view label) {
    for (const auto &p : pipeline)
        if (p.measurement == label)
            return p.steps;
    return {};
}
template <class T> bool has(std::span<const model::SpectrumOperation> steps) {
    for (const auto &s : steps)
        if (std::holds_alternative<T>(s))
            return true;
    return false;
}
bool same_calibration(const core::Detector &a, const core::Detector &b) {
    return a.calibrationLinear == b.calibrationLinear &&
           a.calibrationOffset == b.calibrationOffset &&
           a.calibrationQuadratic == b.calibrationQuadratic;
}
void same_beam(const core::Measurement &a, const core::Measurement &b) {
    if (a.beam.particle != b.beam.particle || a.beam.energy != b.beam.energy ||
        a.beam.spread != b.beam.spread || a.detector.resolution != b.detector.resolution)
        throw std::invalid_argument("unsupported beam or resolution change: " + a.label);
}
const model::Package &supported(const model::Package &p) {
    p.validate();
    processing::require_supported_transforms(p.metadata.preprocessing());
    return p;
}
struct Runtime {
    model::Package package;
    detail::Session session;
    std::vector<float> input, output;
    core::Counts counts, scratch, old_edges, new_edges;
    processing::PileupWorkspace pileup;
    Runtime(model::Package p, InferenceOptions options)
        : package(std::move(p)), session(supported(package), options),
          input(package.metadata.input_width()), output(package.metadata.output_width()) {}
    void run() {
        processing::apply_preprocessing(input, package.metadata.preprocessing());
        session.run(input, output);
    }
    void prepare_spectrum(std::span<const double> source,
                          std::span<const model::SpectrumOperation> steps,
                          const core::Measurement &reference, const core::Measurement &actual,
                          std::size_t channels, bool forward, bool already_removed = false) {
        core::validate_counts(source);
        if (source.empty())
            throw std::invalid_argument("empty spectrum");
        if (source.data() != counts.data())
            counts.assign(source.begin(), source.end());
        const core::Detector *current = forward ? &reference.detector : &actual.detector;
        bool skip_removed = already_removed;
        for (const auto &step : steps)
            std::visit(
                [&](const auto &op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, model::MultiplyCounts>) {
                        for (auto &v : counts)
                            v *= op.factor;
                    } else if constexpr (std::is_same_v<T, model::ExposureScale>) {
                        double from = forward ? reference.acquisition.particlesSr
                                              : actual.acquisition.particlesSr,
                               to = forward ? actual.acquisition.particlesSr
                                            : reference.acquisition.particlesSr;
                        if (from <= 0 || to <= 0)
                            throw std::invalid_argument("positive exposure required");
                        for (auto &v : counts)
                            v *= to / from;
                    } else if constexpr (std::is_same_v<T, model::Rebin>) {
                        const auto &target = forward ? actual.detector : reference.detector;
                        auto size = op.channels ? op.channels : channels;
                        old_edges.resize(counts.size() + 1);
                        new_edges.resize(size + 1);
                        core::channel_edges(*current, old_edges);
                        core::channel_edges(target, new_edges);
                        scratch.resize(size);
                        processing::rebin(counts, old_edges, new_edges, scratch,
                                          op.require_full_coverage
                                              ? processing::Coverage::RequireFull
                                              : processing::Coverage::ZeroOutside);
                        counts.swap(scratch);
                        current = &target;
                    } else {
                        const auto &acquisition =
                            op.acquisition == model::AcquisitionSource::Reference
                                ? reference.acquisition
                                : actual.acquisition;
                        if constexpr (std::is_same_v<T, model::Pileup>) {
                            scratch.resize(2 * counts.size() - 1);
                            processing::apply_pileup(counts, scratch, acquisition.realTime,
                                                     acquisition.liveTime, op.tau_seconds, pileup);
                            counts.swap(scratch);
                        } else {
                            if (skip_removed &&
                                op.acquisition == model::AcquisitionSource::Requested) {
                                skip_removed = false;
                                return;
                            }
                            if (counts.size() % 2 == 0)
                                throw std::invalid_argument(
                                    "pileup removal requires complete odd-length support");
                            scratch.resize(counts.size() / 2 + 1);
                            processing::remove_pileup(counts, scratch, acquisition.realTime,
                                                      acquisition.liveTime, op.tau_seconds,
                                                      op.relative_tolerance);
                            counts.swap(scratch);
                        }
                    }
                    core::validate_counts(counts);
                },
                step);
    }
};
} // namespace
struct ForwardModel::Impl : Runtime {
    std::optional<detail::ParameterPlan> parameters;
    std::optional<detail::LayerPlan> layers;
    struct SpectrumPlan {
        std::size_t reference_index;
        std::span<const model::SpectrumOperation> steps;
    };
    std::vector<SpectrumPlan> spectra;
    Impl(model::Package p, InferenceOptions options)
        : Runtime(std::move(p), options) {
        const auto &f = std::get<model::ForwardModelMetadata>(package.metadata.model);
        if (package.metadata.layer_input()) layers.emplace(f);
        else parameters.emplace(f);
        for (const auto &spec : f.output_spectra.spectra) {
            const auto &reference = core::find_measurement(f.reference_setup, spec.measurement);
            spectra.push_back(
                {static_cast<std::size_t>(&reference - f.reference_setup.measurements.data()),
                 operations(f.postprocessing, spec.measurement)});
        }
    }
    void run_layers(std::span<const float> values, std::size_t count) {
        const auto width = package.metadata.input_width();
        if (!count || count > std::numeric_limits<std::size_t>::max() / width || values.size() != count * width)
            throw std::invalid_argument("AD input must have shape [layers>=1,elements]");
        for (float value : values)
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("AD must be finite and nonnegative");
        const auto &pipeline = package.metadata.preprocessing();
        if (pipeline.empty()) {
            session.run(values, output, 1, count);
            return;
        }
        if (values.data() != input.data()) input.assign(values.begin(), values.end());
        for (std::size_t offset = 0; offset < input.size(); offset += width) {
            auto row = std::span<float>(input).subspan(offset, width);
            if (std::any_of(row.begin(), row.end(), [](float v) { return v != 0; }))
                processing::apply_preprocessing(row, pipeline);
        }
        session.run(input, output, 1, count);
    }
    void finish(simulation::Result &result);
};
ForwardModel::ForwardModel(model::Package p, InferenceOptions options) {
    if (p.metadata.direction() != model::Direction::Forward)
        throw std::invalid_argument("forward package required");
    impl_ = std::make_unique<Impl>(std::move(p), options);
}
ForwardModel::~ForwardModel() = default;
const model::ModelMetadata &ForwardModel::metadata() const { return impl_->package.metadata; }
const InferenceOptions &ForwardModel::options() const { return impl_->session.options(); }
simulation::Result ForwardModel::predict(const core::Experiment &x) {
    simulation::Result result;
    predict(x, result);
    return result;
}
std::vector<simulation::Result>
ForwardModel::predict_batch(std::span<const core::Experiment> inputs) {
    std::vector<simulation::Result> result(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        predict(inputs[i], result[i]);
    return result;
}
void ForwardModel::predict(const core::Experiment &x, simulation::Result &result) {
    auto &r = *impl_;
    if (r.layers) {
        const auto count = x.sample.layers.size(), width = r.package.metadata.input_width();
        if (!count || count > std::numeric_limits<std::size_t>::max() / width)
            throw std::invalid_argument("invalid layer count");
        r.layers->prepare_setup(x.setup);
        r.input.resize(count * width);
        r.layers->pack(x.sample, r.input);
        r.run_layers(r.input, count);
    } else {
        r.parameters->prepare(x, r.input);
        r.run();
    }
    r.finish(result);
}
simulation::Result ForwardModel::predict_areal_density(std::span<const float> densities,
    std::size_t count, const core::ExperimentalSetup *setup) {
    simulation::Result result;
    predict_areal_density(densities, count, result, setup);
    return result;
}
void ForwardModel::predict_areal_density(std::span<const float> densities, std::size_t count,
    simulation::Result &result, const core::ExperimentalSetup *setup) {
    auto &r = *impl_;
    if (!r.layers) throw std::invalid_argument("AD inference requires a layer input schema");
    const auto &f = std::get<model::ForwardModelMetadata>(r.package.metadata.model);
    r.layers->prepare_setup(setup ? *setup : f.reference_setup);
    r.run_layers(densities, count);
    r.finish(result);
}
void ForwardModel::Impl::finish(simulation::Result &result) {
    auto &r = *this;
    const auto &f = std::get<model::ForwardModelMetadata>(r.package.metadata.model);
    result.failure.reset();
    result.spectra.resize(f.output_spectra.spectra.size());
    std::size_t offset = 0;
    for (std::size_t i = 0; i < f.output_spectra.spectra.size(); ++i) {
        const auto &spec = f.output_spectra.spectra[i];
        const auto &plan = r.spectra[i];
        const auto &reference = f.reference_setup.measurements[plan.reference_index];
        const auto &actual = r.layers ? r.layers->measurement(plan.reference_index)
                                     : r.parameters->measurement(plan.reference_index);
        auto &out = result.spectra[i];
        out.measurement = spec.measurement;
        if (plan.steps.empty()) {
            out.counts.resize(spec.channels);
            for (auto &value : out.counts) {
                value = r.output[offset++];
                if (value < 0)
                    throw std::invalid_argument("counts must be finite and nonnegative");
            }
            continue;
        }
        r.counts.resize(spec.channels);
        for (auto &value : r.counts)
            value = r.output[offset++];
        r.prepare_spectrum(r.counts, plan.steps, reference, actual, spec.channels, true);
        out.counts.assign(r.counts.begin(), r.counts.end());
    }
}
struct InverseModel::Impl : Runtime {
    using Runtime::Runtime;
    void finish(Prediction &result) {
        const auto &i = std::get<model::InverseModelMetadata>(package.metadata.model);
        run();
        auto cells = i.layers * i.elements.size();
        result.elements = i.elements;
        result.layers = i.layers;
        result.uncertainty_predicted = i.uncertainty;
        result.values.resize(cells);
        result.presence_probability.resize(cells);
        result.posterior_std.resize(cells);
        double scale = 1;
        for (const auto &step : i.postprocessing) {
            scale *= step.factor;
            if (!std::isfinite(scale))
                throw std::runtime_error("EDP scale overflow");
        }
        for (std::size_t k = 0; k < cells; ++k) {
            double value = output[k] * scale,
                   probability = i.uncertainty ? output[cells + k] : (value > 0 ? 1. : 0.),
                   std = i.uncertainty ? output[2 * cells + k] * scale : 0.;
            if (!std::isfinite(value) || !std::isfinite(std) || value < 0 || probability < 0 ||
                probability > 1 || std < 0)
                throw std::runtime_error("invalid inverse graph output");
            result.values[k] = value;
            result.presence_probability[k] = probability;
            result.posterior_std[k] = std;
        }
    }
};
InverseModel::InverseModel(model::Package p, InferenceOptions options) {
    if (p.metadata.direction() != model::Direction::Inverse)
        throw std::invalid_argument("inverse package required");
    impl_ = std::make_unique<Impl>(std::move(p), options);
}
InverseModel::~InverseModel() = default;
const model::ModelMetadata &InverseModel::metadata() const { return impl_->package.metadata; }
const InferenceOptions &InverseModel::options() const { return impl_->session.options(); }
Prediction InverseModel::predict(const InverseInput &input) {
    Prediction result;
    predict(input, result);
    return result;
}
std::vector<Prediction> InverseModel::predict_batch(std::span<const InverseInput> inputs) {
    std::vector<Prediction> result(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        predict(inputs[i], result[i]);
    return result;
}
void InverseModel::predict_prepared(std::span<const core::Spectrum> spectra, Prediction &result) {
    auto &r = *impl_;
    const auto &i = std::get<model::InverseModelMetadata>(r.package.metadata.model);
    if (spectra.size() != i.input_spectra.spectra.size())
        throw std::invalid_argument("prepared spectrum count mismatch");
    std::size_t offset = 0;
    for (const auto &spec : i.input_spectra.spectra) {
        const auto &s = find_spectrum(spectra, spec.measurement);
        if (s.counts.size() != spec.channels)
            throw std::invalid_argument("prepared spectrum shape mismatch");
        core::validate_counts(s.counts);
        for (double v : s.counts)
            r.input[offset++] = static_cast<float>(v);
    }
    r.finish(result);
}
void InverseModel::predict(const InverseInput &input, Prediction &result) {
    auto &r = *impl_;
    const auto &i = std::get<model::InverseModelMetadata>(r.package.metadata.model);
    input.setup.validate();
    if (input.spectra.size() != i.input_spectra.spectra.size() ||
        input.setup.measurements.size() != i.input_spectra.spectra.size())
        throw std::invalid_argument("inverse measurement count mismatch");
    std::size_t offset = 0;
    for (const auto &spec : i.input_spectra.spectra) {
        const auto &reference = core::find_measurement(i.reference_setup, spec.measurement);
        const auto &actual = core::find_measurement(input.setup, spec.measurement);
        same_beam(reference, actual);
        auto steps = operations(i.input_preparation, spec.measurement);
        if (!has<model::Rebin>(steps) && !same_calibration(reference.detector, actual.detector))
            throw std::invalid_argument("calibration changed without a rebin operation");
        if (!has<model::ExposureScale>(steps) &&
            reference.acquisition.particlesSr != actual.acquisition.particlesSr)
            throw std::invalid_argument("exposure changed without exposure scaling");
        const auto &s = find_spectrum(input.spectra, spec.measurement);
        r.prepare_spectrum(s.counts, steps, reference, actual, spec.channels, false,
                           input.pileup_already_removed);
        if (r.counts.size() != spec.channels)
            throw std::invalid_argument("prepared spectrum width differs from schema");
        for (double v : r.counts)
            r.input[offset++] = static_cast<float>(v);
    }
    r.finish(result);
}
} // namespace ibeamlab::inference
