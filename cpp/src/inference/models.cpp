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
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("batch tensor size overflow");
    return a * b;
}
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
                    if constexpr (std::is_same_v<T, model::ExposureScale>) {
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
                            if (forward && !session.options().apply_pileup) return;
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
const model::TransformPipeline &transform(const model::SpectrumTransforms &pipelines, std::string_view label) {
    static const model::TransformPipeline identity;
    for (const auto &entry : pipelines) if (entry.measurement == label) return entry.transform;
    return identity;
}
struct ForwardModel::Impl : Runtime {
    processing::TransformPlan input_transform;
    std::optional<detail::ParameterPlan> parameters;
    std::optional<detail::LayerPlan> layers;
    struct SpectrumPlan {
        std::size_t reference_index;
        std::span<const model::SpectrumOperation> steps;
        processing::TransformPlan output_transform;
    };
    std::vector<SpectrumPlan> spectra;
    std::vector<std::size_t> channels, reference_channels;
    std::size_t processed_width = 0;
    Impl(model::Package p, InferenceOptions options)
        : Runtime(std::move(p), options), input_transform(
            std::get<model::ForwardModelMetadata>(package.metadata.model).input_transform,
            package.metadata.input_width()) {
        const auto &f = std::get<model::ForwardModelMetadata>(package.metadata.model);
        if (package.metadata.layer_input()) layers.emplace(f);
        else parameters.emplace(f);
        for (const auto &spec : f.output_spectra.spectra) {
            const auto &reference = core::find_measurement(f.reference_setup, spec.measurement);
            spectra.push_back(
                {static_cast<std::size_t>(&reference - f.reference_setup.measurements.data()),
                 operations(f.output_preparation, spec.measurement),
                 processing::TransformPlan(transform(f.output_transform, spec.measurement), spec.channels)});
            reference_channels.push_back(spec.channels);
            auto count = spec.channels;
            for (const auto &step : spectra.back().steps)
                std::visit([&](const auto &op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, model::Rebin>)
                        count = op.channels ? op.channels : spec.channels;
                    else if constexpr (std::is_same_v<T, model::Pileup>) {
                        if (session.options().apply_pileup) count = product(count, 2) - 1;
                    } else if constexpr (std::is_same_v<T, model::RemovePileup>) {
                        if (count % 2 == 0) throw std::invalid_argument("pileup removal requires full odd-length support");
                        count = count / 2 + 1;
                    }
                }, step);
            channels.push_back(count);
            if (processed_width > std::numeric_limits<std::size_t>::max() - count)
                throw std::invalid_argument("processed spectrum size overflow");
            processed_width += count;
        }
    }
    void run_layers(std::span<const float> values, std::size_t count, std::size_t batch = 1) {
        const auto width = package.metadata.input_width();
        if (!count || values.size() != product(product(batch, count), width))
            throw std::invalid_argument("AD input must have shape [batch,layers>=1,elements]");
        output.resize(product(batch, package.metadata.output_width()));
        if (!batch) return;
        for (float value : values)
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("AD must be finite and nonnegative");
        if (input_transform.empty()) {
            session.run(values, output, batch, count);
            return;
        }
        if (values.data() != input.data()) input.assign(values.begin(), values.end());
        for (std::size_t offset = 0; offset < input.size(); offset += width) {
            auto row = std::span<float>(input).subspan(offset, width);
            if (std::any_of(row.begin(), row.end(), [](float v) { return v != 0; })) {
                input_transform.apply(row);
                if (std::none_of(row.begin(), row.end(), [](float v) { return v != 0; }))
                    throw std::invalid_argument("input transform erases a nonempty layer; zero is reserved for padding");
            }
        }
        session.run(input, output, batch, count);
    }
    void finish(simulation::Result &result, PredictionDomain domain, std::size_t row = 0,
                const core::ExperimentalSetup *setup = nullptr);
    void finish_spectrum(std::size_t index, std::span<const float> values,
                         std::span<double> destination, const core::ExperimentalSetup *setup, PredictionDomain domain);
};
ForwardModel::ForwardModel(model::Package p, InferenceOptions options) {
    if (p.metadata.direction() != model::Direction::Forward)
        throw std::invalid_argument("forward package required");
    impl_ = std::make_unique<Impl>(std::move(p), options);
}
ForwardModel::~ForwardModel() = default;
const model::ModelMetadata &ForwardModel::metadata() const { return impl_->package.metadata; }
const InferenceOptions &ForwardModel::options() const { return impl_->session.options(); }
simulation::Result ForwardModel::predict(const core::Experiment &x, PredictionDomain domain) {
    simulation::Result result;
    predict(x, result, domain);
    return result;
}
std::vector<simulation::Result>
ForwardModel::predict_batch(std::span<const core::Experiment> inputs, PredictionDomain domain) {
    std::vector<simulation::Result> result(inputs.size());
    predict_batch(inputs, result, domain);
    return result;
}
void ForwardModel::predict_batch(std::span<const core::Experiment> inputs,
                                 std::span<simulation::Result> result, PredictionDomain domain) {
    if (inputs.size() != result.size()) throw std::invalid_argument("batch result count mismatch");
    if (inputs.empty()) return;
    auto &r = *impl_;
    const auto width = r.package.metadata.input_width();
    std::size_t layers = 1;
    if (r.layers)
        for (const auto &x : inputs) {
            if (x.sample.layers.empty()) throw std::invalid_argument("invalid layer count");
            layers = std::max(layers, x.sample.layers.size());
        }
    const auto stride = product(layers, width);
    r.input.resize(product(inputs.size(), stride));
    r.output.resize(product(inputs.size(), r.package.metadata.output_width()));
    std::fill(r.input.begin(), r.input.end(), 0.f);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto row = std::span<float>(r.input).subspan(i * stride, stride);
        if (r.layers) {
            r.layers->prepare_setup(inputs[i].setup);
            r.layers->pack(inputs[i].sample, row.first(product(inputs[i].sample.layers.size(), width)));
        } else {
            r.parameters->prepare(inputs[i], row);
            r.input_transform.apply(row);
        }
    }
    if (r.layers) r.run_layers(r.input, layers, inputs.size());
    else r.session.run(r.input, r.output, inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        r.finish(result[i], domain, i, &inputs[i].setup);
}
void ForwardModel::predict(const core::Experiment &x, simulation::Result &result, PredictionDomain domain) {
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
        r.input.resize(r.package.metadata.input_width());
        r.output.resize(r.package.metadata.output_width());
        r.parameters->prepare(x, r.input);
        r.input_transform.apply(r.input);
        r.session.run(r.input, r.output);
    }
    r.finish(result, domain);
}
simulation::Result ForwardModel::predict_areal_density(std::span<const float> densities,
    std::size_t count, const core::ExperimentalSetup *setup, PredictionDomain domain) {
    simulation::Result result;
    predict_areal_density(densities, count, result, setup, domain);
    return result;
}
void ForwardModel::predict_areal_density(std::span<const float> densities, std::size_t count,
    simulation::Result &result, const core::ExperimentalSetup *setup, PredictionDomain domain) {
    auto &r = *impl_;
    if (!r.layers) throw std::invalid_argument("AD inference requires a layer input schema");
    const auto &f = std::get<model::ForwardModelMetadata>(r.package.metadata.model);
    r.layers->prepare_setup(setup ? *setup : f.reference_setup);
    r.run_layers(densities, count);
    r.finish(result, domain);
}
std::span<const std::size_t> ForwardModel::spectrum_channels(PredictionDomain domain) const {
    return domain == PredictionDomain::Reference ? impl_->reference_channels : impl_->channels;
}
void ForwardModel::predict_areal_density_batch(std::span<const float> values, std::size_t batch,
    std::size_t layers, std::span<double> output, const core::ExperimentalSetup *setup, PredictionDomain domain) {
    auto &r = *impl_;
    if (!r.layers) throw std::invalid_argument("AD inference requires a layer input schema");
    const auto channels = spectrum_channels(domain);
    std::size_t stride = 0;
    for (auto count : channels) stride += count;
    if (output.size() != product(batch, stride))
        throw std::invalid_argument("batch output shape mismatch");
    const auto source_address = reinterpret_cast<std::uintptr_t>(values.data());
    const auto destination_address = reinterpret_cast<std::uintptr_t>(output.data());
    if (!values.empty() && !output.empty() && source_address < destination_address + output.size_bytes() &&
        destination_address < source_address + values.size_bytes())
        throw std::invalid_argument("AD input and output must not overlap");
    const auto &f = std::get<model::ForwardModelMetadata>(r.package.metadata.model);
    r.layers->prepare_setup(setup ? *setup : f.reference_setup);
    r.run_layers(values, layers, batch);
    for (std::size_t row = 0; row < batch; ++row) {
        std::size_t source = row * r.package.metadata.output_width(), target = row * stride;
        for (std::size_t i = 0; i < r.channels.size(); ++i) {
            const auto count = f.output_spectra.spectra[i].channels;
            r.finish_spectrum(i, std::span<const float>(r.output).subspan(source, count),
                              output.subspan(target, channels[i]), nullptr, domain);
            source += count;
            target += channels[i];
        }
    }
}
void ForwardModel::Impl::finish_spectrum(std::size_t index, std::span<const float> values,
    std::span<double> destination, const core::ExperimentalSetup *setup, PredictionDomain domain) {
    const auto &f = std::get<model::ForwardModelMetadata>(package.metadata.model);
    const auto &plan = spectra[index];
    if (plan.steps.empty() || domain == PredictionDomain::Reference) {
        for (std::size_t k = 0; k < values.size(); ++k) {
            destination[k] = values[k];
        }
        spectra[index].output_transform.apply(destination);
        core::validate_counts(destination);
        return;
    }
    const auto &reference = f.reference_setup.measurements[plan.reference_index];
    const auto &actual = setup ? core::find_measurement(*setup, reference.label)
        : (layers ? layers->measurement(plan.reference_index) : parameters->measurement(plan.reference_index));
    counts.assign(values.begin(), values.end());
    spectra[index].output_transform.apply(std::span<double>(counts));
    prepare_spectrum(counts, plan.steps, reference, actual, values.size(), true);
    if (counts.size() != destination.size()) throw std::invalid_argument("processed spectrum shape mismatch");
    std::copy(counts.begin(), counts.end(), destination.begin());
}
void ForwardModel::Impl::finish(simulation::Result &result, PredictionDomain domain, std::size_t row,
                              const core::ExperimentalSetup *setup) {
    auto &r = *this;
    const auto &f = std::get<model::ForwardModelMetadata>(r.package.metadata.model);
    result.failure.reset();
    result.spectra.resize(f.output_spectra.spectra.size());
    std::size_t offset = row * package.metadata.output_width();
    for (std::size_t i = 0; i < f.output_spectra.spectra.size(); ++i) {
        const auto &spec = f.output_spectra.spectra[i];
        auto &out = result.spectra[i];
        out.measurement = spec.measurement;
        out.counts.resize(domain == PredictionDomain::Reference ? spec.channels : channels[i]);
        finish_spectrum(i, std::span<const float>(output).subspan(offset, spec.channels), out.counts, setup, domain);
        offset += spec.channels;
    }
}
struct InverseModel::Impl : Runtime {
    struct SpectrumPlan {
        const model::SpectrumSpec *spec;
        const core::Measurement *reference;
        std::span<const model::SpectrumOperation> preparation;
        processing::TransformPlan transform;
    };
    std::vector<SpectrumPlan> spectra;
    processing::TransformPlan output_transform;
    Impl(model::Package p, InferenceOptions options)
        : Runtime(std::move(p), options), output_transform(
            std::get<model::InverseModelMetadata>(package.metadata.model).output_transform,
            std::get<model::InverseModelMetadata>(package.metadata.model).output_preparation.layers *
            std::get<model::InverseModelMetadata>(package.metadata.model).output_preparation.elements.size(),
            std::get<model::InverseModelMetadata>(package.metadata.model).output_preparation.uncertainty) {
        const auto &i = std::get<model::InverseModelMetadata>(package.metadata.model);
        for (const auto &spec : i.input_spectra.spectra)
            spectra.push_back({&spec, &core::find_measurement(i.reference_setup, spec.measurement),
                operations(i.input_preparation, spec.measurement),
                processing::TransformPlan(transform(i.input_transform, spec.measurement), spec.channels)});
    }
    void prepare(const InverseInput &, std::span<float>);
    void prepare_reference(std::span<const core::Spectrum> supplied, std::span<float> destination) {
        if (supplied.size() != spectra.size()) throw std::invalid_argument("reference spectrum count mismatch");
        std::size_t offset = 0;
        for (auto &plan : spectra) {
            const auto &source = find_spectrum(supplied, plan.spec->measurement);
            if (source.counts.size() != plan.spec->channels) throw std::invalid_argument("reference spectrum shape mismatch");
            core::validate_counts(source.counts);
            auto row = destination.subspan(offset, plan.spec->channels);
            for (std::size_t k = 0; k < row.size(); ++k) row[k] = static_cast<float>(source.counts[k]);
            plan.transform.apply(row);
            offset += plan.spec->channels;
        }
    }
    void finish(Prediction &result, std::size_t row = 0) {
        const auto &schema = std::get<model::InverseModelMetadata>(package.metadata.model).output_preparation;
        const auto values = std::span<const float>(output).subspan(row * package.metadata.output_width(), package.metadata.output_width());
        const auto cells = schema.layers * schema.elements.size();
        result.elements = schema.elements;
        result.layers = schema.layers;
        result.unit = schema.unit;
        result.uncertainty_predicted = schema.uncertainty;
        result.values.resize(cells);
        result.presence_probability.resize(cells);
        result.posterior_std.resize(cells);
        for (std::size_t k = 0; k < cells; ++k) {
            result.values[k] = values[k];
            result.posterior_std[k] = schema.uncertainty ? values[2 * cells + k] : 0.;
        }
        if (schema.uncertainty) output_transform.apply_uncertainty(result.values, result.posterior_std);
        else output_transform.apply(std::span<double>(result.values));
        for (std::size_t k = 0; k < cells; ++k) {
            const double value = result.values[k], std = result.posterior_std[k];
            const double probability = schema.uncertainty ? values[cells + k] : (value > 0 ? 1. : 0.);
            if (!std::isfinite(value) || !std::isfinite(std) || value < 0 || probability < 0 || probability > 1 || std < 0)
                throw std::runtime_error("invalid inverse graph output");
            result.presence_probability[k] = probability;
        }
    }
};
InverseModel::InverseModel(model::Package p, InferenceOptions options) {
    if (p.metadata.direction() != model::Direction::Inverse) throw std::invalid_argument("inverse package required");
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
std::vector<Prediction> InverseModel::predict_batch(std::span<const InverseInput> inputs, PredictionDomain domain) {
    std::vector<Prediction> result(inputs.size());
    predict_batch(inputs, result, domain);
    return result;
}
void InverseModel::predict_batch(std::span<const InverseInput> inputs, std::span<Prediction> result, PredictionDomain domain) {
    if (inputs.size() != result.size()) throw std::invalid_argument("batch result count mismatch");
    if (inputs.empty()) return;
    auto &r = *impl_;
    const auto width = r.package.metadata.input_width();
    r.input.resize(product(inputs.size(), width));
    r.output.resize(product(inputs.size(), r.package.metadata.output_width()));
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        auto values = std::span<float>(r.input).subspan(row * width, width);
        if (domain == PredictionDomain::Reference) r.prepare_reference(inputs[row].spectra, values);
        else r.prepare(inputs[row], values);
    }
    r.session.run(r.input, r.output, inputs.size());
    for (std::size_t row = 0; row < inputs.size(); ++row) r.finish(result[row], row);
}
void InverseModel::predict_reference(std::span<const core::Spectrum> spectra, Prediction &result) {
    auto &r = *impl_;
    r.input.resize(r.package.metadata.input_width());
    r.output.resize(r.package.metadata.output_width());
    r.prepare_reference(spectra, r.input);
    r.session.run(r.input, r.output);
    r.finish(result);
}
void InverseModel::predict(const InverseInput &input, Prediction &result) {
    auto &r = *impl_;
    r.input.resize(r.package.metadata.input_width());
    r.output.resize(r.package.metadata.output_width());
    r.prepare(input, r.input);
    r.session.run(r.input, r.output);
    r.finish(result);
}
void InverseModel::Impl::prepare(const InverseInput &measurement, std::span<float> destination) {
    measurement.setup.validate();
    if (measurement.spectra.size() != spectra.size() || measurement.setup.measurements.size() != spectra.size())
        throw std::invalid_argument("inverse measurement count mismatch");
    std::size_t offset = 0;
    for (auto &plan : spectra) {
        const auto &spec = *plan.spec;
        const auto &reference = *plan.reference;
        const auto &actual = core::find_measurement(measurement.setup, spec.measurement);
        same_beam(reference, actual);
        if (!has<model::Rebin>(plan.preparation) && !same_calibration(reference.detector, actual.detector))
            throw std::invalid_argument("calibration changed without a rebin operation");
        if (!has<model::ExposureScale>(plan.preparation) && reference.acquisition.particlesSr != actual.acquisition.particlesSr)
            throw std::invalid_argument("exposure changed without exposure scaling");
        const auto &source = find_spectrum(measurement.spectra, spec.measurement);
        prepare_spectrum(source.counts, plan.preparation, reference, actual, spec.channels, false,
                         measurement.pileup_already_removed);
        if (counts.size() != spec.channels) throw std::invalid_argument("prepared spectrum width differs from schema");
        auto row = destination.subspan(offset, spec.channels);
        for (std::size_t k = 0; k < row.size(); ++k) row[k] = static_cast<float>(counts[k]);
        plan.transform.apply(row);
        offset += spec.channels;
    }
}
} // namespace ibeamlab::inference
