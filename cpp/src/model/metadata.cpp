#include <ibeamlab/core/isotopes.h>
#include <algorithm>
#include <ibeamlab/model/metadata.h>
#include <unordered_set>
namespace ibeamlab::model {
namespace {
void coefficients(std::span<const double> values, std::size_t width) {
    if (values.size() != 1 && values.size() != width)
        throw std::invalid_argument("scaler coefficients must be scalar or match tensor width");
    for (double v : values)
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite scaler coefficient");
}
double coefficient(const core::Counts &values, std::size_t i) {
    return values[values.size() == 1 ? 0 : i];
}
void tensor_valid(const TensorPipeline &pipeline, std::size_t width) {
    if (pipeline.size() > 1024)
        throw std::invalid_argument("too many preprocessing steps");
    for (const auto &step : pipeline)
        std::visit(
            [&](const auto &op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, MinMaxScale>) {
                    coefficients(op.minimum, width);
                    coefficients(op.maximum, width);
                    if (!std::isfinite(op.low) || !std::isfinite(op.high) || op.high <= op.low)
                        throw std::invalid_argument("invalid min-max output range");
                    for (std::size_t i = 0; i < width; ++i)
                        if (coefficient(op.maximum, i) <= coefficient(op.minimum, i))
                            throw std::invalid_argument("min-max range must be positive");
                } else if constexpr (std::is_same_v<T, StandardScale>) {
                    coefficients(op.mean, width);
                    coefficients(op.deviation, width);
                    for (double v : op.deviation)
                        if (v <= 0)
                            throw std::invalid_argument("standard deviation must be positive");
                } else if constexpr (std::is_same_v<T, MultiplyScale>)
                    coefficients(op.factors, width);
                else if constexpr (std::is_same_v<T, CustomTransform>) {
                    if (op.identifier.empty() || !op.version)
                        throw std::invalid_argument(
                            "custom transform needs identifier and version");
                    if (op.parameters.size() > 1024)
                        throw std::invalid_argument("too many custom parameters");
                    for (const auto &[name, values] : op.parameters) {
                        if (name.empty())
                            throw std::invalid_argument("empty custom parameter name");
                        for (double v : values)
                            if (!std::isfinite(v))
                                throw std::invalid_argument("nonfinite custom parameter");
                    }
                }
            },
            step);
}
void transform_valid(const TransformPipeline &pipeline, std::size_t width, bool uncertainty = false) {
    if (pipeline.direction != TransformDirection::Forward && pipeline.direction != TransformDirection::Inverse)
        throw std::invalid_argument("unknown transform direction");
    if (pipeline.location != TransformLocation::Runtime && pipeline.location != TransformLocation::Graph)
        throw std::invalid_argument("unknown transform execution location");
    tensor_valid(pipeline.steps, width);
    for (const auto &step : pipeline.steps) {
        if (const auto *scale = std::get_if<MultiplyScale>(&step))
            if (pipeline.direction == TransformDirection::Inverse)
                for (double v : scale->factors)
                    if (v == 0) throw std::invalid_argument("cannot invert zero scaling");
        if (const auto *scale = std::get_if<MinMaxScale>(&step))
            if (scale->clip && (pipeline.direction == TransformDirection::Inverse || uncertainty))
                throw std::invalid_argument("clipped scaling is not invertible or affine for uncertainty");
        if (uncertainty && (std::holds_alternative<Log1p>(step) || std::holds_alternative<CustomTransform>(step)))
            throw std::invalid_argument("uncertainty output transforms must be affine");
    }
}
void transforms_valid(const SpectrumTransforms &pipelines, const std::vector<SpectrumSpec> &schema) {
    if (pipelines.size() > 1024) throw std::invalid_argument("too many spectrum transforms");
    std::unordered_set<std::string> labels;
    for (const auto &entry : pipelines) {
        if (!labels.insert(entry.measurement).second) throw std::invalid_argument("duplicate numerical transform label");
        const auto found = std::find_if(schema.begin(), schema.end(), [&](const auto &spec) { return spec.measurement == entry.measurement; });
        if (found == schema.end()) throw std::invalid_argument("numerical transform references an unknown spectrum");
        transform_valid(entry.transform, found->channels);
    }
}
void spectrum_valid(const SpectrumPipeline &pipeline, const std::vector<SpectrumSpec> &schema,
                    const core::ExperimentalSetup &setup) {
    if (pipeline.size() > 1024)
        throw std::invalid_argument("too many spectrum pipelines");
    std::unordered_set<std::string> labels;
    for (const auto &entry : pipeline) {
        if (!labels.insert(entry.measurement).second)
            throw std::invalid_argument("duplicate spectrum processing label");
        auto found = std::find_if(schema.begin(), schema.end(), [&](const auto &s) {
            return s.measurement == entry.measurement;
        });
        if (found == schema.end())
            throw std::invalid_argument("processing references an unknown spectrum");
        const auto &m = core::find_measurement(setup, entry.measurement);
        if (entry.steps.size() > 1024)
            throw std::invalid_argument("too many spectrum processing steps");
        for (const auto &step : entry.steps)
            std::visit(
                [&](const auto &op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, ExposureScale>) {
                        if (m.acquisition.particlesSr <= 0)
                            throw std::invalid_argument("reference exposure must be positive");
                    } else if constexpr (std::is_same_v<T, Rebin>) {
                        if (op.channels > 1000000)
                            throw std::invalid_argument("rebin grid too large");
                    } else {
                        if (!std::isfinite(op.tau_seconds) || op.tau_seconds < 0)
                            throw std::invalid_argument("invalid pileup time constant");
                        if (op.acquisition != AcquisitionSource::Requested &&
                            op.acquisition != AcquisitionSource::Reference)
                            throw std::invalid_argument("invalid acquisition source");
                        if constexpr (std::is_same_v<T, RemovePileup>)
                            if (!std::isfinite(op.relative_tolerance) ||
                                op.relative_tolerance <= 0 || op.relative_tolerance >= 1)
                                throw std::invalid_argument("invalid pileup removal tolerance");
                    }
                },
                step);
    }
}
} // namespace
std::size_t SpectraSchema::width() const {
    if (spectra.size() > 1024)
        throw std::invalid_argument("too many spectra");
    std::size_t n = 0;
    for (const auto &s : spectra) {
        if ((s.state != SpectrumState::Bare && s.state != SpectrumState::PiledUp) || s.unit != "counts")
            throw std::invalid_argument("invalid reference spectrum state or unit");
        if (!s.channels || s.channels > 1000000 || n > 10000000 - s.channels)
            throw std::invalid_argument("invalid spectrum dimensions");
        n += s.channels;
    }
    return n;
}
std::size_t SpectraSchema::offset(std::string_view label) const {
    width();
    std::size_t offset = 0, result = 0;
    bool found = false;
    for (const auto &s : spectra) {
        if (s.measurement == label) {
            if (found)
                throw std::invalid_argument("duplicate spectrum label");
            result = offset;
            found = true;
        }
        offset += s.channels;
    }
    if (!found)
        throw std::invalid_argument("unknown spectrum label");
    return result;
}
Direction ModelMetadata::direction() const noexcept {
    return std::holds_alternative<ForwardModelMetadata>(model) ? Direction::Forward
                                                               : Direction::Inverse;
}
const core::ExperimentalSetup &ModelMetadata::reference_setup() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return f->reference_setup;
    return std::get<InverseModelMetadata>(model).reference_setup;
}
const std::vector<SpectrumSpec> &ModelMetadata::spectra() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return f->output_spectra.spectra;
    return std::get<InverseModelMetadata>(model).input_spectra.spectra;
}
std::span<const core::Isotope> LayerInputSchema::composition(std::string_view element) const {
    if (std::find(elements.begin(), elements.end(), element) == elements.end())
        throw std::invalid_argument("unsupported model element: " + std::string(element));
    for (const auto &entry : isotope_compositions)
        if (entry.element == element && !entry.isotopes.empty()) return entry.isotopes;
    auto natural = core::natural_isotopes(element);
    if (natural.empty()) throw std::invalid_argument("model requires explicit isotopes for " + std::string(element));
    return natural;
}
std::size_t ModelMetadata::input_width() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return std::visit([](const auto &input) -> std::size_t {
            using T = std::decay_t<decltype(input)>;
            if constexpr (std::is_same_v<T, LayerInputSchema>) return input.elements.size();
            else return input.parameters.size();
        }, f->input_preparation);
    return std::get<InverseModelMetadata>(model).input_spectra.width();
}
std::size_t ModelMetadata::output_width() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return f->output_spectra.width();
    const auto &i = std::get<InverseModelMetadata>(model);
    if (i.output_preparation.elements.size() > 256 || i.output_preparation.layers > 10000)
        throw std::invalid_argument("EDP dimensions too large");
    return i.output_preparation.layers * i.output_preparation.elements.size() * (i.output_preparation.uncertainty ? 3 : 1);
}
bool ModelMetadata::layer_input() const noexcept {
    const auto *f = std::get_if<ForwardModelMetadata>(&model);
    return f && std::holds_alternative<LayerInputSchema>(f->input_preparation);
}
void ModelMetadata::validate() const {
    if (model.valueless_by_exception())
        throw std::invalid_argument("missing model specification");
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model)) {
        f->reference_setup.validate();
        if (f->input_preparation.valueless_by_exception())
            throw std::invalid_argument("missing forward input schema");
        if (const auto *p = std::get_if<ParameterInputSchema>(&f->input_preparation)) {
            core::Experiment reference{p->reference_sample, f->reference_setup};
            reference.validate();
            core::validate_parameters(reference, p->parameters);
            if (p->parameters.empty())
                throw std::invalid_argument("parameter input requires parameters");
        } else {
            const auto &layers = std::get<LayerInputSchema>(f->input_preparation);
            if (layers.elements.empty() || layers.elements.size() > 256)
                throw std::invalid_argument("layer input requires 1..256 ordered elements");
            std::unordered_set<std::string> names;
            for (const auto &e : layers.elements)
                if (e.empty() || !names.insert(e).second)
                    throw std::invalid_argument("invalid layer input elements");
            std::unordered_set<std::string> assumptions;
            for (const auto &entry : layers.isotope_compositions) {
                if (!names.contains(entry.element) || !assumptions.insert(entry.element).second)
                    throw std::invalid_argument("unknown or duplicate model isotope assumption");
                core::validate_isotopes(entry.element, entry.isotopes);
            }
            for (const auto &element : layers.elements)
                core::validate_isotopes(element, layers.composition(element));
        }
    } else {
        const auto &i = std::get<InverseModelMetadata>(model);
        i.reference_setup.validate();
        if (!i.output_preparation.layers || i.output_preparation.elements.empty())
            throw std::invalid_argument("inverse model requires layers and elements");
        std::unordered_set<std::string> names;
        for (const auto &e : i.output_preparation.elements)
            if (e.empty() || !names.insert(e).second)
                throw std::invalid_argument("invalid EDP elements");
        if (i.output_preparation.unit != "1e15 atoms/cm2")
            throw std::invalid_argument("unsupported EDP physical unit");

    }
    const auto input = input_width(), output = output_width();
    if (!input || !output || input_name.empty() || output_name.empty())
        throw std::invalid_argument("invalid tensor schema");
    const auto &setup = reference_setup();
    const auto &schema = spectra();
    if (schema.empty() || schema.size() != setup.measurements.size())
        throw std::invalid_argument("model measurement count mismatch");
    std::unordered_set<std::string> names;
    for (const auto &s : schema) {
        if (!names.insert(s.measurement).second)
            throw std::invalid_argument("duplicate spectrum label");
        auto &m = core::find_measurement(setup, s.measurement);
        core::Counts edges(s.channels + 1);
        core::channel_edges(m.detector, edges);
    }
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model)) {
        transform_valid(f->input_transform, input);
        transforms_valid(f->output_transform, schema);
        spectrum_valid(f->output_preparation, schema, setup);
        for (const auto &entry : f->output_preparation) {
            auto state = std::find_if(schema.begin(), schema.end(), [&](const auto &v) { return v.measurement == entry.measurement; })->state;
            for (const auto &step : entry.steps) {
                if (std::holds_alternative<Pileup>(step)) {
                    if (state != SpectrumState::Bare) throw std::invalid_argument("pileup requires bare reference spectra");
                    state = SpectrumState::PiledUp;
                } else if (std::holds_alternative<RemovePileup>(step)) {
                    if (state != SpectrumState::PiledUp) throw std::invalid_argument("pileup removal requires piled-up spectra");
                    state = SpectrumState::Bare;
                }
            }
        }
    } else {
        const auto &i = std::get<InverseModelMetadata>(model);
        spectrum_valid(i.input_preparation, schema, setup);
        transforms_valid(i.input_transform, schema);
        transform_valid(i.output_transform, i.output_preparation.layers * i.output_preparation.elements.size(),
                        i.output_preparation.uncertainty);
        for (const auto &entry : i.input_preparation) {
            auto state = std::find_if(schema.begin(), schema.end(), [&](const auto &v) { return v.measurement == entry.measurement; })->state;
            for (auto step = entry.steps.rbegin(); step != entry.steps.rend(); ++step) {
                if (std::holds_alternative<Pileup>(*step)) {
                    if (state != SpectrumState::PiledUp) throw std::invalid_argument("prepared pileup state differs from inverse schema");
                    state = SpectrumState::Bare;
                } else if (std::holds_alternative<RemovePileup>(*step)) {
                    if (state != SpectrumState::Bare) throw std::invalid_argument("prepared bare state differs from inverse schema");
                    state = SpectrumState::PiledUp;
                }
            }
        }
    }
}
} // namespace ibeamlab::model
