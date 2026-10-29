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
                    if constexpr (std::is_same_v<T, MultiplyCounts>) {
                        if (!std::isfinite(op.factor) || op.factor < 0)
                            throw std::invalid_argument("invalid spectrum multiplier");
                    } else if constexpr (std::is_same_v<T, ExposureScale>) {
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
        return f->reference.setup;
    return std::get<InverseModelMetadata>(model).reference_setup;
}
const std::vector<SpectrumSpec> &ModelMetadata::spectra() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return f->output_spectra.spectra;
    return std::get<InverseModelMetadata>(model).input_spectra.spectra;
}
const TensorPipeline &ModelMetadata::preprocessing() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return f->preprocessing;
    return std::get<InverseModelMetadata>(model).preprocessing;
}
std::size_t ModelMetadata::input_width() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return f->input_parameters.size();
    return std::get<InverseModelMetadata>(model).input_spectra.width();
}
std::size_t ModelMetadata::output_width() const {
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        return f->output_spectra.width();
    const auto &i = std::get<InverseModelMetadata>(model);
    if (i.elements.size() > 256 || i.layers > 10000)
        throw std::invalid_argument("EDP dimensions too large");
    return i.layers * i.elements.size() * (i.uncertainty ? 3 : 1);
}
void ModelMetadata::validate() const {
    if (model.valueless_by_exception())
        throw std::invalid_argument("missing model specification");
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model)) {
        f->reference.validate();
        core::validate_parameters(f->reference, f->input_parameters);
        if (f->input_parameters.empty())
            throw std::invalid_argument("forward model requires parameters");
    } else {
        const auto &i = std::get<InverseModelMetadata>(model);
        i.reference_setup.validate();
        if (!i.layers || i.elements.empty())
            throw std::invalid_argument("inverse model requires layers and elements");
        std::unordered_set<std::string> names;
        for (const auto &e : i.elements)
            if (e.empty() || !names.insert(e).second)
                throw std::invalid_argument("invalid EDP elements");
        if (i.postprocessing.size() > 1024)
            throw std::invalid_argument("too many EDP processing steps");
        for (const auto &s : i.postprocessing)
            if (!std::isfinite(s.factor) || s.factor <= 0)
                throw std::invalid_argument("invalid EDP multiplier");
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
    tensor_valid(preprocessing(), input);
    if (const auto *f = std::get_if<ForwardModelMetadata>(&model))
        spectrum_valid(f->postprocessing, schema, setup);
    else
        spectrum_valid(std::get<InverseModelMetadata>(model).input_preparation, schema, setup);
}
} // namespace ibeamlab::model
