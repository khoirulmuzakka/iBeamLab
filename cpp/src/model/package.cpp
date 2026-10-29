#include "io.h"
#include <ibeamlab/model/package.h>
namespace ibeamlab::model {
namespace {
void encode_parameters(detail::Encoder &e, std::span<const core::ParameterSpec> parameters) {
    e.integer(parameters.size());
    for (const auto &p : parameters) {
        e.text(p.name);
        detail::encode(e, p.target);
        e.text(p.unit);
    }
}
std::vector<core::ParameterSpec> decode_parameters(detail::Decoder &d) {
    std::vector<core::ParameterSpec> parameters(d.count(100000));
    for (auto &p : parameters) {
        p.name = d.text();
        p.target = detail::target(d);
        p.unit = d.text();
    }
    return parameters;
}
void encode_spectra(detail::Encoder &e, const SpectraSchema &schema) {
    e.integer(schema.spectra.size());
    for (const auto &s : schema.spectra) {
        e.text(s.measurement);
        e.integer(s.channels);
    }
}
SpectraSchema decode_spectra(detail::Decoder &d) {
    std::vector<SpectrumSpec> spectra(d.count(1024));
    for (auto &s : spectra) {
        s.measurement = d.text();
        s.channels = d.count(1000000);
    }
    return SpectraSchema{std::move(spectra)};
}
bool boolean(detail::Decoder &d) {
    auto value = d.integer();
    if (value > 1)
        throw std::runtime_error("invalid boolean");
    return value != 0;
}
void encode_edp(detail::Encoder &e, const InverseModelMetadata &m) {
    e.integer(m.elements.size());
    for (const auto &element : m.elements)
        e.text(element);
    e.integer(m.layers);
    e.integer(m.uncertainty);
}
void decode_edp(detail::Decoder &d, InverseModelMetadata &m) {
    m.elements.resize(d.count(256));
    for (auto &element : m.elements)
        element = d.text();
    m.layers = d.count(10000);
    m.uncertainty = boolean(d);
}

void encode_tensor(detail::Encoder &e, const TensorPipeline &pipeline) {
    e.integer(pipeline.size());
    for (const auto &step : pipeline) {
        e.integer(step.index());
        std::visit(
            [&](const auto &op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, MinMaxScale>) {
                    e.reals(op.minimum);
                    e.reals(op.maximum);
                    e.real(op.low);
                    e.real(op.high);
                    e.integer(op.clip);
                } else if constexpr (std::is_same_v<T, StandardScale>) {
                    e.reals(op.mean);
                    e.reals(op.deviation);
                } else if constexpr (std::is_same_v<T, MultiplyScale>)
                    e.reals(op.factors);
                else if constexpr (std::is_same_v<T, CustomTransform>) {
                    e.text(op.identifier);
                    e.integer(op.version);
                    e.integer(op.parameters.size());
                    for (const auto &[key, values] : op.parameters) {
                        e.text(key);
                        e.reals(values);
                    }
                }
            },
            step);
    }
}
TensorPipeline decode_tensor(detail::Decoder &d) {
    TensorPipeline pipeline;
    auto count = d.count(1024);
    pipeline.reserve(count);
    for (std::size_t j = 0; j < count; ++j)
        switch (d.integer()) {
        case 0:
            pipeline.emplace_back(Log1p{});
            break;
        case 1: {
            MinMaxScale op;
            op.minimum = d.reals();
            op.maximum = d.reals();
            op.low = d.real();
            op.high = d.real();
            op.clip = boolean(d);
            pipeline.emplace_back(std::move(op));
            break;
        }
        case 2: {
            StandardScale op;
            op.mean = d.reals();
            op.deviation = d.reals();
            pipeline.emplace_back(std::move(op));
            break;
        }
        case 3:
            pipeline.emplace_back(MultiplyScale{d.reals()});
            break;
        case 4: {
            CustomTransform op;
            op.identifier = d.text();
            op.version = static_cast<std::uint32_t>(d.count(0xffffffffULL));
            auto n = d.count(1024);
            for (std::size_t k = 0; k < n; ++k) {
                auto name = d.text();
                auto values = d.reals();
                if (!op.parameters.emplace(name, std::move(values)).second)
                    throw std::runtime_error("duplicate custom parameter");
            }
            pipeline.emplace_back(std::move(op));
            break;
        }
        default:
            throw std::runtime_error("unknown preprocessing operation");
        }
    return pipeline;
}
void encode_spectrum_pipeline(detail::Encoder &e, const SpectrumPipeline &pipeline) {
    e.integer(pipeline.size());
    for (const auto &entry : pipeline) {
        e.text(entry.measurement);
        e.integer(entry.steps.size());
        for (const auto &step : entry.steps) {
            e.integer(step.index());
            std::visit(
                [&](const auto &op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, MultiplyCounts>)
                        e.real(op.factor);
                    else if constexpr (std::is_same_v<T, Rebin>) {
                        e.integer(op.channels);
                        e.integer(op.require_full_coverage);
                    } else if constexpr (std::is_same_v<T, Pileup>) {
                        e.real(op.tau_seconds);
                        e.integer(static_cast<std::uint64_t>(op.acquisition));
                    } else if constexpr (std::is_same_v<T, RemovePileup>) {
                        e.real(op.tau_seconds);
                        e.real(op.relative_tolerance);
                        e.integer(static_cast<std::uint64_t>(op.acquisition));
                    }
                },
                step);
        }
    }
}
AcquisitionSource acquisition(detail::Decoder &d) {
    auto value = d.integer();
    if (value > 1)
        throw std::runtime_error("unknown acquisition source");
    return static_cast<AcquisitionSource>(value);
}
SpectrumPipeline decode_spectrum_pipeline(detail::Decoder &d) {
    SpectrumPipeline result(d.count(1024));
    for (auto &entry : result) {
        entry.measurement = d.text();
        auto n = d.count(1024);
        entry.steps.reserve(n);
        for (std::size_t j = 0; j < n; ++j)
            switch (d.integer()) {
            case 0:
                entry.steps.emplace_back(MultiplyCounts{d.real()});
                break;
            case 1:
                entry.steps.emplace_back(ExposureScale{});
                break;
            case 2: {
                Rebin op;
                op.channels = d.count(1000000);
                op.require_full_coverage = boolean(d);
                entry.steps.emplace_back(op);
                break;
            }
            case 3: {
                Pileup op;
                op.tau_seconds = d.real();
                op.acquisition = acquisition(d);
                entry.steps.emplace_back(op);
                break;
            }
            case 4: {
                RemovePileup op;
                op.tau_seconds = d.real();
                op.relative_tolerance = d.real();
                op.acquisition = acquisition(d);
                entry.steps.emplace_back(op);
                break;
            }
            default:
                throw std::runtime_error("unknown spectrum operation");
            }
    }
    return result;
}
void encode(detail::Encoder &e, const ModelMetadata &m) {
    e.integer(m.direction() == Direction::Forward ? 0 : 1);
    if (const auto *f = std::get_if<ForwardModelMetadata>(&m.model)) {
        detail::encode(e, f->reference);
        encode_parameters(e, f->input_parameters);
        encode_spectra(e, f->output_spectra);
        encode_tensor(e, f->preprocessing);
        encode_spectrum_pipeline(e, f->postprocessing);
    } else {
        const auto &i = std::get<InverseModelMetadata>(m.model);
        detail::encode(e, i.reference_setup);
        encode_spectra(e, i.input_spectra);
        encode_edp(e, i);
        encode_spectrum_pipeline(e, i.input_preparation);
        encode_tensor(e, i.preprocessing);
        e.integer(i.postprocessing.size());
        for (const auto &step : i.postprocessing)
            e.real(step.factor);
    }
    e.text(m.input_name);
    e.text(m.output_name);
    e.text(m.provenance);
}
ModelMetadata decode(detail::Decoder &d) {
    ModelMetadata m;
    auto direction = d.integer();
    if (direction == 0) {
        ForwardModelMetadata f;
        f.reference = detail::experiment(d);
        f.input_parameters = decode_parameters(d);
        f.output_spectra = decode_spectra(d);
        f.preprocessing = decode_tensor(d);
        f.postprocessing = decode_spectrum_pipeline(d);
        m.model = std::move(f);
    } else if (direction == 1) {
        InverseModelMetadata i;
        i.reference_setup = detail::setup(d);
        i.input_spectra = decode_spectra(d);
        decode_edp(d, i);
        i.input_preparation = decode_spectrum_pipeline(d);
        i.preprocessing = decode_tensor(d);
        i.postprocessing.resize(d.count(1024));
        for (auto &step : i.postprocessing)
            step.factor = d.real();
        m.model = std::move(i);
    } else
        throw std::runtime_error("unknown model direction");
    m.input_name = d.text();
    m.output_name = d.text();
    m.provenance = d.text();
    m.validate();
    return m;
}
} // namespace
void Package::validate() const {
    metadata.validate();
    if (graph.empty() || graph.size() > detail::max_payload - 1024)
        throw std::invalid_argument("invalid graph size");
}
Package Package::from_onnx(const std::filesystem::path &path, ModelMetadata metadata) {
    auto size = std::filesystem::file_size(path);
    if (size == 0 || size > detail::max_payload - 1024)
        throw std::invalid_argument("invalid ONNX file size");
    Package p;
    p.metadata = std::move(metadata);
    p.graph.resize(static_cast<std::size_t>(size));
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(p.graph.data()), static_cast<std::streamsize>(size));
    if (!f)
        throw std::runtime_error("cannot read ONNX graph");
    p.validate();
    return p;
}
void Package::write(const std::filesystem::path &path) const {
    validate();
    if (std::filesystem::exists(path))
        throw std::runtime_error("package already exists");
    detail::Encoder e;
    encode(e, metadata);
    e.integer(graph.size());
    e.bytes.insert(e.bytes.end(), graph.begin(), graph.end());
    auto temp = path;
    temp += ".tmp";
    detail::write_envelope(temp, "IBMODEL1", e);
    try {
        std::filesystem::rename(temp, path);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(temp, ec);
        throw;
    }
}
Package Package::open(const std::filesystem::path &path) {
    auto bytes = detail::read_envelope(path, "IBMODEL1");
    detail::Decoder d{bytes};
    Package p;
    p.metadata = decode(d);
    auto size = d.count(detail::max_payload);
    if (size != bytes.size() - d.offset)
        throw std::runtime_error("graph payload size mismatch");
    p.graph.assign(bytes.begin() + static_cast<std::ptrdiff_t>(d.offset), bytes.end());
    p.validate();
    return p;
}
} // namespace ibeamlab::model
