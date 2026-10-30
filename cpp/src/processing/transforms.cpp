#include <algorithm>
#include <ibeamlab/processing/transforms.h>
#include <memory>
#include <mutex>
namespace ibeamlab::processing {
namespace {
struct Entry {
    std::string id;
    std::uint32_t version;
    std::shared_ptr<const CustomTransformKernel> kernel;
    std::shared_ptr<const CustomTransformKernel> inverse;
};
std::mutex mutex;
std::vector<Entry> registry;
std::shared_ptr<const CustomTransformKernel> find(const model::CustomTransform &op, bool inverse = false) {
    std::lock_guard lock(mutex);
    for (const auto &e : registry)
        if (e.id == op.identifier && e.version == op.version) {
            auto kernel = inverse ? e.inverse : e.kernel;
            if (!kernel) throw std::invalid_argument("custom transform has no inverse: " + op.identifier);
            return kernel;
        }
    throw std::invalid_argument("unsupported custom transform: " + op.identifier + " version " +
                                std::to_string(op.version));
}
double at(const core::Counts &values, std::size_t i, std::size_t width) {
    if (values.size() != 1 && values.size() != width)
        throw std::invalid_argument("transform coefficient width mismatch");
    return values[values.size() == 1 ? 0 : i];
}
} // namespace
void register_custom_transform(std::string id, std::uint32_t version,
                               CustomTransformKernel kernel, CustomTransformKernel inverse) {
    if (id.empty() || !version || !kernel)
        throw std::invalid_argument("invalid custom transform registration");
    std::lock_guard lock(mutex);
    for (auto &e : registry)
        if (e.id == id && e.version == version)
            throw std::invalid_argument("custom transform already registered");
    registry.push_back(
        {std::move(id), version, std::make_shared<const CustomTransformKernel>(std::move(kernel)),
         inverse ? std::make_shared<const CustomTransformKernel>(std::move(inverse)) : nullptr});
}
void require_supported_transforms(const model::TensorPipeline &pipeline) {
    for (const auto &s : pipeline)
        if (const auto *c = std::get_if<model::CustomTransform>(&s))
            find(*c);
}
void apply_preprocessing(std::span<float> values, const model::TensorPipeline &pipeline) {
    for (float v : values)
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite tensor input");
    for (const auto &step : pipeline) {
        std::visit(
            [&](const auto &op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, model::CustomTransform>) {
                    auto kernel = find(op);
                    (*kernel)(values, op.parameters);
                } else
                    for (std::size_t i = 0; i < values.size(); ++i) {
                        double v = values[i];
                        if constexpr (std::is_same_v<T, model::Log1p>) {
                            if (v < 0)
                                throw std::invalid_argument("log1p requires nonnegative inputs");
                            v = std::log1p(v);
                        } else if constexpr (std::is_same_v<T, model::MinMaxScale>) {
                            double low = at(op.minimum, i, values.size()),
                                   high = at(op.maximum, i, values.size());
                            if (!std::isfinite(low) || !std::isfinite(high) || high <= low ||
                                !std::isfinite(op.low) || !std::isfinite(op.high) ||
                                op.high <= op.low)
                                throw std::invalid_argument("invalid min-max scaler");
                            v = op.low + (v - low) / (high - low) * (op.high - op.low);
                            if (op.clip)
                                v = std::clamp(v, op.low, op.high);
                        } else if constexpr (std::is_same_v<T, model::StandardScale>) {
                            double mean = at(op.mean, i, values.size()),
                                   deviation = at(op.deviation, i, values.size());
                            if (!std::isfinite(mean) || !std::isfinite(deviation) || deviation <= 0)
                                throw std::invalid_argument("invalid standard scaler");
                            v = (v - mean) / deviation;
                        } else
                            v *= at(op.factors, i, values.size());
                        values[i] = static_cast<float>(v);
                    }
            },
            step);
        for (float v : values)
            if (!std::isfinite(v))
                throw std::invalid_argument("nonfinite transform output");
    }
}
struct TransformPlan::Impl {
    struct Step {
        model::TensorTransform operation;
        std::shared_ptr<const CustomTransformKernel> kernel;
    };
    std::size_t width;
    bool inverse;
    std::vector<Step> steps;
    std::vector<float> custom_scratch;
    Impl(const model::TransformPipeline &pipeline, std::size_t count, bool uncertainty)
        : width(count), inverse(pipeline.direction == model::TransformDirection::Inverse) {
        if (pipeline.location == model::TransformLocation::Graph) return;
        steps.reserve(pipeline.steps.size());
        for (std::size_t k = 0; k < pipeline.steps.size(); ++k) {
            const auto &op = pipeline.steps[inverse ? pipeline.steps.size() - k - 1 : k];
            Step step{op, {}};
            if (const auto *custom = std::get_if<model::CustomTransform>(&op)) {
                step.kernel = find(*custom, inverse);
                if (custom_scratch.empty()) custom_scratch.resize(width);
            }
            if (uncertainty && (std::holds_alternative<model::Log1p>(op) ||
                                std::holds_alternative<model::CustomTransform>(op) ||
                (std::holds_alternative<model::MinMaxScale>(op) && std::get<model::MinMaxScale>(op).clip)))
                throw std::invalid_argument("uncertainty output transforms must be affine");
            steps.push_back(std::move(step));
        }
    }
    template<class T> void apply(std::span<T> values, std::span<double> deviation = {}) {
        if (values.size() != width || (!deviation.empty() && deviation.size() != width))
            throw std::invalid_argument("numerical transform width mismatch");
        for (const auto &step : steps) {
            std::visit([&](const auto &op) {
                using S = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<S, model::CustomTransform>) {
                    if constexpr (std::is_same_v<T, float>) (*step.kernel)(values, op.parameters);
                    else {
                        for (std::size_t i = 0; i < width; ++i) custom_scratch[i] = static_cast<float>(values[i]);
                        (*step.kernel)(custom_scratch, op.parameters);
                        std::copy(custom_scratch.begin(), custom_scratch.end(), values.begin());
                    }
                } else for (std::size_t i = 0; i < width; ++i) {
                    double v = values[i], scale = 1;
                    if constexpr (std::is_same_v<S, model::Log1p>) {
                        if (!inverse && v < 0) throw std::invalid_argument("log1p requires nonnegative input");
                        v = inverse ? std::expm1(v) : std::log1p(v);
                    } else if constexpr (std::is_same_v<S, model::MultiplyScale>) {
                        scale = at(op.factors, i, width);
                        if (inverse) {
                            if (scale == 0) throw std::invalid_argument("cannot invert zero scaling");
                            scale = 1 / scale;
                        }
                        v *= scale;
                    } else if constexpr (std::is_same_v<S, model::StandardScale>) {
                        const auto mean = at(op.mean, i, width), sd = at(op.deviation, i, width);
                        scale = inverse ? sd : 1 / sd;
                        v = inverse ? v * sd + mean : (v - mean) / sd;
                    } else {
                        const auto low = at(op.minimum, i, width), high = at(op.maximum, i, width);
                        scale = inverse ? (high - low) / (op.high - op.low) : (op.high - op.low) / (high - low);
                        v = inverse ? low + (v - op.low) * scale : op.low + (v - low) * scale;
                        if (op.clip) v = std::clamp(v, op.low, op.high);
                    }
                    values[i] = static_cast<T>(v);
                    if (!deviation.empty()) deviation[i] *= std::abs(scale);
                }
            }, step.operation);
            for (T v : values)
                if (!std::isfinite(v)) throw std::invalid_argument("nonfinite numerical transform output");
        }
    }
};
TransformPlan::TransformPlan(const model::TransformPipeline &p, std::size_t width, bool uncertainty)
    : impl_(std::make_unique<Impl>(p, width, uncertainty)) {}
TransformPlan::~TransformPlan() = default;
TransformPlan::TransformPlan(TransformPlan &&) noexcept = default;
TransformPlan &TransformPlan::operator=(TransformPlan &&) noexcept = default;
bool TransformPlan::empty() const noexcept { return impl_->steps.empty(); }
void TransformPlan::apply(std::span<float> values) { impl_->apply(values); }
void TransformPlan::apply(std::span<double> values) { impl_->apply(values); }
void TransformPlan::apply_uncertainty(std::span<double> means, std::span<double> std) { impl_->apply(means, std); }
} // namespace ibeamlab::processing
