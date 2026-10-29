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
};
std::mutex mutex;
std::vector<Entry> registry;
std::shared_ptr<const CustomTransformKernel> find(const model::CustomTransform &op) {
    std::lock_guard lock(mutex);
    for (const auto &e : registry)
        if (e.id == op.identifier && e.version == op.version)
            return e.kernel;
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
                               CustomTransformKernel kernel) {
    if (id.empty() || !version || !kernel)
        throw std::invalid_argument("invalid custom transform registration");
    std::lock_guard lock(mutex);
    for (auto &e : registry)
        if (e.id == id && e.version == version)
            throw std::invalid_argument("custom transform already registered");
    registry.push_back(
        {std::move(id), version, std::make_shared<const CustomTransformKernel>(std::move(kernel))});
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
} // namespace ibeamlab::processing
