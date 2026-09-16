#include <ibeamlab/transforms.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ibeamlab::preprocessing {
namespace {
void validate(const Matrix& matrix, std::size_t dimension) {
    for (const auto& row : matrix) {
        if (row.size() != dimension) throw std::invalid_argument("transform input dimension mismatch");
        for (float value : row) if (!std::isfinite(value)) throw std::invalid_argument("transform input is not finite");
    }
}
template<class Function> Matrix mapped(const Matrix& input, std::size_t dimension, Function function) {
    validate(input, dimension); Matrix output = input;
    for (auto& row : output) for (std::size_t column = 0; column < dimension; ++column)
        row[column] = function(row[column], column);
    return output;
}
}

Matrix Transform::inverse(const Matrix&) const { throw std::logic_error(type() + " has no inverse"); }
Matrix IdentityTransform::apply(const Matrix& input) const { validate(input, dimension_); return input; }
Matrix IdentityTransform::inverse(const Matrix& input) const { return apply(input); }
ConstantFactorTransform::ConstantFactorTransform(std::size_t dimension, float factor)
    : dimension_(dimension), factor_(factor) { if (!std::isfinite(factor) || factor == 0) throw std::invalid_argument("factor must be finite and nonzero"); }
Matrix ConstantFactorTransform::apply(const Matrix& in) const { return mapped(in, dimension_, [&](float v, std::size_t){ return v * factor_; }); }
Matrix ConstantFactorTransform::inverse(const Matrix& in) const { return mapped(in, dimension_, [&](float v, std::size_t){ return v / factor_; }); }
StandardScaler::StandardScaler(std::vector<float> mean, std::vector<float> deviation)
    : mean_(std::move(mean)), deviation_(std::move(deviation)) {
    if (mean_.empty() || mean_.size() != deviation_.size()) throw std::invalid_argument("invalid standard scaler arrays");
    for (float value : deviation_) if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("standard deviation must be positive");
}
Matrix StandardScaler::apply(const Matrix& in) const { return mapped(in, mean_.size(), [&](float v, std::size_t c){ return (v - mean_[c]) / deviation_[c]; }); }
Matrix StandardScaler::inverse(const Matrix& in) const { return mapped(in, mean_.size(), [&](float v, std::size_t c){ return v * deviation_[c] + mean_[c]; }); }
MinMaxScaler::MinMaxScaler(std::vector<float> minimum, std::vector<float> scale, float low, float high)
    : minimum_(std::move(minimum)), scale_(std::move(scale)), low_(low), high_(high) {
    if (minimum_.empty() || minimum_.size() != scale_.size() || !(high > low)) throw std::invalid_argument("invalid min-max scaler");
    for (float value : scale_) if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("min-max scale must be positive");
}
Matrix MinMaxScaler::apply(const Matrix& in) const { return mapped(in, minimum_.size(), [&](float v, std::size_t c){ return low_ + (v - minimum_[c]) / scale_[c] * (high_ - low_); }); }
Matrix MinMaxScaler::inverse(const Matrix& in) const { return mapped(in, minimum_.size(), [&](float v, std::size_t c){ return minimum_[c] + (v - low_) / (high_ - low_) * scale_[c]; }); }
LogTransform::LogTransform(std::size_t dimension, float offset)
    : dimension_(dimension), offset_(offset) {
    if (dimension == 0 || !std::isfinite(offset))
        throw std::invalid_argument("log transform dimension and offset are invalid");
}
Matrix LogTransform::apply(const Matrix& input) const {
    validate(input, dimension_); Matrix out = input;
    for (auto& row : out) for (auto& value : row) {
        if (value + offset_ <= 0) throw std::domain_error("log transform input is outside its domain");
        value = std::log(value + offset_);
    }
    return out;
}
Matrix LogTransform::inverse(const Matrix& input) const {
    validate(input, dimension_); Matrix out = input;
    for (auto& row : out) for (auto& value : row) value = std::exp(value) - offset_;
    return out;
}

void TransformPipeline::add(std::shared_ptr<const Transform> transform) {
    if (!transform) throw std::invalid_argument("pipeline transform is null");
    if (!transforms_.empty() && transforms_.back()->inputDimension() != transform->inputDimension())
        throw std::invalid_argument("pipeline transform dimensions differ");
    transforms_.push_back(std::move(transform));
}
Matrix TransformPipeline::apply(const Matrix& input) const { Matrix value = input; for (const auto& transform : transforms_) value = transform->apply(value); return value; }
Matrix TransformPipeline::inverse(const Matrix& input) const { Matrix value = input; for (auto it = transforms_.rbegin(); it != transforms_.rend(); ++it) value = (*it)->inverse(value); return value; }
std::size_t TransformPipeline::inputDimension() const noexcept { return transforms_.empty() ? 0 : transforms_.front()->inputDimension(); }

} // namespace ibeamlab::preprocessing
