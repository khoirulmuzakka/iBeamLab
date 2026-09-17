#pragma once

/**
 * @file transforms.h
 * @brief Reversible matrix transforms used by packaged model preprocessing.
 *
 * These runtime transforms reproduce fitted training transformations during
 * native inference; fitting the transform parameters remains a Python task.
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ibeamlab::preprocessing {

/** @brief Row-major float matrix exchanged with ONNX preprocessing. */
using Matrix = std::vector<std::vector<float>>;

/** @brief Abstract immutable matrix transform. */
class Transform {
public:
    virtual ~Transform() = default;
    virtual Matrix apply(const Matrix& input) const = 0;
    virtual Matrix inverse(const Matrix& input) const;
    virtual std::string type() const = 0;
    virtual std::size_t inputDimension() const noexcept = 0;
};

/** @brief No-op transform that still enforces the expected dimension. */
class IdentityTransform final : public Transform {
public:
    explicit IdentityTransform(std::size_t dimension) : dimension_(dimension) {}
    Matrix apply(const Matrix& input) const override;
    Matrix inverse(const Matrix& input) const override;
    std::string type() const override { return "identity"; }
    std::size_t inputDimension() const noexcept override { return dimension_; }
private: std::size_t dimension_;
};

/** @brief Multiplies values by a fixed factor and supports inversion. */
class ConstantFactorTransform final : public Transform {
public:
    ConstantFactorTransform(std::size_t dimension, float factor);
    Matrix apply(const Matrix& input) const override;
    Matrix inverse(const Matrix& input) const override;
    std::string type() const override { return "constant_factor"; }
    std::size_t inputDimension() const noexcept override { return dimension_; }
    float factor() const noexcept { return factor_; }
private: std::size_t dimension_; float factor_;
};

/** @brief Applies a fitted per-feature mean and standard deviation. */
class StandardScaler final : public Transform {
public:
    StandardScaler(std::vector<float> mean, std::vector<float> deviation);
    Matrix apply(const Matrix& input) const override;
    Matrix inverse(const Matrix& input) const override;
    std::string type() const override { return "standard_scaler"; }
    std::size_t inputDimension() const noexcept override { return mean_.size(); }
    const auto& mean() const noexcept { return mean_; }
    const auto& deviation() const noexcept { return deviation_; }
private: std::vector<float> mean_, deviation_;
};

/** @brief Applies a fitted per-feature affine min/max mapping. */
class MinMaxScaler final : public Transform {
public:
    MinMaxScaler(std::vector<float> minimum, std::vector<float> scale,
                 float low = 0.0F, float high = 1.0F);
    Matrix apply(const Matrix& input) const override;
    Matrix inverse(const Matrix& input) const override;
    std::string type() const override { return "min_max_scaler"; }
    std::size_t inputDimension() const noexcept override { return minimum_.size(); }
    const auto& minimum() const noexcept { return minimum_; }
    const auto& scale() const noexcept { return scale_; }
    float low() const noexcept { return low_; } float high() const noexcept { return high_; }
private: std::vector<float> minimum_, scale_; float low_, high_;
};

/** @brief Applies an offset natural logarithm and its inverse. */
class LogTransform final : public Transform {
public:
    LogTransform(std::size_t dimension, float offset = 1.0F);
    Matrix apply(const Matrix& input) const override;
    Matrix inverse(const Matrix& input) const override;
    std::string type() const override { return "log"; }
    std::size_t inputDimension() const noexcept override { return dimension_; }
    float offset() const noexcept { return offset_; }
private: std::size_t dimension_; float offset_;
};

/** @brief Applies an ordered sequence of transforms and reverses it safely. */
class TransformPipeline final : public Transform {
public:
    void add(std::shared_ptr<const Transform> transform);
    Matrix apply(const Matrix& input) const override;
    Matrix inverse(const Matrix& input) const override;
    std::string type() const override { return "pipeline"; }
    std::size_t inputDimension() const noexcept override;
    const auto& transforms() const noexcept { return transforms_; }
private: std::vector<std::shared_ptr<const Transform>> transforms_;
};

} // namespace ibeamlab::preprocessing
