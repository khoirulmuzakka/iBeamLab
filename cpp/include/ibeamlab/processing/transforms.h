#pragma once
#include <functional>
#include <memory>
#include <ibeamlab/model/metadata.h>
namespace ibeamlab::processing {
/** @brief Application-supplied transform operating in place at fixed tensor width.
 * Kernels must be safe for concurrent workers and must not retain the input span.
 */
using CustomTransformKernel =
    std::function<void(std::span<float>, const model::CustomParameters &)>;
/** @brief Register a process-wide kernel; duplicate identifier/version pairs are rejected. */
IBEAMLAB_API void register_custom_transform(std::string identifier, std::uint32_t version,
                                            CustomTransformKernel, CustomTransformKernel inverse = {});
/** @brief Own fitted coefficients and resolve kernels once; reuse scratch on repeated calls.
 * Graph-embedded stages are identities here. Inverse stages reverse operation
 * order. Affine uncertainty maps std by absolute scale and leaves probability
 * untouched; nonlinear uncertainty transformations are rejected at loading.
 */
class IBEAMLAB_API TransformPlan {
  public:
    TransformPlan(const model::TransformPipeline &, std::size_t width, bool uncertainty = false);
    ~TransformPlan();
    TransformPlan(TransformPlan &&) noexcept;
    TransformPlan &operator=(TransformPlan &&) noexcept;
    bool empty() const noexcept;
    void apply(std::span<float>);
    void apply(std::span<double>);
    void apply_uncertainty(std::span<double> means, std::span<double> standard_deviations);
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
/** @brief Throw if a pipeline refers to an unregistered custom implementation. */
IBEAMLAB_API void require_supported_transforms(const model::TensorPipeline &);
/** @brief Execute ordered transforms in place, without refitting stored coefficients. */
IBEAMLAB_API void apply_preprocessing(std::span<float>, const model::TensorPipeline &);
} // namespace ibeamlab::processing
