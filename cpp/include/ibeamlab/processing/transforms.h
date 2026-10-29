#pragma once
#include <functional>
#include <ibeamlab/model/metadata.h>
namespace ibeamlab::processing {
/** @brief Application-supplied transform operating in place at fixed tensor width.
 * Kernels must be safe for concurrent workers and must not retain the input span.
 */
using CustomTransformKernel =
    std::function<void(std::span<float>, const model::CustomParameters &)>;
/** @brief Register a process-wide kernel; duplicate identifier/version pairs are rejected. */
IBEAMLAB_API void register_custom_transform(std::string identifier, std::uint32_t version,
                                            CustomTransformKernel);
/** @brief Throw if a pipeline refers to an unregistered custom implementation. */
IBEAMLAB_API void require_supported_transforms(const model::TensorPipeline &);
/** @brief Execute ordered transforms in place, without refitting stored coefficients. */
IBEAMLAB_API void apply_preprocessing(std::span<float>, const model::TensorPipeline &);
} // namespace ibeamlab::processing
