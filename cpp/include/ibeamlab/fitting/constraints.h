#pragma once
#include <ibeamlab/core/parameter.h>
namespace ibeamlab::fitting {
/** Additional core target receiving ratio * primary coordinate. Ratios are finite;
 * initialization and resulting physical values are validated by Objective. */
struct LinkedTarget {
    core::ParameterTarget target;
    double ratio = 1;
};
/** Layer properties available for fitting even when a predictor cannot represent
 * them. Nonzero variation requires backend support; forward layer-AD models reject it. */
enum class LayerProperty { Roughness, PorosityFraction, PoreDiameter };
struct LayerPropertyTarget {
    std::size_t layer = 0;
    LayerProperty property = LayerProperty::Roughness;
};
IBEAMLAB_API void validate_ratio(double);
} // namespace ibeamlab::fitting
