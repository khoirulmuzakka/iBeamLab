#pragma once
#include <ibeamlab/core/sample.h>
#include <ibeamlab/profile/functions.h>
namespace ibeamlab::profile {
enum class Spacing { Linear, Logarithmic };
/** Macro layer with species identity/isotopes and one profile per species.
 * Profiles are nonnegative weights normalized at each sublayer midpoint.
 * Logarithmic spacing follows base-4 edges from exponents -3 to 0, shifted
 * to zero with the final edge restored to the complete layer thickness. */
struct Layer {
    core::Layer reference;
    std::vector<Function> functions;
    std::size_t sublayers = 1;
    Spacing spacing = Spacing::Linear;
};
struct IBEAMLAB_API Model {
    std::vector<Layer> layers;
    void validate() const;
    core::Sample reference_sample() const;
};
} // namespace ibeamlab::profile
