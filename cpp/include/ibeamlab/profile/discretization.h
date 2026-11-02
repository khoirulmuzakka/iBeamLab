#pragma once
#include <ibeamlab/profile/model.h>
namespace ibeamlab::profile {
/** Retained normalized edges, compiled once per profile topology. One instance
 * per caller. Coefficients and physical thickness may change; topology may not.
 * Output must not alias model reference layers. Existing species buffers are reused. */
class IBEAMLAB_API Discretizer {
    std::vector<core::Counts> edges_;
    std::vector<std::size_t> species_;
    std::vector<Spacing> spacing_;
    core::Sample identity_;

  public:
    explicit Discretizer(const Model &);
    void apply(const Model &, core::Sample &output) const;
};
IBEAMLAB_API core::Sample discretize(const Model &);
} // namespace ibeamlab::profile
