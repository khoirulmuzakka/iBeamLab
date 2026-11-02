#pragma once
#include <ibeamlab/core/parameter.h>
#include <ibeamlab/profile/discretization.h>
namespace ibeamlab::fitting {
/** Bounded coefficient in a macro layer/species function (all indices zero-based).
 * Coordinates retain their physical coefficient units; bounds may be fixed. */
struct ProfileLink {
    std::size_t layer = 0, species = 0, coefficient = 0;
    double ratio = 1;
};
struct ProfileParameter {
    std::string name;
    std::size_t layer = 0, species = 0, coefficient = 0;
    core::ParameterBounds bounds{0, 1};
    std::vector<ProfileLink> links;
};
/** Compiled profile coefficients and sublayer topology. Macro sample parameters
 * may vary independently of these coefficients. Use one instance per objective. */
class IBEAMLAB_API Parameterization {
    profile::Model model_;
    profile::Discretizer discretizer_;
    std::vector<ProfileParameter> parameters_;

  public:
    Parameterization(profile::Model, std::vector<ProfileParameter>);
    void apply(std::span<const double> coefficients, const core::Sample &macro, core::Sample &out);
};
} // namespace ibeamlab::fitting
