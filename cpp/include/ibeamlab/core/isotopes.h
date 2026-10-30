#pragma once
#include <ibeamlab/core/sample.h>
#include <span>
#include <string_view>

namespace ibeamlab::core {
/** @brief Database identity and atomic mass (u); sample fractions live in Isotope. */
struct IsotopeData { std::int32_t massNumber; double exactMass; };
/** @brief Immutable supported isotopes; throws for unknown element symbols. */
IBEAMLAB_API std::span<const IsotopeData> isotope_data(std::string_view element);
/** @brief Immutable normalized natural fractions. Empty means no defined default. */
IBEAMLAB_API std::span<const Isotope> natural_isotopes(std::string_view element);
/** @brief Resolve a sample's empty list to the shared natural composition, without allocation. */
IBEAMLAB_API std::span<const Isotope> resolved_isotopes(const Species &species);
/** @brief Validate identities, uniqueness, finite fractions and normalization. */
IBEAMLAB_API void validate_isotopes(std::string_view element, std::span<const Isotope> isotopes);
/** @brief Compare validated compositions by identity, independently of order; no allocations. */
IBEAMLAB_API bool same_isotopes(std::span<const Isotope> a, std::span<const Isotope> b,
                                double tolerance = 1e-8) noexcept;
} // namespace ibeamlab::core
