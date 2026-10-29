#pragma once
#include <string_view>
namespace ibeamlab::core::units {
inline constexpr std::string_view energy = "keV";
inline constexpr std::string_view arealDensity = "1e15 atoms/cm2";
inline constexpr std::string_view time = "s";
inline constexpr double electronVoltsPerKeV = 1000.0;
inline constexpr double atomsPerSquareCentimeterPerArealDensity = 1e15;
inline constexpr double fractionTolerance = 1e-6;
} // namespace ibeamlab::core::units
