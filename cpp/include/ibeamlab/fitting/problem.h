#pragma once
#include <ibeamlab/core/data.h>
#include <ibeamlab/core/parameter.h>
#include <variant>
namespace ibeamlab::fitting {
/** @brief Derived per-element layer areal density, in 1e15 atoms/cm2. */
struct ElementArealDensity {
    std::size_t layer = 0;
    std::string element;
};
using Target = std::variant<core::ParameterTarget, ElementArealDensity>;
/** @brief One bounded physical coordinate, optionally tied to equivalent targets.
 * Equal bounds denote a fixed value. The initial sample/setup must satisfy bounds.
 * Tied targets must have the same physical target kind and initial value.
 */
struct Parameter {
    std::string name;
    Target target;
    core::ParameterBounds bounds{0, 1};
    std::vector<Target> tied_targets;
};
/** @brief Bounded composition group represented by N-1 unit-interval coordinates.
 * A sequential bounded stick-breaking map preserves normalization and physical
 * component bounds. Elements outside the group retain their reference fractions.
 */
struct Composition {
    std::size_t layer = 0;
    std::vector<std::string> elements;
    std::vector<core::ParameterBounds> bounds;
    std::string name_prefix;
};
/** @brief Zero-based half-open channel interval with a nonnegative weight. */
struct Region {
    std::size_t start = 0, stop = 0;
    double weight = 1;
};
/** @brief Per-measurement objective weight and disjoint channel selections. */
struct Selection {
    std::string measurement;
    double weight = 1;
    std::vector<Region> regions;
};
enum class Loss { DataChiSquare, VarianceChiSquare };
/** @brief Fixed data, initial sample and optimizer-independent fitting contract. */
struct Problem {
    core::MeasuredData data;
    core::Sample initial_sample;
    std::vector<Parameter> parameters;
    std::vector<Composition> compositions;
    std::vector<Selection> selections;
    Loss loss = Loss::DataChiSquare;
    double data_floor = 1;
};
} // namespace ibeamlab::fitting
