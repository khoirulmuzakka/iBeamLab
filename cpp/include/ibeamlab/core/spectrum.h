#pragma once
#include <cmath>
#include <ibeamlab/core/experiment.h>
#include <span>
#include <stdexcept>
#include <string_view>

namespace ibeamlab::core {
// Counts are integrated per channel (not density); double accumulation avoids
// precision loss in repeated corrections. Views never extend owner lifetimes.
using Counts = std::vector<double>;
/** @brief Read-only, non-owning labelled spectrum; the label and counts must outlive use. */
struct SpectrumView {
    std::string_view measurement;
    std::span<const double> counts;
};
/** @brief Owning spectrum of integrated channel counts for one measurement label. */
struct Spectrum {
    std::string measurement;
    Counts counts;
    SpectrumView view() const { return {measurement, counts}; }
};
inline void validate_counts(std::span<const double> counts) {
    for (double value : counts)
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("counts must be finite and nonnegative");
}
// Calibration describes channel centers E(i)=offset+linear*i+quadratic*i*i.
// Edges use half-integer indices. Increasing finite edges are required.
inline void channel_edges(const Detector &detector, std::span<double> edges) {
    if (edges.size() < 2)
        throw std::invalid_argument("at least two channel edges required");
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const double x = static_cast<double>(i) - .5;
        edges[i] = detector.calibrationOffset +
                   x * (detector.calibrationLinear + x * detector.calibrationQuadratic);
        if (!std::isfinite(edges[i]) || (i && edges[i] <= edges[i - 1]))
            throw std::invalid_argument("channel edges must be finite and increasing");
    }
}
inline const Measurement &find_measurement(const ExperimentalSetup &setup, std::string_view label) {
    for (const auto &measurement : setup.measurements)
        if (measurement.label == label)
            return measurement;
    throw std::invalid_argument("unknown measurement: " + std::string(label));
}
inline const Measurement &find_measurement(const Experiment &experiment, std::string_view label) {
    return find_measurement(experiment.setup, label);
}
} // namespace ibeamlab::core
