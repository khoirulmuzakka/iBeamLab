#pragma once
#include <cstdint>
#include <ibeamlab/simulation/simulator.h>
#include <limits>
namespace ibeamlab::fitting {
/** @brief Per-measurement weighted sum and mean over selected observed channels. */
struct Contribution {
    std::string measurement;
    double cost = 0;
    std::size_t channels = 0;
    double mean() const { return channels ? cost / channels : 0; }
};
/** @brief Bounded scalar progress snapshot; contains no population spectra. */
struct Progress {
    std::size_t evaluations = 0, invalid = 0;
    double best_cost = std::numeric_limits<double>::infinity();
};
/** @brief Best evaluated physical solution, retained even after cancellation.
 * coordinates are optimizer coordinates; named parameter_values are physical.
 * The sample, setup and spectra are snapshots of the same evaluated solution.
 */
struct Result {
    core::Experiment experiment;
    simulation::Result prediction;
    core::Counts coordinates, parameter_values;
    std::vector<std::string> parameter_names;
    std::vector<Contribution> contributions;
    std::vector<Progress> history;
    double cost = std::numeric_limits<double>::infinity();
    std::size_t evaluations = 0, invalid = 0;
    std::string status = "no_feasible_solution", message;
    bool feasible() const { return std::isfinite(cost); }
};
} // namespace ibeamlab::fitting
