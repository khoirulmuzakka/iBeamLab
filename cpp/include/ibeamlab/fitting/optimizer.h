#pragma once
#include <functional>
#include <ibeamlab/fitting/objective.h>
namespace ibeamlab::fitting {
/** @brief ARRDE execution settings; observations and objective remain solver independent. */
struct Options {
    std::size_t max_evaluations = 10000, population_size = 0;
    int seed = 42;
    bool record_history = true;
};
using ProgressCallback = std::function<bool(const Progress &)>;
IBEAMLAB_API bool arrde_available() noexcept;
/** @brief Optimize using upstream ARRDE; callbacks return true to stop gracefully.
 * max_evaluations includes the initial evaluation performed by Objective.
 * Cancellation preserves the best already evaluated prediction without rerunning it.
 * External optimizers may call Objective directly and retrieve best_result().
 */
IBEAMLAB_API Result fit(Objective &, Options = {}, ProgressCallback = {});
} // namespace ibeamlab::fitting
