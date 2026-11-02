#pragma once
#include <functional>
#include <ibeamlab/fitting/objective.h>
#include <map>
#include <variant>
namespace ibeamlab::fitting {
/** @brief ARRDE execution settings; observations and objective remain solver independent. */
using OptimizerSetting = std::variant<bool, int, double, std::string>;
struct Options {
    std::size_t max_evaluations = 10000, population_size = 0;
    int seed = 42;
    bool record_history = true;
    std::string algorithm = "ARRDE"; ///< Built-in ARRDE or DE; external solvers use Objective.
    std::map<std::string, OptimizerSetting> settings;
    std::vector<core::Counts> initial_guesses;
};
using ProgressCallback = std::function<bool(const Progress &)>;
IBEAMLAB_API bool arrde_available() noexcept;
/** @brief Optimize using upstream ARRDE; callbacks return true to stop gracefully.
 * max_evaluations includes the initial evaluation performed by Objective.
 * Cancellation preserves the best already evaluated prediction without rerunning it.
 * External optimizers may call Objective directly and retrieve best_result().
 */
IBEAMLAB_API Result fit(Objective &, Options = {}, ProgressCallback = {});
/** Run stages in order. Each stage budget is additional; initial preflight is
 * included only in the first stage. Best coordinates seed the next stage. */
IBEAMLAB_API Result fit_pipeline(Objective &, std::span<const Options>, ProgressCallback = {});
} // namespace ibeamlab::fitting
