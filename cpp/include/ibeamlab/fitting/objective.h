#pragma once
#include <ibeamlab/fitting/predictor.h>
#include <ibeamlab/fitting/result.h>
#include <memory>
namespace ibeamlab::fitting {
/** @brief Compiled fitting parameterization, weights and retained batch buffers.
 * Constructor evaluates the initial physical solution once, included in counters.
 * Invalid candidate rows receive +infinity; configuration and execution failures
 * raise diagnostics. Calls must be serialized; cancellation is thread-safe.
 */
class IBEAMLAB_API Objective {
  public:
    Objective(Problem, std::shared_ptr<Predictor>);
    ~Objective();
    Objective(const Objective &) = delete;
    Objective &operator=(const Objective &) = delete;
    std::size_t dimension() const;
    std::span<const core::ParameterBounds> bounds() const;
    std::span<const std::string> coordinate_names() const;
    std::span<const double> initial_coordinates() const;
    /** @brief Score a row-major population into caller-owned costs in one backend batch.
     * Invalid rows are compacted out before prediction; costs preserve input order.
     * Capacity is retained when the active population shrinks.
     */
    void evaluate_batch(std::span<const double> candidates, std::size_t population,
                        std::span<double> costs);
    /** @brief Decode one optimizer coordinate vector into a physical experiment. */
    core::Experiment decode(std::span<const double>) const;
    Result best_result() const;
    Progress progress() const;
    void request_stop() noexcept;
    bool stopped() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::fitting
