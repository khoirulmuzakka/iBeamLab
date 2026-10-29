#pragma once
#include <atomic>
#include <functional>
#include <ibeamlab/core/spectrum.h>
#include <optional>
namespace ibeamlab::simulation {
/** @brief Backend failure diagnostic, optionally identifying the affected measurement. */
struct Failure {
    std::string measurement;
    std::string message;
};
/** @brief Labelled spectra from one experiment, or a backend failure diagnostic. */
struct Result {
    std::vector<core::Spectrum> spectra;
    std::optional<Failure> failure;
};
/** @brief Batch progress callback receiving completed and total experiment counts. */
using Progress = std::function<void(std::size_t completed, std::size_t total)>;
/** @brief Backend interface for simulating all measurements of a shared sample.
 * Calls on an instance are serialized by its caller. Cancellation uses an atomic
 * stop flag and is cooperative; reset_stop() permits subsequent work.
 */
class IBEAMLAB_API Simulator {
  public:
    virtual ~Simulator() = default;
    // Calls on one instance are serialized by the caller. Each worker owns a
    // separate backend instance. Output storage may be reused between calls.
    virtual void simulate(const core::Experiment &, Result &output) = 0;
    virtual std::string provenance() const = 0;
    virtual void request_stop() noexcept { stop_.store(true); }
    virtual void reset_stop() noexcept { stop_.store(false); }
    bool stopped() const noexcept { return stop_.load(); }
    virtual std::vector<Result> simulate_batch(std::span<const core::Experiment> experiments,
                                               Progress progress = {});

  protected:
    std::atomic_bool stop_{false};
};
/** @brief Synthetic backend for tests and workflow checks; not a physical simulator. */
class IBEAMLAB_API DummySimulator final : public Simulator {
  public:
    explicit DummySimulator(std::size_t channels = 5000);
    void simulate(const core::Experiment &, Result &) override;
    std::string provenance() const override;

  private:
    std::size_t channels_;
};
} // namespace ibeamlab::simulation
