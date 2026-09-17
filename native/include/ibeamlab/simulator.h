#pragma once

/**
 * @file simulator.h
 * @brief Backend-neutral interfaces and value types for spectrum simulation.
 *
 * Simulator implementations consume fully materialized physical inputs. They
 * do not define sampling distributions or training behavior.
 */

#include <ibeamlab/sample.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ibeamlab::simulator {

/** @brief Single-precision channel counts used at simulator boundaries. */
using Counts = std::vector<float>;

/** @brief Complete sample and experimental setup for one simulation. */
struct SimulationInput {
    sample::SampleModel sample;
    sample::ExperimentalSetup setup;
};

/** @brief One labeled simulated spectrum. */
struct Spectrum {
    std::string label;
    Counts counts;
};

/** @brief Structured failure associated with one input and optional method. */
struct SimulationFailure {
    std::size_t sampleIndex{};
    std::string methodLabel;
    std::string type;
    std::string message;
};

/** @brief Successful spectra or a structured failure for one input. */
struct SimulationResult {
    std::vector<Spectrum> spectra;
    std::map<std::string, std::map<std::string, Counts>> elementalSpectra;
    std::optional<SimulationFailure> failure;
};

/** @brief Backend-neutral options for one batch request. */
struct SimulationOptions {
    bool captureElementalSpectra{false};
    bool continueAfterFailure{false};
};

/**
 * @brief Abstract batch simulator implemented by concrete calculation backends.
 *
 * Implementations are responsible for stable batch ordering, cooperative
 * cancellation, and structured failures.
 */
class ISimulator {
public:
    virtual ~ISimulator() = default;
    /** @brief Simulates all inputs and returns results in input order. */
    virtual std::vector<SimulationResult> simulateBatch(
        const std::vector<SimulationInput>& inputs,
        const SimulationOptions& options = {}) = 0;
    /** @brief Returns reproducibility metadata for the concrete backend. */
    virtual std::string configurationToml() const { return {}; }
    /** @brief Cooperatively requests cancellation. */
    virtual void requestStop() noexcept { stopRequested_.store(true); }
    /** @brief Clears a previous cancellation request before a new run. */
    virtual void resetStop() noexcept { stopRequested_.store(false); }

protected:
    std::atomic_bool stopRequested_{false};
};

/** @brief Deterministic dependency-free simulator used by tests and examples. */
class DummySimulator final : public ISimulator {
public:
    explicit DummySimulator(std::size_t channels = 1024) : channels_(channels) {}
    std::vector<SimulationResult> simulateBatch(
        const std::vector<SimulationInput>& inputs,
        const SimulationOptions& options = {}) override;
    std::string configurationToml() const override;
private:
    std::size_t channels_;
};

} // namespace ibeamlab::simulator
