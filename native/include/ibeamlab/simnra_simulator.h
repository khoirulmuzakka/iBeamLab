#pragma once

/**
 * @file simnra_simulator.h
 * @brief Windows SIMNRA COM implementation of the simulator abstraction.
 *
 * This backend owns COM apartments, worker-bound SIMNRA instances, temporary
 * reference copies, configuration caching, parallel execution, and cleanup.
 */

#include <ibeamlab/simulator.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ibeamlab::simulator {

/** @brief Maps a spectrum label to a SIMNRA reference project. */
struct SimnraMethod {
    std::string label;
    std::filesystem::path referenceFile;
};

/** @brief Construction and execution settings for the SIMNRA worker pool. */
struct SimnraSimulatorConfig {
    std::vector<SimnraMethod> methods;
    std::size_t workers{1};
    bool multithreadedApartment{true};
    int threadPriority{0};
    bool fastCalculation{false};
};

/**
 * @brief Threaded SIMNRA simulator with persistent worker-owned COM objects.
 *
 * Each worker reuses one SIMNRA object per configured method. close() and the
 * destructor release COM resources and remove private temporary files.
 */
class SimnraSimulator final : public ISimulator {
public:
    /** @brief Starts a persistent worker pool for the configured methods. */
    explicit SimnraSimulator(SimnraSimulatorConfig config);
    ~SimnraSimulator() override;
    SimnraSimulator(const SimnraSimulator&) = delete;
    SimnraSimulator& operator=(const SimnraSimulator&) = delete;
    /** @brief Simulates a batch while preserving input and method ordering. */
    std::vector<SimulationResult> simulateBatch(const std::vector<SimulationInput>& inputs,
        const SimulationOptions& options = {}) override;
    /** @brief Returns the backend configuration and reference-file provenance. */
    std::string configurationToml() const override;
    /**
     * @brief Configures SIMNRA and reads the effective sample/setup back.
     * @details Intended for integration tests and configuration diagnostics.
     */
    SimulationInput inspectConfiguration(const SimulationInput& input,
        const std::string& methodLabel);
    void requestStop() noexcept override;
    void resetStop() noexcept override;
    /** @brief Idempotently stops workers and releases all COM resources. */
    void close() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ibeamlab::simulator
