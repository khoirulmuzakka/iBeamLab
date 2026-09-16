#pragma once

#include <ibeamlab/sample.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ibeamlab::simulator {

using Counts = std::vector<float>;

struct SimulationInput {
    sample::SampleModel sample;
    sample::ExperimentalSetup setup;
};

struct Spectrum {
    std::string label;
    Counts counts;
};

struct SimulationFailure {
    std::size_t sampleIndex{};
    std::string methodLabel;
    std::string type;
    std::string message;
};

struct SimulationResult {
    std::vector<Spectrum> spectra;
    std::map<std::string, std::map<std::string, Counts>> elementalSpectra;
    std::optional<SimulationFailure> failure;
};

struct SimulationOptions {
    bool captureElementalSpectra{false};
    bool continueAfterFailure{false};
};

class ISimulator {
public:
    virtual ~ISimulator() = default;
    virtual std::vector<SimulationResult> simulateBatch(
        const std::vector<SimulationInput>& inputs,
        const SimulationOptions& options = {}) = 0;
    virtual void requestStop() noexcept { stopRequested_.store(true); }
    virtual void resetStop() noexcept { stopRequested_.store(false); }

protected:
    std::atomic_bool stopRequested_{false};
};

class DummySimulator final : public ISimulator {
public:
    explicit DummySimulator(std::size_t channels = 1024) : channels_(channels) {}
    std::vector<SimulationResult> simulateBatch(
        const std::vector<SimulationInput>& inputs,
        const SimulationOptions& options = {}) override;
private:
    std::size_t channels_;
};

} // namespace ibeamlab::simulator
