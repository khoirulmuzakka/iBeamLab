#pragma once

#include <ibeamlab/simulator.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ibeamlab::simulator {

struct SimnraMethod {
    std::string label;
    std::filesystem::path referenceFile;
};

struct SimnraSimulatorConfig {
    std::vector<SimnraMethod> methods;
    std::size_t workers{1};
    bool multithreadedApartment{true};
    int threadPriority{0};
    bool fastCalculation{false};
};

class SimnraSimulator final : public ISimulator {
public:
    explicit SimnraSimulator(SimnraSimulatorConfig config);
    ~SimnraSimulator() override;
    SimnraSimulator(const SimnraSimulator&) = delete;
    SimnraSimulator& operator=(const SimnraSimulator&) = delete;
    std::vector<SimulationResult> simulateBatch(const std::vector<SimulationInput>& inputs,
        const SimulationOptions& options = {}) override;
    std::string configurationToml() const override;
    SimulationInput inspectConfiguration(const SimulationInput& input,
        const std::string& methodLabel);
    void requestStop() noexcept override;
    void resetStop() noexcept override;
    void close() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ibeamlab::simulator
