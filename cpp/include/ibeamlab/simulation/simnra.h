#pragma once
#include <filesystem>
#include <ibeamlab/simulation/simulator.h>
#include <memory>
namespace ibeamlab::simulation {
/** @brief Bind one measurement label and particle to a SIMNRA reference file.
 * The reference supplies geometry and cross-section choices. Measurements may
 * use different particles and energies; particle changes require a matching reference.
 */
struct SimnraMethod {
    std::string measurement;
    std::filesystem::path reference;
    std::string particle;
};
/** @brief Reference mappings and COM worker configuration for the SIMNRA backend. */
struct SimnraConfig {
    std::vector<SimnraMethod> methods;
    std::size_t workers = 1;
    bool multithreaded_apartment = false;
    // Disable for diagnostics that compare rebuilding against cached updates.
    bool reuse_target_topology = true;
};
/** @brief Windows COM backend requiring a locally installed, licensed SIMNRA.
 * Worker sessions retain reference settings and reuse compatible layer topology.
 * When rebuilding, new layers are appended before old layers are removed to
 * preserve cross-section choices. Calls on one instance must be serialized.
 */
class IBEAMLAB_API SimnraSimulator final : public Simulator {
  public:
    explicit SimnraSimulator(SimnraConfig);
    ~SimnraSimulator() override;
    void simulate(const core::Experiment &, Result &) override;
    std::vector<Result> simulate_batch(std::span<const core::Experiment>, Progress = {}) override;
    std::string provenance() const override;
    /** @brief Release backend sessions explicitly; destruction also releases them. */
    void close() noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::simulation
