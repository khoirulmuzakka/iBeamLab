#pragma once
#include <ibeamlab/datasets/dataset.h>
#include <random>
namespace ibeamlab::generation {
/** @brief On simulation failure, stop generation, store the failure, or omit the row. */
enum class FailurePolicy { Stop, Record, Discard };
/** Reference experiment and ordered physical parameters varied during generation. */
struct IBEAMLAB_API GenerationPlan {
    core::Experiment reference;
    std::vector<core::ParameterSpec> parameters;
    void validate() const;
};
/** @brief Outcome counters and cancellation status for one generation run. */
struct Summary {
    std::uint64_t accepted = 0, failed = 0, discarded = 0;
    bool cancelled = false;
};
// Row-major caller-owned physical values; no implicit concentration normalization.
// Cancellation/Stop leave an explicitly incomplete dataset, never a complete subset.
IBEAMLAB_API Summary generate(const GenerationPlan &, std::span<const double> rows,
                              std::size_t samples, simulation::Simulator &,
                              const std::filesystem::path &, FailurePolicy = FailurePolicy::Stop,
                              simulation::Progress = {}, std::string provenance = {},
                              std::size_t records_per_shard = 64);
/** @brief Independent uniform sampling interval for one physical parameter column. */
struct Uniform {
    core::ParameterBounds bounds;
};
// Independent uniform sampler. For concentrations supply an explicitly coupled
// external sampler instead; generated rows are validated by generate().
IBEAMLAB_API void sample_uniform(std::span<const Uniform>, std::size_t samples, std::uint64_t seed,
                                 std::span<double> output);
} // namespace ibeamlab::generation
