#pragma once

/**
 * @file data_generator.h
 * @brief Executes externally planned simulations and stores their results.
 *
 * Sampling policy belongs to Python. This API accepts complete open-parameter
 * rows, validates/materializes them through GenerationConfig, invokes an
 * ISimulator in bounded batches, and writes a native dataset.
 */

#include <ibeamlab/datasets.h>
#include <ibeamlab/generation.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ibeamlab::generation {

/** @brief Selects how generation handles a simulator failure. */
enum class FailurePolicy { Stop, Record, Discard };

/** @brief Runtime and provenance options for native dataset generation. */
struct GenerationOptions {
    std::size_t batchSize{1024};
    std::size_t shardCount{1};
    std::uint64_t seed{};
    std::string sampler{"external"};
    std::uint32_t samplerVersion{1};
    std::string samplingConfigToml;
    FailurePolicy failurePolicy{FailurePolicy::Stop};
};
/** @brief Snapshot reported after a completed simulation batch. */
struct GenerationProgress { std::size_t attempted{}, accepted{}, invalid{}, failed{}, total{}; };
/** @brief Final counters and cancellation state for a generation run. */
struct GenerationSummary { std::size_t requested{}, accepted{}, invalid{}, failed{}; bool cancelled{}; };
/** @brief Optional callback invoked after each native batch. */
using ProgressCallback = std::function<void(const GenerationProgress&)>;

/**
 * @brief Connects externally sampled parameters to a simulator and dataset writer.
 *
 * DataGenerator deliberately does not choose parameter distributions. Its
 * responsibility is validation, materialization, bounded execution, failure
 * handling, progress reporting, and durable dataset output.
 */
class DataGenerator {
public:
    /** @brief Creates a generator using a fixed schema and simulator backend. */
    DataGenerator(GenerationConfig config, std::shared_ptr<simulator::ISimulator> simulator);
    /**
     * @brief Generates a dataset from rows ordered like openParameterNames().
     * @param output New dataset directory to create.
     * @param parameterRows Complete externally sampled open-parameter matrix.
     * @param options Execution, storage, and sampling-provenance options.
     * @param progress Optional callback invoked after each batch.
     * @return Counters describing the completed or cancelled run.
     * @throws std::invalid_argument if any row or option is invalid.
     */
    GenerationSummary generate(const std::filesystem::path& output,
        const std::vector<std::vector<double>>& parameterRows,
        const GenerationOptions& options, ProgressCallback progress = {});
    /** @brief Cooperatively requests cancellation of this run and its simulator. */
    void requestStop() noexcept;
private:
    GenerationConfig config_;
    std::shared_ptr<simulator::ISimulator> simulator_;
    std::atomic_bool stopRequested_{false};
};

} // namespace ibeamlab::generation
