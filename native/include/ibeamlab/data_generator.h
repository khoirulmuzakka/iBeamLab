#pragma once

#include <ibeamlab/datasets.h>
#include <ibeamlab/generation.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace ibeamlab::generation {

enum class FailurePolicy { Stop, Record, Discard };
struct GenerationOptions {
    std::size_t samples{};
    std::size_t batchSize{1024};
    std::size_t shardCount{1};
    std::uint64_t seed{};
    FailurePolicy failurePolicy{FailurePolicy::Stop};
};
struct GenerationProgress { std::size_t attempted{}, accepted{}, invalid{}, failed{}, total{}; };
struct GenerationSummary { std::size_t requested{}, accepted{}, invalid{}, failed{}; bool cancelled{}; };
using ProgressCallback = std::function<void(const GenerationProgress&)>;

class DataGenerator {
public:
    DataGenerator(GenerationConfig config, std::shared_ptr<simulator::ISimulator> simulator);
    GenerationSummary generate(const std::filesystem::path& output,
        const GenerationOptions& options, ProgressCallback progress = {});
    void requestStop() noexcept;
private:
    GenerationConfig config_;
    std::shared_ptr<simulator::ISimulator> simulator_;
    std::atomic_bool stopRequested_{false};
};

} // namespace ibeamlab::generation
