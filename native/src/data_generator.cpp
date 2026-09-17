#include <ibeamlab/data_generator.h>

#include <toml++/toml.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ibeamlab::generation {
namespace {
std::string optionsToml(const GenerationOptions &options) {
    const char *failurePolicy = options.failurePolicy == FailurePolicy::Stop ? "stop"
        : options.failurePolicy == FailurePolicy::Record ? "record" : "discard";
    toml::table table{{"batch_size", static_cast<std::int64_t>(options.batchSize)},
        {"shard_count", static_cast<std::int64_t>(options.shardCount)},
        {"seed", static_cast<std::int64_t>(options.seed)},
        {"failure_policy", failurePolicy}, {"sampler", options.sampler},
        {"sampler_version", static_cast<std::int64_t>(options.samplerVersion)}};
    std::ostringstream output;
    output << table;
    return output.str();
}
} // namespace

DataGenerator::DataGenerator(GenerationConfig config,
                             std::shared_ptr<simulator::ISimulator> simulator)
    : config_(std::move(config)), simulator_(std::move(simulator)) {
    if (!simulator_)
        throw std::invalid_argument("data generator simulator is null");
    config_.validate();
}
void DataGenerator::requestStop() noexcept {
    stopRequested_.store(true);
    simulator_->requestStop();
}
GenerationSummary DataGenerator::generate(const std::filesystem::path &output,
                                          const std::vector<std::vector<double>> &rows,
                                          const GenerationOptions &options,
                                          ProgressCallback progress) {
    if (rows.empty() || options.batchSize == 0 || options.shardCount == 0 ||
        options.shardCount > rows.size())
        throw std::invalid_argument("invalid generation sizes");
    // Reject the complete externally sampled matrix before creating a dataset
    // directory or starting an expensive simulator batch.
    for (const auto &row : rows)
        static_cast<void>(config_.materialize(row));
    stopRequested_.store(false);
    simulator_->resetStop();
    datasets::DatasetMetadata metadata;
    metadata.requested = rows.size();
    metadata.seed = options.seed;
    metadata.simulator = "ibeamlab";
    metadata.generationConfigToml = generationConfigToToml(config_);
    metadata.generationOptionsToml = optionsToml(options);
    metadata.samplingConfigToml = options.samplingConfigToml;
    metadata.simulatorConfigToml = simulator_->configurationToml();
    metadata.parameterNames = config_.openParameterNames();
    metadata.provenance.ibeamlabVersion = IBEAMLAB_VERSION;
    metadata.provenance.build = IBEAMLAB_BUILD_TYPE;
    metadata.provenance.platform = IBEAMLAB_PLATFORM;
    metadata.provenance.simulator = "configured ISimulator";
    metadata.provenance.sampler = options.sampler;
    metadata.provenance.samplerVersion = options.samplerVersion;
    metadata.provenance.seed = options.seed;
    for (const auto &method : config_.methods)
        metadata.spectrumLabels.push_back(method.label);
    datasets::DatasetWriter writer(output, metadata, options.shardCount);
    GenerationSummary summary{rows.size()};
    for (std::size_t start = 0; start < rows.size() && !stopRequested_.load();
         start += options.batchSize) {
        const auto end = std::min(rows.size(), start + options.batchSize);
        std::vector<simulator::SimulationInput> inputs;
        inputs.reserve(end - start);
        for (std::size_t i = start; i < end; ++i)
            inputs.push_back(config_.materialize(rows[i]));
        simulator::SimulationOptions simOptions;
        simOptions.continueAfterFailure = options.failurePolicy == FailurePolicy::Record;
        auto results = simulator_->simulateBatch(inputs, simOptions);
        if (results.size() != inputs.size() && !stopRequested_.load())
            throw std::runtime_error("simulator returned incomplete batch");
        for (std::size_t local = 0; local < results.size(); ++local) {
            const auto index = start + local;
            std::ostringstream id;
            id << "sample-" << std::setw(8) << std::setfill('0') << index;
            if (results[local].failure) {
                const auto &failure = *results[local].failure;
                if (options.failurePolicy == FailurePolicy::Discard)
                    writer.noteDiscardedFailure();
                else
                    writer.appendFailure(
                        {index, id.str(), failure.methodLabel, failure.type, failure.message});
                ++summary.failed;
                if (options.failurePolicy == FailurePolicy::Stop)
                    throw std::runtime_error(failure.message);
            } else {
                bool valid = results[local].spectra.size() == metadata.spectrumLabels.size();
                for (std::size_t s = 0; valid && s < results[local].spectra.size(); ++s) {
                    const auto &spectrum = results[local].spectra[s];
                    valid = spectrum.label == metadata.spectrumLabels[s] &&
                            !spectrum.counts.empty() &&
                            std::all_of(spectrum.counts.begin(), spectrum.counts.end(),
                                        [](float value) { return std::isfinite(value); });
                }
                if (!valid) {
                    writer.appendInvalid({index,
                                          id.str(),
                                          {},
                                          "InvalidSpectrum",
                                          "simulator returned invalid labels, length, or counts"});
                    ++summary.invalid;
                } else {
                    writer.append({index, id.str(), rows[index], std::move(results[local])});
                    ++summary.accepted;
                }
            }
        }
        if (progress)
            progress({end, summary.accepted, summary.invalid, summary.failed, rows.size()});
    }
    summary.cancelled = stopRequested_.load();
    if (!summary.cancelled)
        writer.finalize();
    return summary;
}

} // namespace ibeamlab::generation
