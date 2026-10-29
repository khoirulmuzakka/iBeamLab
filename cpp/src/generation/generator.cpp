#include <algorithm>
#include <ibeamlab/generation/generator.h>
#include <optional>
namespace ibeamlab::generation {
void GenerationPlan::validate() const {
    datasets::Metadata m;
    m.reference = reference;
    m.parameters = parameters;
    m.validate();
}
Summary generate(const GenerationPlan &plan, std::span<const double> rows, std::size_t samples,
                 simulation::Simulator &sim, const std::filesystem::path &path,
                 FailurePolicy policy, simulation::Progress progress, std::string provenance,
                 std::size_t shard_size, std::size_t batch_size) {
    plan.validate();
    auto width = plan.parameters.size();
    if (width && samples > rows.size() / width)
        throw std::invalid_argument("parameter matrix too short");
    if (rows.size() != samples * width)
        throw std::invalid_argument("parameter matrix shape mismatch");
    if (!batch_size || batch_size > 4096)
        throw std::invalid_argument("batch_size must be between 1 and 4096");
    datasets::Metadata metadata;
    metadata.reference = plan.reference;
    metadata.parameters = plan.parameters;
    metadata.requested = samples;
    metadata.provenance = sim.provenance() + "\n" + provenance;
    datasets::Writer writer(path, std::move(metadata), shard_size);
    const auto capacity = std::min(samples, batch_size);
    std::vector<core::Experiment> candidates(capacity, plan.reference);
    std::vector<std::size_t> source_indices(capacity);
    std::vector<std::size_t> slots(capacity);
    std::vector<std::optional<simulation::Failure>> invalid(capacity);
    datasets::Record record;
    record.parameters.resize(width);
    Summary summary;
    std::size_t reported = 0;
    auto report = [&](std::size_t completed) {
        while (reported < completed && !sim.stopped()) {
            ++reported;
            if (progress)
                progress(reported, samples);
        }
    };
    for (std::size_t begin = 0; begin < samples; begin += batch_size) {
        if (sim.stopped()) {
            summary.cancelled = true;
            break;
        }
        auto count = std::min(batch_size, samples - begin);
        std::size_t valid = 0;
        for (std::size_t i = 0; i < count; ++i) {
            invalid[i].reset();
            slots[i] = capacity;
            auto &candidate = candidates[valid];
            try {
                for (std::size_t j = 0; j < width; ++j)
                    core::write_parameter(candidate, plan.parameters[j].target,
                                          rows[(begin + i) * width + j]);
                candidate.validate();
                slots[i] = valid;
                source_indices[valid++] = begin + i;
            } catch (const std::exception &error) {
                invalid[i] = simulation::Failure{"", error.what()};
                if (policy == FailurePolicy::Stop) {
                    count = i + 1;
                    break;
                }
            }
        }
        // Keep candidates alive until the backend has drained every in-flight job.
        // SIMNRA can now keep its worker pool busy instead of waiting on each row.
        auto results =
            sim.simulate_batch(std::span<const core::Experiment>(candidates.data(), valid),
                               [&](std::size_t done, std::size_t) {
                                   if (done && done <= valid)
                                       report(source_indices[done - 1] + 1);
                               });
        if (sim.stopped()) {
            summary.cancelled = true;
            break;
        }
        if (results.size() != valid)
            throw std::runtime_error("simulation batch returned an unexpected result count");
        for (std::size_t i = 0; i < count; ++i) {
            record.sample_index = begin + i;
            record.sample_id = std::to_string(begin + i);
            for (std::size_t j = 0; j < width; ++j)
                record.parameters[j] = rows[(begin + i) * width + j];
            if (invalid[i]) {
                record.result.spectra.clear();
                record.result.failure = std::move(invalid[i]);
            } else {
                record.result = std::move(results[slots[i]]);
            }
            if (record.result.failure) {
                if (policy == FailurePolicy::Stop)
                    throw std::runtime_error("generation failed at sample " +
                                             std::to_string(begin + i) + ": " +
                                             record.result.failure->message);
                for (double v : record.parameters)
                    if (!std::isfinite(v))
                        throw std::invalid_argument("nonfinite generation parameter");
                if (policy == FailurePolicy::Discard) {
                    writer.discard();
                    ++summary.discarded;
                } else {
                    writer.append(record);
                    ++summary.failed;
                }
            } else {
                writer.append(record);
                ++summary.accepted;
            }
            report(begin + i + 1);
            if (sim.stopped()) {
                summary.cancelled = true;
                break;
            }
        }
        if (summary.cancelled)
            break;
    }
    if (!summary.cancelled)
        writer.finish();
    return summary;
}
void sample_uniform(std::span<const Uniform> specs, std::size_t samples, std::uint64_t seed,
                    std::span<double> output) {
    if (specs.size() && samples > output.size() / specs.size())
        throw std::invalid_argument("sampling output too short");
    if (output.size() != samples * specs.size())
        throw std::invalid_argument("sampling shape mismatch");
    for (auto &s : specs)
        s.bounds.validate();
    std::mt19937_64 engine(seed);
    for (std::size_t i = 0; i < samples; ++i)
        for (std::size_t j = 0; j < specs.size(); ++j) {
            double u = static_cast<double>(engine() >> 11) * 0x1.0p-53;
            output[i * specs.size() + j] =
                std::lerp(specs[j].bounds.minimum, specs[j].bounds.maximum, u);
        }
}
} // namespace ibeamlab::generation
