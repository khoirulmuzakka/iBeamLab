#include <ibeamlab/generation/generator.h>
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
                 std::size_t shard_size) {
    plan.validate();
    auto width = plan.parameters.size();
    if (width && samples > rows.size() / width)
        throw std::invalid_argument("parameter matrix too short");
    if (rows.size() != samples * width)
        throw std::invalid_argument("parameter matrix shape mismatch");
    datasets::Metadata metadata;
    metadata.reference = plan.reference;
    metadata.parameters = plan.parameters;
    metadata.requested = samples;
    metadata.provenance = sim.provenance() + "\n" + provenance;
    datasets::Writer writer(path, std::move(metadata), shard_size);
    auto working = plan.reference;
    datasets::Record record;
    record.parameters.resize(width);
    Summary summary;
    for (std::size_t i = 0; i < samples; ++i) {
        if (sim.stopped()) {
            summary.cancelled = true;
            break;
        }
        record.sample_index = i;
        record.sample_id = std::to_string(i);
        record.result.failure.reset();
        for (std::size_t j = 0; j < width; ++j)
            record.parameters[j] = rows[i * width + j];
        try {
            for (std::size_t j = 0; j < width; ++j)
                core::write_parameter(working, plan.parameters[j].target, record.parameters[j]);
            working.validate();
            sim.simulate(working, record.result);
        } catch (const std::exception &e) {
            record.result.failure = simulation::Failure{"", e.what()};
        }
        if (sim.stopped()) {
            summary.cancelled = true;
            break;
        }
        if (record.result.failure) {
            if (policy == FailurePolicy::Stop)
                throw std::runtime_error("generation failed at sample " + std::to_string(i) + ": " +
                                         record.result.failure->message);
            // Failure records must still carry finite parameter values.
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
        if (progress)
            progress(i + 1, samples);
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
