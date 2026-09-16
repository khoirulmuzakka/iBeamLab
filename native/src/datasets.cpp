#include <ibeamlab/datasets.h>

#include <toml++/toml.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ibeamlab::datasets {
namespace {
constexpr std::array<char, 8> Magic{'I', 'B', 'E', 'A', 'M', 'D', 'S', '1'};
std::string nowUtc() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%FT%TZ");
    return output.str();
}
template <class T> void writeValue(std::ostream &output, const T &value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    if (!output)
        throw std::runtime_error("dataset write failed");
}
template <class T> T readValue(std::istream &input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input)
        throw std::runtime_error("truncated dataset");
    return value;
}
void writeString(std::ostream &output, const std::string &value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("dataset string too long");
    writeValue(output, static_cast<std::uint32_t>(value.size()));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
}
std::string readString(std::istream &input) {
    const auto size = readValue<std::uint32_t>(input);
    if (size > 16U * 1024U * 1024U)
        throw std::runtime_error("dataset string exceeds safety limit");
    std::string value(size, '\0');
    input.read(value.data(), size);
    if (!input)
        throw std::runtime_error("truncated dataset string");
    return value;
}
void header(std::ostream &output) { output.write(Magic.data(), Magic.size()); }
void verifyHeader(std::istream &input) {
    std::array<char, 8> value{};
    input.read(value.data(), value.size());
    if (value != Magic)
        throw std::runtime_error("invalid dataset shard magic");
}
std::vector<std::string> strings(const toml::array *values) {
    std::vector<std::string> result;
    if (!values)
        return result;
    for (const auto &value : *values) {
        auto text = value.value<std::string>();
        if (!text)
            throw std::runtime_error("invalid TOML string array");
        result.push_back(*text);
    }
    return result;
}
void insertToml(toml::table &destination, const char *key, const std::string &source) {
    if (!source.empty())
        destination.insert(key, toml::parse(source));
}
std::string tableToml(const toml::table *table) {
    if (!table)
        return {};
    std::ostringstream output;
    output << *table;
    return output.str();
}
} // namespace

DatasetWriter::DatasetWriter(std::filesystem::path directory, DatasetMetadata metadata,
                             std::size_t shardCount)
    : directory_(std::move(directory)), metadata_(std::move(metadata)) {
    if (shardCount == 0)
        throw std::invalid_argument("shard count must be positive");
    if (std::filesystem::exists(directory_) && !std::filesystem::is_empty(directory_))
        throw std::runtime_error("dataset directory is not empty");
    std::filesystem::create_directories(directory_);
    metadata_.createdUtc = nowUtc();
    metadata_.complete = false;
    if (metadata_.spectrumLengths.empty())
        metadata_.spectrumLengths.assign(metadata_.spectrumLabels.size(), 0);
    else if (metadata_.spectrumLengths.size() != metadata_.spectrumLabels.size())
        throw std::invalid_argument("dataset spectrum lengths and labels differ in size");
    for (std::size_t i = 0; i < shardCount; ++i) {
        std::ostringstream name;
        name << "part-" << std::setw(6) << std::setfill('0') << i << ".ibd";
        metadata_.shardFiles.push_back(name.str());
        shards_.emplace_back(directory_ / name.str(), std::ios::binary | std::ios::trunc);
        if (!shards_.back())
            throw std::runtime_error("cannot create dataset shard");
        header(shards_.back());
    }
    failures_.open(directory_ / "failures.ibd", std::ios::binary | std::ios::trunc);
    header(failures_);
    writeManifest();
}
DatasetWriter::~DatasetWriter() {
    for (auto &stream : shards_)
        if (stream.is_open())
            stream.close();
    if (failures_.is_open())
        failures_.close();
}
void DatasetWriter::append(const DatasetRecord &record) {
    if (finalized_)
        throw std::logic_error("dataset is finalized");
    if (record.openParameters.size() != metadata_.parameterNames.size())
        throw std::invalid_argument("dataset parameter count does not match metadata");
    if (record.result.spectra.size() != metadata_.spectrumLabels.size())
        throw std::invalid_argument("dataset spectrum count does not match metadata");
    for (std::size_t index = 0; index < record.result.spectra.size(); ++index) {
        if (record.result.spectra[index].label != metadata_.spectrumLabels[index])
            throw std::invalid_argument("dataset spectrum ordering differs from metadata");
        metadata_.spectrumLengths[index] = std::max<std::uint64_t>(
            metadata_.spectrumLengths[index], record.result.spectra[index].counts.size());
    }
    auto &out = shards_[nextShard_++ % shards_.size()];
    writeValue(out, record.sampleIndex);
    writeString(out, record.sampleId);
    writeValue(out, static_cast<std::uint32_t>(record.openParameters.size()));
    for (double value : record.openParameters)
        writeValue(out, value);
    writeValue(out, static_cast<std::uint32_t>(record.result.spectra.size()));
    for (const auto &spectrum : record.result.spectra) {
        writeString(out, spectrum.label);
        writeValue(out, static_cast<std::uint64_t>(spectrum.counts.size()));
        out.write(reinterpret_cast<const char *>(spectrum.counts.data()),
                  static_cast<std::streamsize>(spectrum.counts.size() * sizeof(float)));
    }
    ++metadata_.accepted;
}
void DatasetWriter::appendFailure(const FailureRecord &failure) {
    if (finalized_)
        throw std::logic_error("dataset is finalized");
    writeValue(failures_, failure.sampleIndex);
    writeString(failures_, failure.sampleId);
    writeString(failures_, failure.methodLabel);
    writeString(failures_, failure.type);
    writeString(failures_, failure.message);
    ++metadata_.failed;
}
void DatasetWriter::noteDiscardedFailure() {
    if (finalized_)
        throw std::logic_error("dataset is finalized");
    ++metadata_.failed;
}
void DatasetWriter::appendInvalid(const FailureRecord &failure) {
    if (finalized_)
        throw std::logic_error("dataset is finalized");
    writeValue(failures_, failure.sampleIndex);
    writeString(failures_, failure.sampleId);
    writeString(failures_, failure.methodLabel);
    writeString(failures_, failure.type);
    writeString(failures_, failure.message);
    ++metadata_.invalid;
}
void DatasetWriter::finalize() {
    if (finalized_)
        return;
    for (auto &stream : shards_) {
        stream.flush();
        stream.close();
    }
    failures_.flush();
    failures_.close();
    metadata_.complete = true;
    metadata_.completedUtc = nowUtc();
    writeManifest();
    finalized_ = true;
}
void DatasetWriter::writeManifest() const {
    toml::table root{{"format", "ibeamlab.dataset"},
                     {"format_version", static_cast<std::int64_t>(metadata_.formatVersion)},
                     {"complete", metadata_.complete},
                     {"requested", static_cast<std::int64_t>(metadata_.requested)},
                     {"accepted", static_cast<std::int64_t>(metadata_.accepted)},
                     {"invalid", static_cast<std::int64_t>(metadata_.invalid)},
                     {"failed", static_cast<std::int64_t>(metadata_.failed)},
                     {"seed", static_cast<std::int64_t>(metadata_.seed)},
                     {"created_utc", metadata_.createdUtc},
                     {"completed_utc", metadata_.completedUtc},
                     {"simulator", metadata_.simulator}};
    toml::array parameters, labels, lengths, shards;
    for (const auto &x : metadata_.parameterNames)
        parameters.push_back(x);
    for (const auto &x : metadata_.spectrumLabels)
        labels.push_back(x);
    for (const auto x : metadata_.spectrumLengths)
        lengths.push_back(static_cast<std::int64_t>(x));
    for (const auto &x : metadata_.shardFiles)
        shards.push_back(x);
    root.insert("parameter_names", std::move(parameters));
    root.insert("spectrum_labels", std::move(labels));
    root.insert("spectrum_lengths", std::move(lengths));
    root.insert("shards", std::move(shards));
    insertToml(root, "generation", metadata_.generationConfigToml);
    insertToml(root, "generation_options", metadata_.generationOptionsToml);
    insertToml(root, "simulator_config", metadata_.simulatorConfigToml);
    root.insert("provenance",
                toml::table{{"ibeamlab_version", metadata_.provenance.ibeamlabVersion},
                            {"build", metadata_.provenance.build},
                            {"platform", metadata_.provenance.platform},
                            {"simulator", metadata_.provenance.simulator},
                            {"sampler", metadata_.provenance.sampler},
                            {"sampler_version",
                             static_cast<std::int64_t>(metadata_.provenance.samplerVersion)},
                            {"seed", static_cast<std::int64_t>(metadata_.provenance.seed)}});
    const auto temporary = directory_ / "dataset.toml.tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        output << root;
        output.flush();
        if (!output)
            throw std::runtime_error("cannot write dataset manifest");
    }
    const auto destination = directory_ / "dataset.toml";
    std::error_code error;
    std::filesystem::remove(destination, error);
    error.clear();
    std::filesystem::rename(temporary, destination, error);
    if (error)
        throw std::runtime_error("cannot publish dataset manifest: " + error.message());
}
DatasetReader::DatasetReader(std::filesystem::path directory) : directory_(std::move(directory)) {
    const auto root = toml::parse_file((directory_ / "dataset.toml").string());
    if (root["format"].value_or<std::string>("") != "ibeamlab.dataset")
        throw std::runtime_error("unsupported dataset format");
    metadata_.formatVersion = static_cast<std::uint32_t>(root["format_version"].value_or(0));
    if (metadata_.formatVersion != DatasetFormatVersion)
        throw std::runtime_error("unsupported dataset version");
    metadata_.complete = root["complete"].value_or(false);
    metadata_.requested = root["requested"].value_or<std::uint64_t>(0);
    metadata_.accepted = root["accepted"].value_or<std::uint64_t>(0);
    metadata_.invalid = root["invalid"].value_or<std::uint64_t>(0);
    metadata_.failed = root["failed"].value_or<std::uint64_t>(0);
    metadata_.seed = root["seed"].value_or<std::uint64_t>(0);
    metadata_.createdUtc = root["created_utc"].value_or<std::string>("");
    metadata_.completedUtc = root["completed_utc"].value_or<std::string>("");
    metadata_.simulator = root["simulator"].value_or<std::string>("");
    metadata_.generationConfigToml = tableToml(root["generation"].as_table());
    metadata_.generationOptionsToml = tableToml(root["generation_options"].as_table());
    metadata_.simulatorConfigToml = tableToml(root["simulator_config"].as_table());
    metadata_.parameterNames = strings(root["parameter_names"].as_array());
    metadata_.spectrumLabels = strings(root["spectrum_labels"].as_array());
    if (auto values=root["spectrum_lengths"].as_array()) for(const auto& value:*values) {
        auto length=value.value<std::uint64_t>(); if(!length) throw std::runtime_error("invalid spectrum length in dataset manifest"); metadata_.spectrumLengths.push_back(*length);
    }
    metadata_.shardFiles = strings(root["shards"].as_array());
    if (metadata_.spectrumLengths.size() != metadata_.spectrumLabels.size())
        throw std::runtime_error("dataset spectrum lengths and labels differ in size");
    if (auto p = root["provenance"].as_table()) {
        metadata_.provenance.ibeamlabVersion = (*p)["ibeamlab_version"].value_or<std::string>("");
        metadata_.provenance.build = (*p)["build"].value_or<std::string>("");
        metadata_.provenance.platform = (*p)["platform"].value_or<std::string>("");
        metadata_.provenance.simulator = (*p)["simulator"].value_or<std::string>("");
        metadata_.provenance.sampler = (*p)["sampler"].value_or<std::string>("uniform");
        metadata_.provenance.samplerVersion = (*p)["sampler_version"].value_or<std::uint32_t>(1);
        metadata_.provenance.seed = (*p)["seed"].value_or<std::uint64_t>(0);
    }
}
std::vector<DatasetRecord> DatasetReader::readAll() const {
    std::vector<DatasetRecord> records;
    for (const auto &name : metadata_.shardFiles) {
        std::ifstream in(directory_ / name, std::ios::binary);
        verifyHeader(in);
        while (in.peek() != std::char_traits<char>::eof()) {
            DatasetRecord record;
            record.sampleIndex = readValue<std::uint64_t>(in);
            record.sampleId = readString(in);
            const auto parameters = readValue<std::uint32_t>(in);
            if (parameters > 100000)
                throw std::runtime_error("too many dataset parameters");
            record.openParameters.resize(parameters);
            for (auto &value : record.openParameters)
                value = readValue<double>(in);
            const auto spectra = readValue<std::uint32_t>(in);
            if (spectra > 10000)
                throw std::runtime_error("too many spectra");
            if (spectra != metadata_.spectrumLabels.size())
                throw std::runtime_error("dataset spectrum count differs from metadata");
            for (std::uint32_t i = 0; i < spectra; ++i) {
                simulator::Spectrum spectrum;
                spectrum.label = readString(in);
                if (spectrum.label != metadata_.spectrumLabels[i])
                    throw std::runtime_error("dataset spectrum ordering differs from metadata");
                const auto count = readValue<std::uint64_t>(in);
                if (count > 100000000)
                    throw std::runtime_error("spectrum exceeds safety limit");
                if (count > metadata_.spectrumLengths[i])
                    throw std::runtime_error("dataset spectrum exceeds declared maximum length");
                spectrum.counts.resize(static_cast<std::size_t>(count));
                in.read(reinterpret_cast<char *>(spectrum.counts.data()),
                        static_cast<std::streamsize>(count * sizeof(float)));
                if (!in)
                    throw std::runtime_error("truncated spectrum");
                spectrum.counts.resize(
                    static_cast<std::size_t>(metadata_.spectrumLengths[i]), 0.0F);
                record.result.spectra.push_back(std::move(spectrum));
            }
            records.push_back(std::move(record));
        }
    }
    return records;
}
std::vector<FailureRecord> DatasetReader::readFailures() const {
    std::vector<FailureRecord> records;
    std::ifstream in(directory_ / "failures.ibd", std::ios::binary);
    verifyHeader(in);
    while (in.peek() != std::char_traits<char>::eof()) {
        FailureRecord value;
        value.sampleIndex = readValue<std::uint64_t>(in);
        value.sampleId = readString(in);
        value.methodLabel = readString(in);
        value.type = readString(in);
        value.message = readString(in);
        records.push_back(std::move(value));
    }
    return records;
}

} // namespace ibeamlab::datasets
