#pragma once

#include <ibeamlab/simulator.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ibeamlab::datasets {

inline constexpr std::uint32_t DatasetFormatVersion = 1;

struct FailureRecord {
    std::uint64_t sampleIndex{};
    std::string sampleId;
    std::string methodLabel;
    std::string type;
    std::string message;
};

struct DatasetRecord {
    std::uint64_t sampleIndex{};
    std::string sampleId;
    std::vector<double> openParameters;
    simulator::SimulationResult result;
};

struct DatasetBatch { std::vector<DatasetRecord> records; };
struct DatasetProvenance {
    std::string ibeamlabVersion;
    std::string build;
    std::string platform;
    std::string simulator;
    std::string sampler{"uniform"};
    std::uint32_t samplerVersion{1};
    std::uint64_t seed{};
};

struct DatasetMetadata {
    std::uint32_t formatVersion{DatasetFormatVersion};
    bool complete{false};
    std::uint64_t requested{};
    std::uint64_t accepted{};
    std::uint64_t invalid{};
    std::uint64_t failed{};
    std::uint64_t seed{};
    std::string createdUtc;
    std::string completedUtc;
    std::string simulator;
    DatasetProvenance provenance;
    std::vector<std::string> parameterNames;
    std::vector<std::string> spectrumLabels;
    std::vector<std::uint64_t> spectrumLengths;
    std::vector<std::string> shardFiles;
};

class IDatasetWriter {
public:
    virtual ~IDatasetWriter() = default;
    virtual void append(const DatasetRecord& record) = 0;
    virtual void appendFailure(const FailureRecord& failure) = 0;
    virtual void noteDiscardedFailure() = 0;
    virtual void finalize() = 0;
};

class IDatasetReader {
public:
    virtual ~IDatasetReader() = default;
    virtual const DatasetMetadata& metadata() const noexcept = 0;
    virtual std::vector<DatasetRecord> readAll() const = 0;
    virtual std::vector<FailureRecord> readFailures() const = 0;
};

class DatasetWriter final : public IDatasetWriter {
public:
    DatasetWriter(std::filesystem::path directory, DatasetMetadata metadata,
                  std::size_t shardCount = 1);
    ~DatasetWriter() override;
    void append(const DatasetRecord& record) override;
    void appendFailure(const FailureRecord& failure) override;
    void noteDiscardedFailure() override;
    void appendInvalid(const FailureRecord& failure);
    void finalize() override;
    const DatasetMetadata& metadata() const noexcept { return metadata_; }
private:
    void writeManifest() const;
    std::filesystem::path directory_;
    DatasetMetadata metadata_;
    std::vector<std::ofstream> shards_;
    std::ofstream failures_;
    std::size_t nextShard_{};
    bool finalized_{false};
};

class DatasetReader final : public IDatasetReader {
public:
    explicit DatasetReader(std::filesystem::path directory);
    const DatasetMetadata& metadata() const noexcept override { return metadata_; }
    std::vector<DatasetRecord> readAll() const override;
    std::vector<FailureRecord> readFailures() const override;
private:
    std::filesystem::path directory_;
    DatasetMetadata metadata_;
};

} // namespace ibeamlab::datasets
