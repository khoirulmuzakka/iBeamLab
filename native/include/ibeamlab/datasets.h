#pragma once

/**
 * @file datasets.h
 * @brief Versioned, streaming storage for simulated spectra and their metadata.
 *
 * This module owns the portable native dataset schema and reliable I/O. It does
 * not choose samples, split training data, or perform ML-specific augmentation.
 */

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

/** @brief Current on-disk dataset format version. */
inline constexpr std::uint32_t DatasetFormatVersion = 1;

/** @brief Serializable description of a failed or invalid sample. */
struct FailureRecord {
    std::uint64_t sampleIndex{};
    std::string sampleId;
    std::string methodLabel;
    std::string type;
    std::string message;
};

/** @brief One successful sample, its open parameters, and simulated spectra. */
struct DatasetRecord {
    std::uint64_t sampleIndex{};
    std::string sampleId;
    std::vector<double> openParameters;
    simulator::SimulationResult result;
};

/** @brief In-memory group of successful dataset records. */
struct DatasetBatch { std::vector<DatasetRecord> records; };
/** @brief Build, simulator, and external-sampler provenance. */
struct DatasetProvenance {
    std::string ibeamlabVersion;
    std::string build;
    std::string platform;
    std::string simulator;
    std::string sampler{"uniform"};
    std::uint32_t samplerVersion{1};
    std::uint64_t seed{};
};

/** @brief Manifest data stored in and loaded from dataset.toml. */
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
    std::string generationConfigToml;
    std::string generationOptionsToml;
    std::string samplingConfigToml;
    std::string simulatorConfigToml;
    DatasetProvenance provenance;
    std::vector<std::string> parameterNames;
    std::vector<std::string> spectrumLabels;
    std::vector<std::uint64_t> spectrumLengths;
    std::vector<std::string> shardFiles;
};

/** @brief Abstract streaming dataset sink. */
class IDatasetWriter {
public:
    virtual ~IDatasetWriter() = default;
    virtual void append(const DatasetRecord& record) = 0;
    virtual void appendFailure(const FailureRecord& failure) = 0;
    virtual void noteDiscardedFailure() = 0;
    virtual void finalize() = 0;
};

/** @brief Abstract dataset source. */
class IDatasetReader {
public:
    virtual ~IDatasetReader() = default;
    virtual const DatasetMetadata& metadata() const noexcept = 0;
    virtual std::vector<DatasetRecord> readAll() const = 0;
    virtual std::vector<FailureRecord> readFailures() const = 0;
};

/**
 * @brief Writes sharded dataset records and an atomic TOML manifest.
 *
 * The writer keeps the manifest incomplete until finalize() succeeds and tracks
 * the maximum spectrum length per method without buffering the full dataset.
 */
class DatasetWriter final : public IDatasetWriter {
public:
    /** @brief Creates a new dataset directory and its output shards. */
    DatasetWriter(std::filesystem::path directory, DatasetMetadata metadata,
                  std::size_t shardCount = 1);
    ~DatasetWriter() override;
    void append(const DatasetRecord& record) override;
    void appendFailure(const FailureRecord& failure) override;
    void noteDiscardedFailure() override;
    void appendInvalid(const FailureRecord& failure);
    /** @brief Flushes output and marks the manifest complete. */
    void finalize() override;
    /** @brief Returns the writer's current manifest metadata. */
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

/**
 * @brief Validates and reads a native dataset directory.
 *
 * Shorter spectra are zero-padded to the per-method maximum declared by the
 * manifest so callers receive rectangular method inputs.
 */
class DatasetReader final : public IDatasetReader {
public:
    /** @brief Opens a dataset directory and parses its manifest. */
    explicit DatasetReader(std::filesystem::path directory);
    /** @brief Returns parsed manifest metadata. */
    const DatasetMetadata& metadata() const noexcept override { return metadata_; }
    /** @brief Reads every successful record and pads spectra as declared. */
    std::vector<DatasetRecord> readAll() const override;
    /** @brief Reads all recorded simulator and validation failures. */
    std::vector<FailureRecord> readFailures() const override;
private:
    std::filesystem::path directory_;
    DatasetMetadata metadata_;
};

} // namespace ibeamlab::datasets
