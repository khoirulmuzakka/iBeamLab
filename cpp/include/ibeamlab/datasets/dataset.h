#pragma once
#include <filesystem>
#include <ibeamlab/core/parameter.h>
#include <ibeamlab/simulation/simulator.h>
#include <memory>
namespace ibeamlab::datasets {
using Parameter = core::ParameterSpec;
/** @brief Dataset reference, ordered physical parameter columns and outcome counters.
 * A complete dataset accounts for requested rows as accepted, failed or discarded.
 */
struct IBEAMLAB_API Metadata {
    core::Experiment reference;
    std::vector<Parameter> parameters;
    std::string provenance;
    std::uint64_t requested = 0, accepted = 0, failed = 0, discarded = 0;
    void validate() const;
};
/** @brief One generated row, containing physical parameters and spectra or a failure.
 * sample_index refers to the original input row, even when other rows are discarded.
 */
struct Record {
    std::uint64_t sample_index = 0;
    std::string sample_id;
    core::Counts parameters;
    simulation::Result result;
};
/** @brief Single-owner, sharded dataset writer that refuses an existing directory.
 * Call finish() explicitly after accounting for every requested row. Destruction
 * does not finalize; interrupted generation leaves an incomplete dataset marker.
 */
class IBEAMLAB_API Writer {
  public:
    Writer(const std::filesystem::path &, Metadata, std::size_t records_per_shard = 64);
    ~Writer();
    Writer(const Writer &) = delete;
    Writer &operator=(const Writer &) = delete;
    /** @brief Append a success or failure record with a strictly increasing sample index. */
    void append(const Record &);
    /** @brief Account for one omitted input row without writing a record. */
    void discard();
    /** @brief Flush pending records and publish the completed manifest; idempotent. */
    void finish();
    const Metadata &metadata() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
/** @brief Single-cursor reader for a completed dataset, loading one shard at a time.
 * Each caller owns its reader; concurrent access to one reader is unsupported.
 */
class IBEAMLAB_API Reader {
  public:
    explicit Reader(const std::filesystem::path &);
    ~Reader();
    Reader(const Reader &) = delete;
    Reader &operator=(const Reader &) = delete;
    const Metadata &metadata() const;
    // Loads one shard at a time. Reuses output's parameter/spectrum buffers.
    // A reader is a single cursor, exclusively owned by one caller.
    /** @brief Fill output, reusing its buffers; return false at the end of the dataset. */
    bool next(Record &output);
    /** @brief Reset the cursor so the next read returns the first stored record. */
    void rewind();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ibeamlab::datasets
