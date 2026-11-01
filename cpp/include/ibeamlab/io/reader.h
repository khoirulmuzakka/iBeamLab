#pragma once
#include <filesystem>
#include <ibeamlab/core/data.h>
#include <optional>
namespace ibeamlab::io {
/** @brief Explicit interpretation of numeric experimental spectrum records. */
enum class SpectrumFormat { Counts, ChannelCounts, Csv };
/** @brief Text import controls. Column indices and stored channels are zero-based. */
struct ReadOptions {
    SpectrumFormat format = SpectrumFormat::ChannelCounts;
    std::size_t counts_column = 1;
    std::optional<std::size_t> channel_column = 0, variance_column;
    std::size_t channel_base = 0, skip_rows = 0, maximum_channels = 1000000;
    bool header = false;
    char delimiter = '\0', comment = '#';
};
/** @brief Read one spectrum with filename/physical-line error context.
 * Counts format uses consecutive numeric records; blank/comment lines are ignored.
 * Channel indices must be unique nonnegative integers after channel_base removal.
 * CSV supports quoted fields and doubled quotes, but not multiline records.
 * Calibration and detector geometry must be supplied separately in MeasuredData.
 */
IBEAMLAB_API core::MeasuredSpectrum read_spectrum(const std::filesystem::path &,
                                                  std::string measurement, ReadOptions = {});
} // namespace ibeamlab::io
