#pragma once
#include <ibeamlab/io/reader.h>
namespace ibeamlab::io {
/** Optional numeric header field. Match is case-sensitive; the first numeric
 * token after marker is scaled to seconds. Defaults never hide malformed values. */
struct TimeField {
    std::string marker;
    double seconds_per_unit = 1;
    std::optional<double> fallback;
};
struct Parser {
    ReadOptions records;
    std::string start_marker, end_marker;
    TimeField real_time, live_time;
    bool missing_channels_are_zero = false;
};
struct ImportedSpectrum {
    core::MeasuredSpectrum spectrum;
    std::optional<double> real_time, live_time;
};
IBEAMLAB_API ImportedSpectrum read_spectrum(const std::filesystem::path &, std::string measurement,
                                            const Parser &);
} // namespace ibeamlab::io
