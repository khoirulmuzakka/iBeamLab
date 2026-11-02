#pragma once
#include <filesystem>
#include <ibeamlab/fitting/result.h>
namespace ibeamlab::io {
/** Write a new directory of CSV sample, parameter, prediction and objective tables.
 * Never overwrites a result directory. The supplied provenance is saved verbatim. */
IBEAMLAB_API void write_result(const std::filesystem::path &, const fitting::Result &,
                               std::string provenance = {});
} // namespace ibeamlab::io
