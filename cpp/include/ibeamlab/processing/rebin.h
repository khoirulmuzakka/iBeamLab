#pragma once
#include <ibeamlab/core/spectrum.h>
namespace ibeamlab::processing {
/** @brief Treat destination bins outside the source as zero, or reject that grid. */
enum class Coverage { ZeroOutside, RequireFull };
// Output must not overlap any input. Partial input bins are split uniformly
// in energy. Counts outside the output grid are discarded, never renormalized.
IBEAMLAB_API void rebin(std::span<const double> counts, std::span<const double> old_edges,
                             std::span<const double> new_edges, std::span<double> output,
                             Coverage coverage = Coverage::ZeroOutside);
IBEAMLAB_API core::Counts rebin(std::span<const double> counts,
                                     std::span<const double> old_edges,
                                     std::span<const double> new_edges,
                                     Coverage coverage = Coverage::ZeroOutside);
} // namespace ibeamlab::processing
