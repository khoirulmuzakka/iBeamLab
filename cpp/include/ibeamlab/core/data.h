#pragma once
#include <cstdint>
#include <ibeamlab/core/spectrum.h>
namespace ibeamlab::core {
/** @brief Experimental counts, optional validity/variance arrays and source identity.
 * Empty valid means every channel is observed. Missing channels retain their
 * positions and are excluded through valid; they are never measured zero counts.
 */
struct IBEAMLAB_API MeasuredSpectrum {
    std::string measurement;
    Counts counts;
    std::vector<std::uint8_t> valid;
    Counts variances;
    std::string source;
    void validate() const;
};
/** @brief Observations of one shared, unknown sample; no sample assumption is stored. */
struct IBEAMLAB_API MeasuredData {
    ExperimentalSetup setup;
    std::vector<MeasuredSpectrum> spectra;
    void validate() const;
};
} // namespace ibeamlab::core
