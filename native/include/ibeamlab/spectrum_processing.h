#pragma once

/**
 * @file spectrum_processing.h
 * @brief Stateless operations on physical spectra and channel grids.
 *
 * Operations required by a deployed model belong here so native inference can
 * reproduce Python training inputs exactly.
 */

#include <cstddef>
#include <vector>

namespace ibeamlab::spectrum {

/** @brief Crops high channels or pads them to an exact size. */
std::vector<double> cropOrPad(const std::vector<double>& spectrum,
                              std::size_t size, double padding = 0.0);
/** @brief Concatenates spectra in caller-provided method order. */
std::vector<double> concatenate(const std::vector<std::vector<double>>& spectra);
/** @brief Clamps every channel to an inclusive numeric interval. */
std::vector<double> clip(const std::vector<double>& spectrum,double minimum,double maximum);

/** @brief Conservatively rebins counts between arbitrary monotonic bin edges. */
std::vector<double> rebin(
    const std::vector<double>& oldEdges,
    const std::vector<double>& newEdges,
    const std::vector<double>& spectrum);

/** @brief Applies the detector pileup model to a channel spectrum. */
std::vector<double> pileup(
    const std::vector<double>& spectrum,
    double realTime,
    double liveTime,
    double fudgeFactor,
    bool clipNegative = true);

/** @brief Converts an energy spectrum to channels and applies detector pileup. */
std::vector<double> energyToChannelAndPileup(
    const std::vector<double>& energySpectrum,
    double calibrationOffset,
    double calibrationLinear,
    double calibrationQuadratic,
    double realTime,
    double liveTime,
    double fudgeFactor,
    double scale = 1.0,
    bool clipNegative = true);

} // namespace ibeamlab::spectrum
