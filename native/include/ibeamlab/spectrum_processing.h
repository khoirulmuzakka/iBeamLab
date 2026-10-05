#pragma once

/**
 * @file spectrum_processing.h
 * @brief Stateless operations on physical spectra and channel grids.
 *
 * Operations required by a deployed model belong here so native inference can
 * reproduce Python training inputs exactly.
 */

#include <cstddef>
#include <ibeamlab/export.h>
#include <vector>

namespace ibeamlab::spectrum {

/** @brief Crops high channels or pads them to an exact size. */
IBEAMLAB_API std::vector<double> cropOrPad(const std::vector<double> &spectrum, std::size_t size,
                                           double padding = 0.0);
/** @brief Concatenates spectra in caller-provided method order. */
IBEAMLAB_API std::vector<double> concatenate(const std::vector<std::vector<double>> &spectra);
/** @brief Clamps every channel to an inclusive numeric interval. */
IBEAMLAB_API std::vector<double> clip(const std::vector<double> &spectrum, double minimum,
                                      double maximum);

/** @brief Conservatively rebins counts between arbitrary monotonic bin edges. */
IBEAMLAB_API std::vector<double> rebin(const std::vector<double> &oldEdges,
                                       const std::vector<double> &newEdges,
                                       const std::vector<double> &spectrum);

/** @brief Applies the detector pileup model; realTime, liveTime and fudgeFactor are seconds. */
IBEAMLAB_API std::vector<double> pileup(const std::vector<double> &spectrum, double realTime,
                                        double liveTime, double fudgeFactor,
                                        bool clipNegative = true);

/** @brief Reverses pileup() including its live/real-time scaling.
 * Requires the complete 2*N-1-channel spectrum, including the pileup tail;
 * returns N original channels. Times and fudgeFactor are in seconds and both
 * times must be positive. Empty spectra return empty. Input must be finite and
 * nonnegative. relativeTolerance allows rounding noise, not a noisy-data fit.
 * Rejects spectra inconsistent with this model; cropped tails cannot use this
 * inversion. Small negative reconstruction roundoff is clamped to zero.
 * Uses O(N^2) channel reconstruction and an FFT consistency check.
 */
IBEAMLAB_API std::vector<double> removePileup(const std::vector<double> &spectrum, double realTime,
                                              double liveTime, double fudgeFactor,
                                              double relativeTolerance = 1e-6);

/** @brief Converts an energy spectrum to channels and applies detector pileup. */
IBEAMLAB_API std::vector<double>
energyToChannelAndPileup(const std::vector<double> &energySpectrum, double calibrationOffset,
                         double calibrationLinear, double calibrationQuadratic, double realTime,
                         double liveTime, double fudgeFactor, double scale = 1.0,
                         bool clipNegative = true);

} // namespace ibeamlab::spectrum
