#pragma once

#include <cstddef>
#include <vector>

namespace ibeamlab::preprocessing {

std::vector<double> cropOrPad(const std::vector<double>& spectrum,
                              std::size_t size, double padding = 0.0);
std::vector<double> concatenate(const std::vector<std::vector<double>>& spectra);
std::vector<double> clip(const std::vector<double>& spectrum,double minimum,double maximum);

std::vector<double> rebin(
    const std::vector<double>& oldEdges,
    const std::vector<double>& newEdges,
    const std::vector<double>& spectrum);

std::vector<double> pileup(
    const std::vector<double>& spectrum,
    double realTime,
    double liveTime,
    double fudgeFactor,
    bool clipNegative = true);

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

} // namespace ibeamlab::preprocessing
