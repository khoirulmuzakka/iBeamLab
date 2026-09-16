#include <ibeamlab/preprocessing.h>

#include "fftconv.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ibeamlab::preprocessing {

std::vector<double> cropOrPad(const std::vector<double>& spectrum,
                              std::size_t size, double padding) {
    if (!std::isfinite(padding)) throw std::invalid_argument("padding must be finite");
    std::vector<double> result(size, padding);
    std::copy_n(spectrum.begin(), std::min(size, spectrum.size()), result.begin());
    return result;
}

std::vector<double> concatenate(const std::vector<std::vector<double>>& spectra) {
    std::size_t size = 0;
    for (const auto& spectrum : spectra) size += spectrum.size();
    std::vector<double> result;
    result.reserve(size);
    for (const auto& spectrum : spectra)
        result.insert(result.end(), spectrum.begin(), spectrum.end());
    return result;
}

std::vector<double> clip(const std::vector<double>& spectrum,double minimum,double maximum){
    if(!std::isfinite(minimum)||!std::isfinite(maximum)||minimum>maximum)throw std::invalid_argument("invalid clipping bounds");
    std::vector<double> result=spectrum;for(auto& value:result){if(!std::isfinite(value))throw std::invalid_argument("spectrum contains a non-finite value");value=std::clamp(value,minimum,maximum);}return result;
}

std::vector<double> rebin(const std::vector<double>& oldEdges,
                          const std::vector<double>& newEdges,
                          const std::vector<double>& spectrum) {
    if (oldEdges.size() != spectrum.size() + 1 || oldEdges.size() < 2 || newEdges.size() < 2)
        throw std::invalid_argument("bin edges and spectrum have incompatible sizes");
    for (std::size_t i = 1; i < oldEdges.size(); ++i)
        if (!(oldEdges[i] > oldEdges[i - 1])) throw std::invalid_argument("old edges must increase");
    for (std::size_t i = 1; i < newEdges.size(); ++i)
        if (!(newEdges[i] > newEdges[i - 1])) throw std::invalid_argument("new edges must increase");

    std::vector<double> output(newEdges.size() - 1, 0.0);
    std::size_t first = 0;
    for (std::size_t out = 0; out < output.size(); ++out) {
        while (first < spectrum.size() && oldEdges[first + 1] <= newEdges[out]) ++first;
        for (std::size_t in = first; in < spectrum.size() && oldEdges[in] < newEdges[out + 1]; ++in) {
            const double overlap = std::min(oldEdges[in + 1], newEdges[out + 1]) -
                                   std::max(oldEdges[in], newEdges[out]);
            if (overlap > 0.0) output[out] += spectrum[in] * overlap / (oldEdges[in + 1] - oldEdges[in]);
        }
    }
    return output;
}

std::vector<double> pileup(const std::vector<double>& spectrum, double realTime,
                           double liveTime, double fudgeFactor, bool clipNegative) {
    if (spectrum.empty()) return {};
    if (!std::isfinite(realTime) || realTime <= 0.0 || !std::isfinite(liveTime) || liveTime < 0.0)
        throw std::invalid_argument("real time must be positive and live time non-negative");
    if (!std::isfinite(fudgeFactor) || fudgeFactor < 0.0)
        throw std::invalid_argument("pileup fudge factor must be finite and non-negative");
    const auto count = std::accumulate(spectrum.begin(), spectrum.end(), 0.0);
    const double factor = fudgeFactor / realTime * std::exp(-(count / realTime) * fudgeFactor);
    const double liveRatio = liveTime / realTime;
    std::vector<double> convolution(2 * spectrum.size() - 1);
    std::vector<fftconv::cplx> workspace;
    fftconv::convolve_fft_self_workspace(spectrum.data(), spectrum.size(), convolution.data(), workspace);
    std::vector<double> output(convolution.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        const double original = i < spectrum.size() ? spectrum[i] : 0.0;
        output[i] = liveRatio * (original - 2.0 * factor * count * original + factor * convolution[i]);
        if (clipNegative && output[i] < 0.0) output[i] = 0.0;
    }
    return output;
}

std::vector<double> energyToChannelAndPileup(
    const std::vector<double>& energySpectrum, double offset, double linear,
    double quadratic, double realTime, double liveTime, double fudgeFactor,
    double scale, bool clipNegative) {
    if (energySpectrum.empty()) return {};
    const double maximumEnergy = static_cast<double>(energySpectrum.size());
    double channel = 0.0;
    if (std::abs(quadratic) < 1e-15) {
        if (linear <= 0.0) throw std::invalid_argument("linear calibration must be positive");
        channel = (maximumEnergy - offset) / linear;
    } else {
        const double discriminant = linear * linear - 4.0 * quadratic * (offset - maximumEnergy);
        if (discriminant <= 0.0) throw std::invalid_argument("calibration does not reach spectrum energy");
        channel = std::max((-linear + std::sqrt(discriminant)) / (2.0 * quadratic),
                           (-linear - std::sqrt(discriminant)) / (2.0 * quadratic));
    }
    const auto channelCount = static_cast<std::size_t>(std::max(1.0, std::ceil(channel)));
    std::vector<double> oldEdges(energySpectrum.size() + 1), newEdges(channelCount + 1);
    std::iota(oldEdges.begin(), oldEdges.end(), 0.0);
    for (std::size_t i = 0; i <= channelCount; ++i) {
        const double x = static_cast<double>(i);
        newEdges[i] = offset + linear * x + quadratic * x * x;
    }
    auto channelSpectrum = rebin(oldEdges, newEdges, energySpectrum);
    for (auto& value : channelSpectrum) value *= scale;
    return pileup(channelSpectrum, realTime, liveTime, fudgeFactor, clipNegative);
}

} // namespace ibeamlab::preprocessing
