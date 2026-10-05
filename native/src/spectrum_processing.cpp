#include <ibeamlab/spectrum_processing.h>

#include "fftconv.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ibeamlab::spectrum {

std::vector<double> cropOrPad(const std::vector<double> &spectrum, std::size_t size,
                              double padding) {
    if (!std::isfinite(padding))
        throw std::invalid_argument("padding must be finite");
    std::vector<double> result(size, padding);
    std::copy_n(spectrum.begin(), std::min(size, spectrum.size()), result.begin());
    return result;
}

std::vector<double> concatenate(const std::vector<std::vector<double>> &spectra) {
    std::size_t size = 0;
    for (const auto &spectrum : spectra)
        size += spectrum.size();
    std::vector<double> result;
    result.reserve(size);
    for (const auto &spectrum : spectra)
        result.insert(result.end(), spectrum.begin(), spectrum.end());
    return result;
}

std::vector<double> clip(const std::vector<double> &spectrum, double minimum, double maximum) {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum)
        throw std::invalid_argument("invalid clipping bounds");
    std::vector<double> result = spectrum;
    for (auto &value : result) {
        if (!std::isfinite(value))
            throw std::invalid_argument("spectrum contains a non-finite value");
        value = std::clamp(value, minimum, maximum);
    }
    return result;
}

std::vector<double> rebin(const std::vector<double> &oldEdges, const std::vector<double> &newEdges,
                          const std::vector<double> &spectrum) {
    if (oldEdges.size() != spectrum.size() + 1 || oldEdges.size() < 2 || newEdges.size() < 2)
        throw std::invalid_argument("bin edges and spectrum have incompatible sizes");
    for (std::size_t i = 1; i < oldEdges.size(); ++i)
        if (!(oldEdges[i] > oldEdges[i - 1]))
            throw std::invalid_argument("old edges must increase");
    for (std::size_t i = 1; i < newEdges.size(); ++i)
        if (!(newEdges[i] > newEdges[i - 1]))
            throw std::invalid_argument("new edges must increase");

    std::vector<double> output(newEdges.size() - 1, 0.0);
    std::size_t first = 0;
    for (std::size_t out = 0; out < output.size(); ++out) {
        while (first < spectrum.size() && oldEdges[first + 1] <= newEdges[out])
            ++first;
        for (std::size_t in = first; in < spectrum.size() && oldEdges[in] < newEdges[out + 1];
             ++in) {
            const double overlap = std::min(oldEdges[in + 1], newEdges[out + 1]) -
                                   std::max(oldEdges[in], newEdges[out]);
            if (overlap > 0.0)
                output[out] += spectrum[in] * overlap / (oldEdges[in + 1] - oldEdges[in]);
        }
    }
    return output;
}

std::vector<double> pileup(const std::vector<double> &spectrum, double realTime, double liveTime,
                           double fudgeFactor, bool clipNegative) {
    if (spectrum.empty())
        return {};
    if (!std::isfinite(realTime) || realTime <= 0.0 || !std::isfinite(liveTime) || liveTime < 0.0)
        throw std::invalid_argument("real time must be positive and live time non-negative");
    if (!std::isfinite(fudgeFactor) || fudgeFactor < 0.0)
        throw std::invalid_argument("pileup fudge factor must be finite and non-negative");
    const auto count = std::accumulate(spectrum.begin(), spectrum.end(), 0.0);
    const double factor = fudgeFactor / realTime * std::exp(-(count / realTime) * fudgeFactor);
    const double liveRatio = liveTime / realTime;
    std::vector<double> convolution(2 * spectrum.size() - 1);
    std::vector<fftconv::cplx> workspace;
    fftconv::convolve_fft_self_workspace(spectrum.data(), spectrum.size(), convolution.data(),
                                         workspace);
    std::vector<double> output(convolution.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        const double original = i < spectrum.size() ? spectrum[i] : 0.0;
        output[i] =
            liveRatio * (original - 2.0 * factor * count * original + factor * convolution[i]);
        if (clipNegative && output[i] < 0.0)
            output[i] = 0.0;
    }
    return output;
}

std::vector<double> removePileup(const std::vector<double> &spectrum, double realTime,
                                 double liveTime, double fudgeFactor, double relativeTolerance) {
    if (!std::isfinite(realTime) || realTime <= 0 || !std::isfinite(liveTime) || liveTime <= 0)
        throw std::invalid_argument("pileup removal requires positive finite real and live times");
    if (!std::isfinite(fudgeFactor) || fudgeFactor < 0 || !std::isfinite(relativeTolerance) ||
        relativeTolerance <= 0 || relativeTolerance >= 1)
        throw std::invalid_argument("invalid pileup removal factor or tolerance");
    if (spectrum.empty())
        return {};
    if (spectrum.size() % 2 == 0)
        throw std::invalid_argument("pileup removal requires the full 2*N-1-channel spectrum");
    const double liveRatio = liveTime / realTime;
    if (!std::isfinite(liveRatio) || liveRatio <= 0)
        throw std::invalid_argument("live/real time ratio is not representable");
    std::vector<double> observed(spectrum.size());
    double total = 0;
    for (std::size_t i = 0; i < spectrum.size(); ++i) {
        if (!std::isfinite(spectrum[i]) || spectrum[i] < 0)
            throw std::invalid_argument("pileup spectrum must be finite and nonnegative");
        observed[i] = spectrum[i] / liveRatio;
        total += observed[i];
    }
    if (!std::isfinite(total))
        throw std::invalid_argument("pileup spectrum total is not representable");
    const std::size_t size = spectrum.size() / 2 + 1;
    if (total == 0)
        return std::vector<double>(size, 0);

    // With complete convolution support, Z=sum(y/liveRatio)=N*(1-h),
    // h=u*exp(-u), u=tau*N/realTime. 0<=h<=1/e; Z(N) is strictly
    // increasing. Solve for N/Z in [1, 1/(1-1/e)] to avoid large bounds.
    auto pileupFraction = [&](double count) {
        if (fudgeFactor == 0)
            return 0.0;
        // Compute the rate in log space to avoid overflow in count/realTime
        // when a tiny fudge factor would otherwise bring it back into range.
        const double logRate = std::log(count) + std::log(fudgeFactor) - std::log(realTime);
        if (logRate > std::log(745.0))
            return 0.0;
        const double u = std::exp(logRate);
        // Beyond this point h is negligible; avoid inf*0 for extreme rates.
        return u > 745 ? 0.0 : u * std::exp(-u);
    };
    double lower = 1, upper = 1 / (1 - std::exp(-1.0));
    for (int iteration = 0; iteration < 100; ++iteration) {
        const double ratio = lower + (upper - lower) / 2;
        if (ratio * (1 - pileupFraction(total * ratio)) < 1)
            lower = ratio;
        else
            upper = ratio;
    }
    const double count = total * (lower + (upper - lower) / 2);
    if (!std::isfinite(count))
        throw std::invalid_argument("original spectrum total is not representable");
    const double h = fudgeFactor == 0 ? 0 : pileupFraction(count);
    const double linear = 1 - 2 * h; // Always >= 1-2/e > 0.
    for (auto &value : observed)
        value /= count;
    const double tolerance =
        relativeTolerance * *std::max_element(observed.begin(), observed.end());
    std::vector<double> probabilities(size);
    // Stable positive root of h*p0^2 + linear*p0 = observed[0].
    probabilities[0] =
        2 * observed[0] / (linear + std::sqrt(linear * linear + 4 * h * observed[0]));
    const double denominator = linear + 2 * h * probabilities[0];
    for (std::size_t i = 1; i < size; ++i) {
        double interior = 0;
        for (std::size_t j = 1; j < i; ++j)
            interior += probabilities[j] * probabilities[i - j];
        const double residual = observed[i] - h * interior;
        if (residual < -tolerance)
            throw std::runtime_error("spectrum is inconsistent with nonnegative pileup removal");
        probabilities[i] = std::max(0.0, residual) / denominator;
    }
    // The tail provides a model check: cropping or arbitrary spectra must not
    // silently yield an allegedly recovered original.
    std::vector<double> convolution(spectrum.size());
    std::vector<fftconv::cplx> workspace;
    fftconv::convolve_fft_self_workspace(probabilities.data(), size, convolution.data(), workspace);
    for (std::size_t i = 0; i < spectrum.size(); ++i) {
        const double reconstructed =
            linear * (i < size ? probabilities[i] : 0) + h * convolution[i];
        if (!std::isfinite(reconstructed) || std::abs(reconstructed - observed[i]) > tolerance)
            throw std::runtime_error("spectrum does not contain a consistent full pileup tail");
    }
    for (auto &value : probabilities)
        value *= count;
    return probabilities;
}

std::vector<double> energyToChannelAndPileup(const std::vector<double> &energySpectrum,
                                             double offset, double linear, double quadratic,
                                             double realTime, double liveTime, double fudgeFactor,
                                             double scale, bool clipNegative) {
    if (energySpectrum.empty())
        return {};
    const double maximumEnergy = static_cast<double>(energySpectrum.size());
    double channel = 0.0;
    if (std::abs(quadratic) < 1e-15) {
        if (linear <= 0.0)
            throw std::invalid_argument("linear calibration must be positive");
        channel = (maximumEnergy - offset) / linear;
    } else {
        const double discriminant = linear * linear - 4.0 * quadratic * (offset - maximumEnergy);
        if (discriminant <= 0.0)
            throw std::invalid_argument("calibration does not reach spectrum energy");
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
    for (auto &value : channelSpectrum)
        value *= scale;
    return pileup(channelSpectrum, realTime, liveTime, fudgeFactor, clipNegative);
}

} // namespace ibeamlab::spectrum
