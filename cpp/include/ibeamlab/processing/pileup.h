#pragma once
#include <complex>
#include <ibeamlab/core/spectrum.h>
namespace ibeamlab::processing {
/** @brief Reusable FFT and convolution storage; exclusively owned during each call.
 * Capacity grows as needed, allowing repeated spectra to reuse allocations.
 */
struct PileupWorkspace {
    std::vector<std::complex<double>> fft;
    core::Counts convolution;
};
// Pair-sum model on channel indices, not absolute energy; full support is 2*N-1.
// a=tau/T*exp(-tau*sum(x)/T), y=(L/T)*[(1-2*a*sum(x))*x+a*(x*x)].
// Requires 0<L<=T; nonnegative model regime tau*sum(x)/T<1. No clipping.
// Output must be disjoint from input; workspace is exclusively owned per call.
IBEAMLAB_API void apply_pileup(std::span<const double> input, std::span<double> output,
                                    double real_time, double live_time, double tau,
                                    PileupWorkspace &workspace);
// Requires complete 2*N-1 support. Verifies reconstruction against the model;
// noisy or truncated spectra may have no physical inverse and are rejected.
IBEAMLAB_API void remove_pileup(std::span<const double> input, std::span<double> output,
                                     double real_time, double live_time, double tau,
                                     double relative_tolerance = 1e-7);
} // namespace ibeamlab::processing
