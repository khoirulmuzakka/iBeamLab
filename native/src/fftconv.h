#ifndef CONV_H
#define CONV_H


#pragma once
#include <vector>
#include <complex>
#include <cstddef>

namespace fftconv {

using cplx = std::complex<double>;

// In-place radix-2 FFT.
// - a.size() must be a power of two (throws std::invalid_argument otherwise).
// - inverse=false -> forward FFT; inverse=true -> inverse FFT (normalized by N).
void fft(std::vector<cplx>& a, bool inverse = false);

// FFT-based convolution between two real sequences.
// out must have size N+M-1.
void convolve_fft(const double* a, std::size_t N,
                  const double* b, std::size_t M,
                  double* out);

// Optimized self-convolution: out = a (*) a (length 2N-1)
// Uses a reusable complex buffer to avoid repeated allocations.
void convolve_fft_self_workspace(const double* a, std::size_t N,
                                 double* out,
                                 std::vector<cplx>& buffer);

} // namespace fftconv


#endif
