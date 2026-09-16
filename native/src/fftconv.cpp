#include "fftconv.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fftconv {

namespace {
    constexpr double PI = 3.141592653589793238462643383279502884;

    inline bool is_power_of_two(std::size_t n) noexcept {
        return n && ((n & (n - 1)) == 0);
    }

    // In-place bit-reversal permutation.
    inline void bit_reverse_permute(std::vector<cplx>& a) noexcept {
        const std::size_t n = a.size();
        for (std::size_t i = 1, j = 0; i < n; ++i) {
            std::size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j |= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
    }
} // namespace

void fft(std::vector<cplx>& a, bool inverse) {
    const std::size_t n = a.size();
    if (n == 0) return;
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("fft(): input size must be a power of two");
    }

    bit_reverse_permute(a);

    // Iterative Cooley–Tukey radix-2
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = (inverse ? -1.0 : 1.0) * (2.0 * PI / static_cast<double>(len));
        const cplx wlen(std::cos(ang), std::sin(ang));

        for (std::size_t i = 0; i < n; i += len) {
            cplx w(1.0, 0.0);
            const std::size_t half = len >> 1;

            // Tight inner loop; w is updated multiplicatively.
            for (std::size_t j = 0; j < half; ++j) {
                const cplx u = a[i + j];
                const cplx v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (cplx& x : a) x *= inv_n;
    }
}

void convolve_fft(const double* a, std::size_t N,
                  const double* b, std::size_t M,
                  double* out) {
    // Handle edge cases explicitly and thread-safely (no globals).
    if (N == 0 || M == 0) return;

    const std::size_t L = N + M - 1;

    // Next power of two >= L
    std::size_t P = 1;
    while (P < L) P <<= 1;

    std::vector<cplx> A(P, cplx(0.0, 0.0));
    std::vector<cplx> B(P, cplx(0.0, 0.0));

    for (std::size_t i = 0; i < N; ++i) A[i] = a[i];
    for (std::size_t i = 0; i < M; ++i) B[i] = b[i];

    fft(A, /*inverse=*/false);
    fft(B, /*inverse=*/false);

    for (std::size_t i = 0; i < P; ++i) A[i] *= B[i];

    fft(A, /*inverse=*/true);

    for (std::size_t i = 0; i < L; ++i) out[i] = A[i].real();
}

void convolve_fft_self_workspace(const double* a, std::size_t N,
                                 double* out,
                                 std::vector<cplx>& buffer) {
    if (N == 0) return;

    const std::size_t L = 2 * N - 1;
    std::size_t P = 1;
    while (P < L) P <<= 1;

    // Reuse buffer; ensure capacity and size
    if (buffer.size() != P) buffer.assign(P, cplx(0.0, 0.0));
    else {
        // zero only the first L elements is not sufficient; clear all P
        std::fill(buffer.begin(), buffer.end(), cplx(0.0, 0.0));
    }

    for (std::size_t i = 0; i < N; ++i) buffer[i] = a[i];

    fft(buffer, /*inverse=*/false);

    for (std::size_t i = 0; i < P; ++i) buffer[i] *= buffer[i];

    fft(buffer, /*inverse=*/true);

    for (std::size_t i = 0; i < L; ++i) out[i] = buffer[i].real();
}

} // namespace fftconv
