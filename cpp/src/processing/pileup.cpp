#include <algorithm>
#include <ibeamlab/processing/pileup.h>
#include <numbers>
#include <numeric>
namespace ibeamlab::processing {
namespace {
void times(double t, double l, double tau) {
    if (!std::isfinite(t) || !std::isfinite(l) || !std::isfinite(tau) || t <= 0 || l <= 0 ||
        l > t || tau < 0)
        throw std::invalid_argument("invalid pileup times");
}
void fft(std::vector<std::complex<double>> &a, bool inverse) {
    const auto n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len *= 2) {
        const double angle = (inverse ? 2 : -2) * std::numbers::pi / static_cast<double>(len);
        const auto root = std::polar(1., angle);
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w = 1.;
            for (std::size_t j = 0; j < len / 2; ++j) {
                auto u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= root;
            }
        }
    }
    if (inverse)
        for (auto &v : a)
            v /= static_cast<double>(n);
}
void disjoint(std::span<const double> a, std::span<double> b) {
    auto x = reinterpret_cast<std::uintptr_t>(a.data()),
         y = reinterpret_cast<std::uintptr_t>(b.data());
    if (x < y + b.size_bytes() && y < x + a.size_bytes())
        throw std::invalid_argument("pileup output overlaps input");
}
} // namespace
void apply_pileup(std::span<const double> input, std::span<double> output, double t, double l,
                  double tau, PileupWorkspace &w) {
    times(t, l, tau);
    core::validate_counts(input);
    if (input.empty() || output.size() != 2 * input.size() - 1)
        throw std::invalid_argument("pileup requires 2*N-1 output");
    disjoint(input, output);
    const double total = std::accumulate(input.begin(), input.end(), 0.);
    const double u = tau * total / t;
    if (!std::isfinite(total) || !std::isfinite(u))
        throw std::invalid_argument("pileup count rate overflow");
    const double a = tau / t * std::exp(-u);
    std::size_t n = 1;
    while (n < output.size())
        n *= 2;
    w.fft.resize(n);
    std::fill(w.fft.begin(), w.fft.end(), 0.);
    for (std::size_t i = 0; i < input.size(); ++i)
        w.fft[i] = input[i];
    fft(w.fft, false);
    for (auto &v : w.fft)
        v *= v;
    fft(w.fft, true);
    for (std::size_t i = 0; i < output.size(); ++i) {
        const double x = i < input.size() ? input[i] : 0.;
        output[i] = (l / t) * ((1 - 2 * a * total) * x + a * std::max(0., w.fft[i].real()));
        if (!std::isfinite(output[i]))
            throw std::invalid_argument("pileup overflow");
    }
}
void remove_pileup(std::span<const double> input, std::span<double> output, double t, double l,
                   double tau, double tolerance) {
    times(t, l, tau);
    core::validate_counts(input);
    if (input.empty() || input.size() % 2 == 0 || output.size() != input.size() / 2 + 1 ||
        !std::isfinite(tolerance) || tolerance <= 0 || tolerance >= 1)
        throw std::invalid_argument("invalid pileup inverse dimensions or tolerance");
    disjoint(input, output);
    const double observed = std::accumulate(input.begin(), input.end(), 0.) * t / l;
    if (!std::isfinite(observed))
        throw std::invalid_argument("pileup total overflow");
    double lo = observed, hi = observed / (1 - 1 / std::numbers::e);
    for (int i = 0; i < 100; ++i) {
        double mid = (lo + hi) / 2, u = tau * mid / t;
        if (mid * (1 - u * std::exp(-u)) < observed)
            lo = mid;
        else
            hi = mid;
    }
    const double total = (lo + hi) / 2, u = tau * total / t;
    if (!std::isfinite(u))
        throw std::invalid_argument("pileup inverse count rate overflow");
    const double a = tau / t * std::exp(-u), b = 1 - 2 * a * total;
    double x0 = input[0] * t / l;
    output[0] = a == 0 ? x0 : 2 * x0 / (b + std::sqrt(b * b + 4 * a * x0));
    for (std::size_t i = 1; i < output.size(); ++i) {
        double sum = 0;
        for (std::size_t j = 1; j < i; ++j)
            sum += output[j] * output[i - j];
        double value = (input[i] * t / l - a * sum) / (b + 2 * a * output[0]);
        if (!std::isfinite(value) || value < -tolerance * std::max(1., total))
            throw std::invalid_argument("no nonnegative pileup inverse");
        output[i] = std::max(0., value);
    }
    // Full support must agree, including channels not used by the recurrence.
    for (std::size_t i = 0; i < input.size(); ++i) {
        double sum = 0;
        for (std::size_t j = (i >= output.size() ? i - output.size() + 1 : 0);
             j < output.size() && j <= i; ++j)
            sum += output[j] * output[i - j];
        double predicted = (l / t) * ((i < output.size() ? b * output[i] : 0) + a * sum);
        if (std::abs(predicted - input[i]) > tolerance * std::max(1., observed * l / t))
            throw std::invalid_argument("spectrum is inconsistent with full pileup model");
    }
}
} // namespace ibeamlab::processing
