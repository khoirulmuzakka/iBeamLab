#include <algorithm>
#include <cstdint>
#include <ibeamlab/processing/rebin.h>
namespace ibeamlab::processing {
namespace {
void edges_valid(std::span<const double> edges) {
    if (edges.size() < 2)
        throw std::invalid_argument("too few edges");
    for (std::size_t i = 0; i < edges.size(); ++i)
        if (!std::isfinite(edges[i]) || (i && edges[i] <= edges[i - 1]))
            throw std::invalid_argument("edges must be finite and increasing");
}
bool overlap(std::span<const double> a, std::span<const double> b) {
    if (a.empty() || b.empty())
        return false;
    auto x = reinterpret_cast<std::uintptr_t>(a.data()),
         y = reinterpret_cast<std::uintptr_t>(b.data());
    return x < y + b.size_bytes() && y < x + a.size_bytes();
}
} // namespace
void rebin(std::span<const double> counts, std::span<const double> old_edges,
           std::span<const double> new_edges, std::span<double> output, Coverage coverage) {
    edges_valid(old_edges);
    edges_valid(new_edges);
    core::validate_counts(counts);
    if (counts.size() != old_edges.size() - 1 || output.size() != new_edges.size() - 1)
        throw std::invalid_argument("incompatible rebin dimensions");
    if (overlap(output, counts) || overlap(output, old_edges) || overlap(output, new_edges))
        throw std::invalid_argument("rebin output overlaps input");
    if (coverage == Coverage::RequireFull &&
        (new_edges.front() < old_edges.front() || new_edges.back() > old_edges.back()))
        throw std::invalid_argument("input does not cover output grid");
    std::fill(output.begin(), output.end(), 0.);
    std::size_t i = 0, j = 0;
    while (i < counts.size() && j < output.size()) {
        const double lo = std::max(old_edges[i], new_edges[j]);
        const double hi = std::min(old_edges[i + 1], new_edges[j + 1]);
        if (hi > lo)
            output[j] += counts[i] * ((hi - lo) / (old_edges[i + 1] - old_edges[i]));
        if (old_edges[i + 1] < new_edges[j + 1])
            ++i;
        else
            ++j;
    }
}
core::Counts rebin(std::span<const double> counts, std::span<const double> old_edges,
                   std::span<const double> new_edges, Coverage coverage) {
    if (new_edges.size() < 2)
        throw std::invalid_argument("too few edges");
    core::Counts output(new_edges.size() - 1);
    rebin(counts, old_edges, new_edges, output, coverage);
    return output;
}
} // namespace ibeamlab::processing
