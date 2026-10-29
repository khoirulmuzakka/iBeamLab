#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ibeamlab/processing/pileup.h>
#include <ibeamlab/processing/rebin.h>
#include <iostream>
#include <new>
#ifdef IBEAMLAB_BENCHMARK_ONNX
#include <ibeamlab/inference/models.h>
#endif
std::atomic_size_t allocations{0};
bool counting = false;
void *operator new(std::size_t size) {
    if (counting)
        ++allocations;
    if (void *p = std::malloc(size ? size : 1))
        return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { ::operator delete(p); }
int main() {
    constexpr std::size_t channels = 5000;
    constexpr int iterations = 100;
    ibeamlab::core::Counts input(channels, 1), edges(channels + 1), rebinned(channels),
        piled(2 * channels - 1);
    for (std::size_t i = 0; i < edges.size(); ++i)
        edges[i] = static_cast<double>(i) - .5;
    ibeamlab::processing::PileupWorkspace workspace;
    ibeamlab::processing::rebin(input, edges, edges, rebinned);
    ibeamlab::processing::apply_pileup(input, piled, 10, 8, 1e-6, workspace);
    auto start = std::chrono::steady_clock::now();
    counting = true;
    for (int i = 0; i < iterations; ++i) {
        ibeamlab::processing::rebin(input, edges, edges, rebinned);
        ibeamlab::processing::apply_pileup(input, piled, 10, 8, 1e-6, workspace);
    }
    counting = false;
    double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << channels << " channels, " << iterations << " rebin+pileup iterations: " << elapsed
              << " ms, " << allocations << " allocations after warmup\n";
    bool passed = allocations == 0 && rebinned == input;
#ifdef IBEAMLAB_BENCHMARK_ONNX
    using namespace ibeamlab;
    model::ForwardModelMetadata metadata;
    metadata.reference.sample.layers = {{100, 0, 0, 0, {{"Si", 1, {}}}}};
    metadata.reference.setup.measurements = {
        {"single-detector-measurement", {"He", 2000, 0}, {}, {1000, 10, 8}}};
    metadata.input_parameters = {{"thickness", core::LayerThickness{0}, "1e15 atoms/cm2"}};
    metadata.output_spectra.spectra = {{"single-detector-measurement", channels}};
    auto experiment = metadata.reference;
    inference::ForwardModel model(
        model::Package::from_onnx(std::filesystem::path(IBEAMLAB_FIXTURES) / "forward_5000.onnx",
                                  model::ModelMetadata(metadata)));
    simulation::Result result;
    for (int i = 0; i < 10; ++i)
        model.predict(experiment, result);
    allocations = 0;
    start = std::chrono::steady_clock::now();
    counting = true;
    for (int i = 0; i < iterations; ++i) {
        experiment.sample.layers[0].thickness = 100 + i;
        model.predict(experiment, result);
    }
    counting = false;
    elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << channels << " channels, " << iterations
              << " single-detector predictions: " << elapsed << " ms, " << allocations
              << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0 && result.spectra[0].counts.size() == channels &&
             result.spectra[0].counts.back() == (100 + iterations - 1) * channels;
#endif
    return passed ? 0 : 1;
}
