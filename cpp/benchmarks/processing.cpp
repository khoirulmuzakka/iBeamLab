#include <ibeamlab/core/isotopes.h>
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
    std::get<model::ParameterInputSchema>(metadata.input_preparation).reference_sample.layers = {{100, 0, 0, 0, {{"Si", 1, {}}}}};
    metadata.reference_setup.measurements = {
        {"single-detector-measurement", {"He", 2000, 0}, {}, {1000, 10, 8}}};
    std::get<model::ParameterInputSchema>(metadata.input_preparation).parameters = {{"thickness", core::LayerThickness{0}, "1e15 atoms/cm2"}};
    metadata.output_spectra.spectra = {{"single-detector-measurement", channels}};
    core::Experiment experiment{std::get<model::ParameterInputSchema>(metadata.input_preparation).reference_sample, metadata.reference_setup};
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

    // Vary layer count after warming maximum capacity; neither packing nor
    // independent per-element preprocessing should allocate on the hot path.
    model::ForwardModelMetadata dynamic;
    dynamic.reference_setup = metadata.reference_setup;
    dynamic.input_preparation = model::LayerInputSchema{{"Si", "O"}};
    dynamic.input_transform.steps = {model::MultiplyScale{{.01, .02}}};
    dynamic.output_spectra = metadata.output_spectra;
    auto dynamic_package = model::Package::from_onnx(
        std::filesystem::path(IBEAMLAB_FIXTURES) / "forward_layers.onnx", model::ModelMetadata(dynamic));
    inference::ForwardModel sequence(dynamic_package);
    std::vector<core::Experiment> cases;
    for (std::size_t layers : {1u, 2u, 8u, 32u}) {
        core::Experiment sample;
        sample.setup = dynamic.reference_setup;
        sample.sample.layers.resize(layers, core::Layer{100, 0, 0, 0, {{"O", .5, {}}, {"Si", .5, {}}}});
        cases.push_back(std::move(sample));
    }
    simulation::Result layered_result;
    for (int i = 0; i < 10; ++i) sequence.predict(cases.back(), layered_result);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i) sequence.predict(cases[i % cases.size()], layered_result);
    counting = false;
    std::cout << channels << " channels, variable 1..32 layers: " << allocations
              << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0 && layered_result.spectra[0].counts.back() ==
        2.5 * cases[(iterations - 1) % cases.size()].sample.layers.size() * channels;

    std::vector<float> ad(32 * 2, 50.f);
    for (int i = 0; i < 10; ++i) sequence.predict_areal_density(ad, 32, layered_result);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i)
        sequence.predict_areal_density(std::span<const float>(ad).first(2 * cases[i % cases.size()].sample.layers.size()),
            cases[i % cases.size()].sample.layers.size(), layered_result);
    counting = false;
    std::cout << "Direct physical AD with preprocessing: " << allocations << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0;
    std::get<model::ForwardModelMetadata>(dynamic_package.metadata.model).input_transform.steps.clear();
    inference::ForwardModel borrowed(dynamic_package);
    for (int i = 0; i < 10; ++i) borrowed.predict_areal_density(ad, 32, layered_result);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i) borrowed.predict_areal_density(ad, 32, layered_result);
    counting = false;
    std::cout << "Direct AD without preprocessing (borrowed input): " << allocations << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0 && ad.front() == 50.f &&
        layered_result.spectra[0].counts.back() == 32. * 150 * channels;

    // A DE population uses one graph call and caller-owned physical output.
    constexpr std::size_t population = 64;
    std::vector<float> population_ad(population * 8 * 2, 50.f);
    std::vector<double> population_counts(population * channels);
    for (auto *network : {&sequence, &borrowed}) {
        for (int i = 0; i < 5; ++i)
            network->predict_areal_density_batch(population_ad, population, 8, population_counts);
        allocations = 0;
        start = std::chrono::steady_clock::now();
        counting = true;
        for (int i = 0; i < iterations; ++i)
            network->predict_areal_density_batch(population_ad, population, 8, population_counts);
        counting = false;
        elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "DE population 64 x 8 layers x 5000 channels ("
                  << (network == &sequence ? "preprocessed" : "borrowed") << "): "
                  << elapsed << " ms for " << iterations << " batches, " << allocations
                  << " C++ new allocations after warmup\n";
        const double expected = (network == &sequence ? 20. : 1200.) * channels;
        passed = passed && allocations == 0 && population_counts.back() == expected;
        // ARRDE reduces the active population without replacing its backing arrays.
        allocations = 0;
        counting = true;
        for (std::size_t active : {64u, 48u, 32u, 16u, 8u, 4u}) {
            auto input = std::span<const float>(population_ad).first(active * 8 * 2);
            auto output = std::span<double>(population_counts).first(active * channels);
            network->predict_areal_density_batch(input, active, 8, output);
            passed = passed && output.back() == expected;
        }
        counting = false;
        std::cout << "Reducing population 64..4 ("
                  << (network == &sequence ? "preprocessed" : "borrowed") << "): "
                  << allocations << " C++ new allocations after warmup\n";
        passed = passed && allocations == 0;
    }
    std::vector<simulation::Result> population_results(cases.size());
    sequence.predict_batch(cases, population_results);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i) sequence.predict_batch(cases, population_results);
    counting = false;
    std::cout << "Mixed-layer experiment batch: " << allocations << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0;
    for (auto &experiment : cases)
        for (auto &layer : experiment.sample.layers)
            for (auto &species : layer.species) {
                auto natural = core::natural_isotopes(species.element);
                species.isotopes.assign(natural.rbegin(), natural.rend());
            }
    sequence.predict_batch(cases, population_results);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i) sequence.predict_batch(cases, population_results);
    counting = false;
    std::cout << "Explicit natural-isotope batch: " << allocations << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0;
#endif
    return passed ? 0 : 1;
}
