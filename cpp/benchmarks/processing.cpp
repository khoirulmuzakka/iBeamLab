#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ibeamlab/analysis/predictor.h>
#include <ibeamlab/core/isotopes.h>
#include <ibeamlab/fitting/fit.h>
#include <ibeamlab/processing/pileup.h>
#include <ibeamlab/processing/rebin.h>
#include <ibeamlab/profile/discretization.h>
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
namespace {
// Allocation-free provider isolates framework work from simulator/ORT internals.
class InventoryPredictor final : public ibeamlab::fitting::Predictor {
  public:
    void predict(std::span<const ibeamlab::core::Experiment> inputs,
                 std::span<ibeamlab::simulation::Result> outputs) override {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            double inventory = 0;
            for (const auto &layer : inputs[i].sample.layers)
                inventory += layer.thickness * layer.species[0].concentration;
            auto &out = outputs[i];
            out.failure.reset();
            out.spectra.resize(inputs[i].setup.measurements.size());
            for (std::size_t j = 0; j < out.spectra.size(); ++j) {
                auto &s = out.spectra[j];
                s.measurement = inputs[i].setup.measurements[j].label;
                s.counts.resize(5000);
                std::fill(s.counts.begin(), s.counts.end(), inventory);
            }
        }
    }
};
} // namespace
int main() {
#if defined(_WIN32) && defined(IBEAMLAB_BENCHMARK_SHARED)
    std::cout << "Allocation scope: executable only; use BUILD_SHARED_LIBS=OFF to audit library "
                 "C++ allocations on Windows.\n";
#endif
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
    ibeamlab::profile::Model profile_model;
    ibeamlab::core::Layer profile_layer;
    profile_layer.thickness = 1000;
    profile_layer.species = {{"Si", .5, {}}, {"O", .5, {}}};
    profile_model.layers.push_back({profile_layer,
                                    {{ibeamlab::profile::FunctionKind::Polynomial, {1, -1}},
                                     {ibeamlab::profile::FunctionKind::Polynomial, {0, 1}}},
                                    32});
    ibeamlab::profile::Discretizer profile_grid(profile_model);
    ibeamlab::core::Sample discretized;
    profile_grid.apply(profile_model, discretized);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i)
        profile_grid.apply(profile_model, discretized);
    counting = false;
    std::cout << "profile discretization: " << allocations << " allocations after warmup\n";
    passed = passed && allocations == 0;

    // Exercise the new framework path, not just the discretizer: different
    // detector counts per route, long labels, shrinking/regrowing populations,
    // and improvements that update the retained best prediction.
    {
        using namespace ibeamlab;
        fitting::Problem routed;
        routed.profile = profile_model;
        routed.initial_sample = profile_model.reference_sample();
        for (const auto *label : {"route-a-long-measurement", "route-b-first-measurement",
                                  "route-b-second-measurement"}) {
            core::Measurement m;
            m.label = label;
            m.beam = {"H", 2000, 0};
            routed.data.setup.measurements.push_back(m);
            routed.data.spectra.push_back({label, core::Counts(channels, 1), {}, {}, {}});
        }
        routed.parameters = {
            {"thickness", core::ParameterTarget{core::LayerThickness{0}}, {100, 2000}, {}}};
        auto one = std::make_shared<InventoryPredictor>(),
             two = std::make_shared<InventoryPredictor>();
        auto provider = std::make_shared<analysis::Predictor>(
            analysis::Plan{{{{"route-a-long-measurement"}, one},
                            {{"route-b-first-measurement", "route-b-second-measurement"}, two}}});
        fitting::Objective objective(std::move(routed), provider);
        std::vector<double> rows(64, 1100), costs(64);
        objective.evaluate_batch(rows, 64, costs);
        allocations = 0;
        start = std::chrono::steady_clock::now();
        counting = true;
        for (int repeat = 0; repeat < 5; ++repeat)
            for (std::size_t active : {64u, 48u, 32u, 16u, 8u, 4u}) {
                std::fill(rows.begin(), rows.end(), 900 - repeat * 100 + double(active) / 64);
                objective.evaluate_batch(std::span<const double>(rows).first(active), active,
                                         std::span<double>(costs).first(active));
            }
        counting = false;
        elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::cout << "Profile + mixed-detector routing + objective, 5000 channels/detector, "
                     "populations 64..4: "
                  << elapsed << " ms, " << allocations << " C++ new allocations after warmup\n";
        passed = passed && allocations == 0 && std::isfinite(costs.front());
    }

#ifdef IBEAMLAB_BENCHMARK_ONNX
    using namespace ibeamlab;
    model::ForwardModelMetadata metadata;
    std::get<model::ParameterInputSchema>(metadata.input_preparation).reference_sample.layers = {
        {100, 0, 0, 0, {{"Si", 1, {}}}}};
    metadata.reference_setup.measurements = {
        {"single-detector-measurement", {"He", 2000, 0}, {}, {1000, 10, 8}}};
    std::get<model::ParameterInputSchema>(metadata.input_preparation).parameters = {
        {"thickness", core::LayerThickness{0}, "1e15 atoms/cm2"}};
    metadata.output_spectra.spectra = {{"single-detector-measurement", channels}};
    core::Experiment experiment{
        std::get<model::ParameterInputSchema>(metadata.input_preparation).reference_sample,
        metadata.reference_setup};
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
    auto dynamic_package =
        model::Package::from_onnx(std::filesystem::path(IBEAMLAB_FIXTURES) / "forward_layers.onnx",
                                  model::ModelMetadata(dynamic));
    inference::ForwardModel sequence(dynamic_package);
    std::vector<core::Experiment> cases;
    for (std::size_t layers : {1u, 2u, 8u, 32u}) {
        core::Experiment sample;
        sample.setup = dynamic.reference_setup;
        sample.sample.layers.resize(layers,
                                    core::Layer{100, 0, 0, 0, {{"O", .5, {}}, {"Si", .5, {}}}});
        cases.push_back(std::move(sample));
    }
    simulation::Result layered_result;
    for (int i = 0; i < 10; ++i)
        sequence.predict(cases.back(), layered_result);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i)
        sequence.predict(cases[i % cases.size()], layered_result);
    counting = false;
    std::cout << channels << " channels, variable 1..32 layers: " << allocations
              << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0 &&
             layered_result.spectra[0].counts.back() ==
                 2.5 * cases[(iterations - 1) % cases.size()].sample.layers.size() * channels;

    std::vector<float> ad(32 * 2, 50.f);
    for (int i = 0; i < 10; ++i)
        sequence.predict_areal_density(ad, 32, layered_result);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i)
        sequence.predict_areal_density(
            std::span<const float>(ad).first(2 * cases[i % cases.size()].sample.layers.size()),
            cases[i % cases.size()].sample.layers.size(), layered_result);
    counting = false;
    std::cout << "Direct physical AD with preprocessing: " << allocations
              << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0;
    std::get<model::ForwardModelMetadata>(dynamic_package.metadata.model)
        .input_transform.steps.clear();
    inference::ForwardModel borrowed(dynamic_package);
    for (int i = 0; i < 10; ++i)
        borrowed.predict_areal_density(ad, 32, layered_result);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i)
        borrowed.predict_areal_density(ad, 32, layered_result);
    counting = false;
    std::cout << "Direct AD without preprocessing (borrowed input): " << allocations
              << " C++ new allocations after warmup\n";
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
        elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::cout << "DE population 64 x 8 layers x 5000 channels ("
                  << (network == &sequence ? "preprocessed" : "borrowed") << "): " << elapsed
                  << " ms for " << iterations << " batches, " << allocations
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
                  << (network == &sequence ? "preprocessed" : "borrowed") << "): " << allocations
                  << " C++ new allocations after warmup\n";
        passed = passed && allocations == 0;
    }
    std::vector<simulation::Result> population_results(cases.size());
    sequence.predict_batch(cases, population_results);
    allocations = 0;
    counting = true;
    for (int i = 0; i < iterations; ++i)
        sequence.predict_batch(cases, population_results);
    counting = false;
    std::cout << "Mixed-layer experiment batch: " << allocations
              << " C++ new allocations after warmup\n";
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
    for (int i = 0; i < iterations; ++i)
        sequence.predict_batch(cases, population_results);
    counting = false;
    std::cout << "Explicit natural-isotope batch: " << allocations
              << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0;
    fitting::Problem fit_problem;
    fit_problem.data.setup = experiment.setup;
    fit_problem.data.spectra = {
        {"single-detector-measurement", core::Counts(channels, 1), {}, {}, {}}};
    fit_problem.initial_sample = experiment.sample;
    fit_problem.parameters = {
        {"thickness", core::ParameterTarget{core::LayerThickness{0}}, {1, 1000}, {}}};
    fitting::Objective objective(std::move(fit_problem),
                                 std::make_shared<fitting::ForwardPredictor>(model));
    std::vector<double> fit_rows(64, 200), fit_costs(64);
    objective.evaluate_batch(fit_rows, 64, fit_costs);
    allocations = 0;
    counting = true;
    for (int repeat = 0; repeat < 5; ++repeat)
        for (std::size_t n = 64; n >= 4; n -= 4) {
            std::fill(fit_rows.begin(), fit_rows.end(), 190 - repeat * 10 + double(n) / 64);
            objective.evaluate_batch(std::span<const double>(fit_rows.data(), n), n,
                                     std::span<double>(fit_costs.data(), n));
        }
    counting = false;
    std::cout << "Fitting objective reducing population 64..4, 5000 channels: " << allocations
              << " C++ new allocations after warmup\n";
    passed = passed && allocations == 0;
#endif
    return passed ? 0 : 1;
}
