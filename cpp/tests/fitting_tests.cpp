#include <cmath>
#include <ibeamlab/fitting/fit.h>
#include <iostream>
#include <stdexcept>
using namespace ibeamlab;
namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
class Linear final : public fitting::Predictor {
  public:
    std::vector<std::size_t> populations;
    void predict(std::span<const core::Experiment> inputs,
                 std::span<simulation::Result> outputs) override {
        populations.push_back(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            auto &out = outputs[i];
            out.spectra.resize(1);
            out.spectra[0].measurement = "RBS";
            out.spectra[0].counts.resize(4);
            for (std::size_t j = 0; j < 4; ++j)
                out.spectra[0].counts[j] = inputs[i].sample.layers[0].thickness * (j + 1);
        }
    }
};
fitting::Problem problem() {
    fitting::Problem p;
    p.initial_sample.layers = {{100, 0, 0, 0, {{"Si", .5, {}}, {"O", .5, {}}}}};
    p.data.setup.measurements = {{"RBS", {"H", 2000, 0}, {}, {}}};
    p.data.spectra = {{"RBS", {150, 300, 450, 600}, {}, {}, {}}};
    p.parameters = {{"thickness", core::ParameterTarget{core::LayerThickness{0}}, {10, 200}, {}}};
    return p;
}
} // namespace
int main() {
    try {
        auto backend = std::make_shared<Linear>();
        fitting::Objective objective(problem(), backend);
        std::vector<double> rows{100, 150, 210}, costs(3);
        objective.evaluate_batch(rows, 3, costs);
        check(costs[1] == 0 && std::isinf(costs[2]), "invalid candidate handling");
        check(backend->populations == std::vector<std::size_t>{1, 2},
              "valid population compaction");
        auto best = objective.best_result();
        check(best.experiment.sample.layers[0].thickness == 150, "best sample");
        check(best.prediction.spectra[0].counts == problem().data.spectra[0].counts,
              "best prediction");
        check(best.evaluations == 4 && best.invalid == 1, "evaluation counters");
        auto p = problem();
        p.parameters.clear();
        p.compositions = {{0, {"Si", "O"}, {{.2, .7}, {.3, .8}}, "mix"}};
        fitting::Objective composition(p, std::make_shared<Linear>());
        const std::vector<double> endpoint{1};
        auto decoded = composition.decode(endpoint);
        check(std::abs(decoded.sample.layers[0].species[0].concentration - .7) < 1e-12,
              "bounded composition");
        objective.request_stop();
        check(objective.best_result().status == "cancelled", "best retained on cancellation");
        if (fitting::arrde_available()) {
            fitting::Objective optimize(problem(), std::make_shared<Linear>());
            auto result = fitting::fit(optimize, {400, 16, 42, true});
            check(result.cost < 1e-3 && result.evaluations <= 400, "ARRDE recovery/budget");
        }
        std::cout << "Native fitting checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
