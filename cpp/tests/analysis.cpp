#include <ibeamlab/analysis/executor.h>
#include <ibeamlab/analysis/predictor.h>
#include <ibeamlab/io/results.h>
#include <iostream>
#include <stdexcept>
using namespace ibeamlab;
void check(bool v) {
    if (!v)
        throw std::runtime_error("analysis regression failed");
}
class Provider : public fitting::Predictor {
  public:
    std::size_t calls = 0;
    void predict(std::span<const core::Experiment> input,
                 std::span<simulation::Result> out) override {
        ++calls;
        for (std::size_t i = 0; i < input.size(); ++i) {
            out[i].failure.reset();
            out[i].spectra.clear();
            double value = 0;
            for (const auto &l : input[i].sample.layers)
                value += l.thickness * l.species[0].concentration;
            for (const auto &m : input[i].setup.measurements)
                out[i].spectra.push_back({m.label, {value, value}});
        }
    }
};
core::Experiment experiment() {
    core::Experiment e;
    core::Layer l;
    l.thickness = 100;
    l.species = {{"Si", .5, {}}, {"O", .5, {}}};
    e.sample.layers = {l};
    core::Measurement m;
    m.label = "A";
    m.beam = {"H", 2000, 0};
    e.setup.measurements = {m};
    return e;
}
int main() {
    try {
        auto e = experiment();
        auto b = e.setup.measurements[0];
        b.label = "B";
        e.setup.measurements.push_back(b);
        auto p = std::make_shared<Provider>(), q = std::make_shared<Provider>();
        analysis::Predictor composite({{{{"A"}, p}, {{"B"}, q}}});
        std::vector<core::Experiment> rows(3, e);
        std::vector<simulation::Result> out(3);
        composite.predict(rows, out);
        check(p->calls == 1 && q->calls == 1 && out[0].spectra.size() == 2);
        composite.predict({rows.data(), 1}, {out.data(), 1});
        check(p->calls == 2 && out[0].spectra[1].measurement == "B");
        fitting::Problem problem;
        problem.initial_sample = e.sample;
        problem.data.setup = e.setup;
        for (auto label : {"A", "B"})
            problem.data.spectra.push_back({label, {50, 50}, {}, {}, ""});
        problem.parameters.push_back(
            {"width", core::ParameterTarget{core::LayerThickness{0}}, {50, 200}, {}, {}});
        fitting::Objective objective(problem, std::make_shared<Provider>());
        double candidate = 120, cost = 0;
        objective.evaluate_batch({&candidate, 1}, 1, {&cost, 1});
        check(cost > 0);
        profile::Model profile;
        profile.layers.push_back({e.sample.layers[0],
                                  {{profile::FunctionKind::Polynomial, {1, -1}},
                                   {profile::FunctionKind::Polynomial, {0, 1}}},
                                  2});
        problem.profile = profile;
        problem.parameters.clear();
        problem.profile_parameters.push_back({"slope", 0, 0, 1, {-1, 0}});
        fitting::Objective profiled(problem, std::make_shared<Provider>());
        check(profiled.dimension() == 1 &&
              profiled.decode(profiled.initial_coordinates()).sample.layers.size() == 2);
        double slope = -.5;
        profiled.evaluate_batch({&slope, 1}, 1, {&cost, 1});
        check(std::isfinite(cost) && cost > 0);
        analysis::Executor executor(2, [](std::size_t) { return std::make_shared<Provider>(); });
        analysis::Task task;
        task.id = "one";
        task.problem = problem;
        task.options.max_evaluations = 1;
        auto second = task;
        second.id = "two";
        second.initial_from = {"one"};
        std::vector<analysis::Task> tasks{task, second};
        auto results = executor.run(tasks);
        check(results.size() == 2 && results[0].error.empty() && results[1].result.feasible());
        check(results[0].id == "one" && results[1].id == "two");
        std::cout << "composite batching, profile fitting and executor checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
