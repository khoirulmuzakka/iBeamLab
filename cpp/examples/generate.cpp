#include <ibeamlab/generation/generator.h>
#include <iostream>
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: ibeamlab_generate NEW_DATASET_DIRECTORY\n";
        return 2;
    }
    try {
        ibeamlab::core::Experiment x;
        ibeamlab::core::Layer layer;
        layer.thickness = 100;
        layer.species = {{"Si", 1, {}}};
        x.sample.layers = {layer};
        ibeamlab::core::Measurement measurement;
        measurement.label = "RBS";
        measurement.beam = {"H", 2000, 0};
        x.setup.measurements = {measurement};
        ibeamlab::generation::GenerationPlan plan{
            x, {{"thickness", ibeamlab::core::LayerThickness{0}, "1e15 atoms/cm2"}}};
        ibeamlab::core::Counts values(100);
        ibeamlab::generation::Uniform range[]{{{100, 1000}}};
        ibeamlab::generation::sample_uniform(range, 100, 42, values);
        ibeamlab::simulation::DummySimulator simulator;
        auto summary = ibeamlab::generation::generate(plan, values, 100, simulator, argv[1],
                                                      ibeamlab::generation::FailurePolicy::Stop, {},
                                                      "sampler=uniform-v1;seed=42");
        std::cout << "Accepted " << summary.accepted << " records\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
