#include <ibeamlab/core/parameter.h>
#include <ibeamlab/core/units.h>
#include <ibeamlab/core/validation.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
using namespace ibeamlab::core;
void require(bool value) {
    if (!value)
        throw std::runtime_error("core test failed");
}
template <class F> void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected);
}
int main() {
    Experiment experiment;
    Layer layer;
    layer.thickness = 500;
    layer.species = {{"Si", .5, {}}, {"O", .5, {}}};
    experiment.sample.layers.push_back(layer);
    Measurement measurement;
    measurement.label = "RBS";
    measurement.beam = {"H", 2974, 0};
    experiment.setup.measurements.push_back(measurement);
    experiment.validate();
    require(validate(experiment).empty());
    auto second = measurement;
    second.label = "RBS_high";
    second.beam.energy = 4000;
    experiment.setup.measurements.push_back(second);
    experiment.validate();
    write_parameter(experiment, BeamEnergy{"RBS_high"}, 4100);
    require(read_parameter(experiment, BeamEnergy{"RBS"}) == 2974);
    require(read_parameter(experiment, BeamEnergy{"RBS_high"}) == 4100);
    write_parameter(experiment, DetectorResolution{"RBS_high"}, 20);
    write_parameter(experiment, ParticlesSr{"RBS_high"}, 1000);
    require(read_parameter(experiment, DetectorResolution{"RBS"}) == 0);
    require(read_parameter(experiment, DetectorResolution{"RBS_high"}) == 20);
    require(read_parameter(experiment, ParticlesSr{"RBS_high"}) == 1000);
    write_parameter(experiment, SpeciesConcentration{0, "Si"}, .25);
    rejects([&] { experiment.validate(); });
    write_parameter(experiment, SpeciesConcentration{0, "O"}, .75);
    experiment.validate();
    ParameterUpdate composition[] = {{SpeciesConcentration{0, "Si"}, .25},
                                     {SpeciesConcentration{0, "O"}, .75}};
    apply(experiment, composition);
    require(read_parameter(experiment, SpeciesConcentration{0, "Si"}) == .25);
    ParameterUpdate invalid[] = {{LayerThickness{0}, -1}};
    rejects([&] { apply(experiment, invalid); });
    require(experiment.sample.layers[0].thickness == 500);
    ParameterUpdate duplicate[] = {{LayerThickness{0}, 1}, {LayerThickness{0}, 2}};
    rejects([&] { apply(experiment, duplicate); });
    require(experiment.sample.layers[0].thickness == 500);
    rejects([&] { read_parameter(experiment, BeamEnergy{"unknown"}); });
    rejects([&] { read_parameter(experiment, LayerThickness{99}); });
    rejects([&] {
        write_parameter(experiment, LayerThickness{0}, std::numeric_limits<double>::quiet_NaN());
    });
    require(ParameterBounds{0, 1}.contains(.5));
    rejects([] { ParameterBounds{1, 0}.validate(); });
    experiment.sample.layers[0].species[0].concentration = .9;
    experiment.setup.measurements.push_back(measurement);
    const auto issues = validate(experiment);
    require(issues.size() == 2);
    require(issues[0].path == "sample.layers[0]");
    require(issues[1].path == "setup.measurements[2]");
    require(units::electronVoltsPerKeV == 1000);
    std::cout << "Core tests passed\n";
}
