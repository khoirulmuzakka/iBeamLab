#include <ibeamlab/datasets.h>
#include <ibeamlab/data_generator.h>
#include <ibeamlab/inference.h>
#include <ibeamlab/preprocessing.h>
#include <ibeamlab/transforms.h>
#include <ibeamlab/sample_toml.h>
#include <ibeamlab/sample.h>
#include <ibeamlab/simulator.h>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

template <class F> bool throws(F &&function) {
    try {
        function();
    } catch (...) {
        return true;
    }
    return false;
}

int main() {
    using namespace ibeamlab;
    const auto rebinned = preprocessing::rebin({0, 1, 2}, {0, 2}, {2, 4});
    assert(rebinned.size() == 1 && std::abs(rebinned[0] - 6.0) < 1e-12);
    preprocessing::StandardScaler scaler({1, 2}, {2, 4});
    const auto scaled = scaler.apply({{3, 6}});
    assert(scaled[0][0] == 1 && scaled[0][1] == 1);
    assert(scaler.inverse(scaled)[0][0] == 3);
    preprocessing::LogTransform logarithm(2);
    const auto logged = logarithm.apply({{0, 3}});
    assert(std::abs(logarithm.inverse(logged)[0][1] - 3) < 1e-5);
    assert(throws([&] { logarithm.apply({{-2, 0}}); }));
    assert(preprocessing::cropOrPad({1, 2, 3}, 2) == std::vector<double>({1, 2}));
    assert(preprocessing::cropOrPad({1}, 3, -1) == std::vector<double>({1, -1, -1}));
    assert(preprocessing::concatenate({{1, 2}, {3}}) == std::vector<double>({1, 2, 3}));
    assert(throws([] { preprocessing::rebin({0, 1, 1}, {0, 1}, {1, 2}); }));

    sample::SampleModel model{{sample::Layer{1.0, 0, 0, 0, {sample::Species{"Si", 1.0, {}}}}}};
    sample::ExperimentalSetup setup{
        {sample::Detector{"RBS", sample::Beam{"He", 100.0, 0.0}, 1.0, 1e9, 0, 0, 5.0}}};
    const auto modelRoundTrip = sample::sampleModelFromToml(sample::toToml(model));
    const auto setupRoundTrip = sample::experimentalSetupFromToml(sample::toToml(setup));
    assert(modelRoundTrip.layers[0].species[0].element == "Si");
    assert(setupRoundTrip.detectors[0].beam.energy == 100.0);
    simulator::DummySimulator simulator(128);
    const auto results = simulator.simulateBatch({{model, setup}});
    assert(results.size() == 1 && results[0].spectra.size() == 1);
    assert(results[0].spectra[0].counts.size() == 128);

    generation::GenerationConfig config{
        model,
        setup,
        {generation::MethodConfig{"RBS", "RBS", "reference.xnra"}},
        {generation::ParameterSpec{"energy", generation::BeamEnergy{"RBS"}, 90, 110, std::nullopt,
                                   "keV"}}};
    config.validate();
    const auto generated = config.materialize({105});
    assert(generated.setup.detectors[0].beam.energy == 105);

    sample::SampleModel mixture{{sample::Layer{1.0, 0, 0, 0,
        {sample::Species{"C", 0.4, {}}, sample::Species{"O", 0.4, {}},
         sample::Species{"Zr", 0.2, {}}}}}};
    generation::GenerationConfig mixtureConfig{
        mixture,
        setup,
        {generation::MethodConfig{"RBS", "RBS", "reference.xnra"}},
        {
            generation::ParameterSpec{"C", generation::SpeciesConcentration{0, "C"}, 0, 0.8,
                                      std::nullopt, "fraction"},
            generation::ParameterSpec{"Zr", generation::SpeciesConcentration{0, "Zr"}, 0, 1,
                                      0.2, "fraction"},
            generation::ParameterSpec{"O", generation::SpeciesConcentration{0, "O"}, 0, 0.8,
                                      std::nullopt, "fraction"},
        }};
    const auto concentrationRows = generation::sampleParameters(mixtureConfig, 32, 7);
    for (const auto& row : concentrationRows) {
        const auto normalized = mixtureConfig.materialize(row);
        const auto& species = normalized.sample.layers[0].species;
        assert(std::abs(species[0].concentration + species[1].concentration - 0.8) < 1e-9);
        assert(std::abs(species[2].concentration - 0.2) < 1e-12);
    }
    assert(std::abs(concentrationRows[0][0] - concentrationRows[1][0]) > 1e-9);

    const auto variableDirectory =
        std::filesystem::temp_directory_path() / "ibeamlab-variable-spectrum-test";
    std::filesystem::remove_all(variableDirectory);
    datasets::DatasetMetadata variableMetadata;
    variableMetadata.requested = 2;
    variableMetadata.spectrumLabels = {"RBS"};
    {
        datasets::DatasetWriter writer(variableDirectory, variableMetadata);
        writer.append({0, "short", {}, {{{"RBS", {1.0F, 2.0F}}}, {}, std::nullopt}});
        writer.append(
            {1, "long", {}, {{{"RBS", {3.0F, 4.0F, 5.0F, 6.0F}}}, {}, std::nullopt}});
        writer.finalize();
    }
    datasets::DatasetReader variableReader(variableDirectory);
    assert(variableReader.metadata().spectrumLengths == std::vector<std::uint64_t>{4});
    const auto paddedRecords = variableReader.readAll();
    assert(paddedRecords.size() == 2);
    assert(paddedRecords[0].result.spectra[0].counts ==
           std::vector<float>({1.0F, 2.0F, 0.0F, 0.0F}));
    assert(paddedRecords[1].result.spectra[0].counts ==
           std::vector<float>({3.0F, 4.0F, 5.0F, 6.0F}));
    std::filesystem::remove_all(variableDirectory);

    const auto directory = std::filesystem::temp_directory_path() / "ibeamlab-native-test-dataset";
    std::filesystem::remove_all(directory);
    const auto package =
        model::ModelPackage::open(std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package");
    assert(throws([] { model::ModelPackage::open(
        std::filesystem::path(IBEAMLAB_TEST_DATA) / "invalid-package"); }));
    const auto zipped =
        model::ModelPackage::open(std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package.zip");
    assert(zipped.modelBytes() == package.modelBytes());
    inference::OnnxInferenceEngine zippedEngine(zipped);
    assert(std::abs(zippedEngine.predict({{{"RBS", {1.0F, 2.0F}}}})[0].values[0].value - 9.0F) <
           1e-5F);
    const auto predictions = zippedEngine.predict({{{"RBS", {4.0F, 5.0F}}}});
    assert(std::abs(predictions[0].values[0].value - 24.0F) < 1e-5F);
    assert(predictions[0].values[0].unit == "arb");
    assert(throws([&] { zippedEngine.predict({{{"RBS", {1.0F}}}}); }));
    auto dummy = std::make_shared<simulator::DummySimulator>(128);
    generation::DataGenerator generator(config, dummy);
    const auto summary =
        generator.generate(directory, {.samples = 4, .batchSize = 2, .shardCount = 2, .seed = 42});
    assert(summary.accepted == 4 && summary.failed == 0);
    datasets::DatasetReader reader(directory);
    assert(reader.metadata().complete && reader.readAll().size() == 4);
    assert(reader.metadata().generationConfigToml.find("species_concentration") ==
           std::string::npos);
    assert(reader.metadata().generationConfigToml.find("beam_energy") != std::string::npos);
    assert(reader.metadata().generationOptionsToml.find("batch_size") != std::string::npos);
    assert(reader.metadata().simulatorConfigToml.find("dummy") != std::string::npos);
    std::filesystem::resize_file(directory / reader.metadata().shardFiles[0], 9);
    assert(throws([&] { reader.readAll(); }));
    std::filesystem::remove_all(directory);
    const auto incomplete = std::filesystem::temp_directory_path() / "ibeamlab-incomplete-test";
    std::filesystem::remove_all(incomplete);
    {
        datasets::DatasetMetadata metadata;
        datasets::DatasetWriter writer(incomplete, metadata);
    }
    assert(!datasets::DatasetReader(incomplete).metadata().complete);
    std::filesystem::remove_all(incomplete);
    std::cout << "native tests passed\n";
}
