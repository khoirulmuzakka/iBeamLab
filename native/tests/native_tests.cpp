#include <ibeamlab/datasets.h>
#include <ibeamlab/data_generator.h>
#include <ibeamlab/inference.h>
#include <ibeamlab/forward_model.h>
#include <ibeamlab/inverse_model.h>
#include <ibeamlab/spectrum_processing.h>
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
    const auto rebinned = spectrum::rebin({0, 1, 2}, {0, 2}, {2, 4});
    assert(rebinned.size() == 1 && std::abs(rebinned[0] - 6.0) < 1e-12);
    preprocessing::StandardScaler scaler({1, 2}, {2, 4});
    const auto scaled = scaler.apply({{3, 6}});
    assert(scaled[0][0] == 1 && scaled[0][1] == 1);
    assert(scaler.inverse(scaled)[0][0] == 3);
    preprocessing::LogTransform logarithm(2);
    const auto logged = logarithm.apply({{0, 3}});
    assert(std::abs(logarithm.inverse(logged)[0][1] - 3) < 1e-5);
    assert(throws([&] { logarithm.apply({{-2, 0}}); }));
    assert(spectrum::cropOrPad({1, 2, 3}, 2) == std::vector<double>({1, 2}));
    assert(spectrum::cropOrPad({1}, 3, -1) == std::vector<double>({1, -1, -1}));
    assert(spectrum::concatenate({{1, 2}, {3}}) == std::vector<double>({1, 2, 3}));
    assert(throws([] { spectrum::rebin({0, 1, 1}, {0, 1}, {1, 2}); }));

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
    const auto normalized = mixtureConfig.materialize({0.3, 0.5});
    assert(std::abs(normalized.sample.layers[0].species[0].concentration - 0.3) < 1e-12);
    assert(std::abs(normalized.sample.layers[0].species[1].concentration - 0.5) < 1e-12);
    assert(std::abs(normalized.sample.layers[0].species[2].concentration - 0.2) < 1e-12);

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
    assert(throws([] { ibeamlab::model::ModelPackage::open(
        std::filesystem::path(IBEAMLAB_TEST_DATA) / "invalid-package"); }));
    const auto writtenPackageDirectory =
        std::filesystem::temp_directory_path() / "ibeamlab-written-model-package";
    const auto writtenPackageZip =
        std::filesystem::temp_directory_path() / "ibeamlab-written-model-package.zip";
    std::filesystem::remove_all(writtenPackageDirectory);
    std::filesystem::remove(writtenPackageZip);
    ibeamlab::model::ModelMetadata inverseMetadata;
    inverseMetadata.modelType = ibeamlab::model::ModelType::Inverse;
    inverseMetadata.className = "Linear";
    inverseMetadata.inputDimension = 2;
    inverseMetadata.outputDimension = 1;
    inverseMetadata.opsetVersion = 18;
    inverseMetadata.inputTransform.inputDimension = 2;
    inverseMetadata.outputTransform.inputDimension = 1;
    inverseMetadata.inverse.sampleTemplate = model;
    inverseMetadata.inverse.setupTemplate = setup;
    inverseMetadata.inverse.inputSpectra = {{"RBS", 2}};
    inverseMetadata.inverse.outputParameters = {{"thickness", generation::LayerThickness{0},
                                                  0, 100, std::nullopt, "arb"}};
    auto inversePackage = ibeamlab::model::ModelPackage::fromOnnx(
        std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package" / "model.onnx",
        inverseMetadata);
    inversePackage.write(writtenPackageDirectory);
    inversePackage.write(writtenPackageZip);
    const auto inverseFromDirectory =
        ibeamlab::model::ModelPackage::open(writtenPackageDirectory);
    const auto inverseFromZip = ibeamlab::model::ModelPackage::open(writtenPackageZip);
    assert(inverseFromZip.modelBytes() == inverseFromDirectory.modelBytes());
    inference::InverseModel inverse(inverseFromZip);
    const auto inverseResults = inverse.predict({{{"RBS", {1.0F, 2.0F}}}});
    assert(std::abs(inverseResults[0].sample.layers[0].thickness - 9.0) < 1e-5);
    assert(inverseResults[0].parameters[0].name == "thickness");
    assert(throws([&] { inverse.predict({{{"RBS", {1.0F}}}}); }));
    std::filesystem::remove_all(writtenPackageDirectory);
    std::filesystem::remove(writtenPackageZip);

    ibeamlab::model::ModelMetadata forwardMetadata;
    forwardMetadata.modelType = ibeamlab::model::ModelType::Forward;
    forwardMetadata.className = "Linear";
    forwardMetadata.inputDimension = 2;
    forwardMetadata.outputDimension = 1;
    forwardMetadata.opsetVersion = 18;
    forwardMetadata.inputTransform.inputDimension = 2;
    forwardMetadata.outputTransform.inputDimension = 1;
    forwardMetadata.forward.sampleTemplate = model;
    forwardMetadata.forward.setupTemplate = setup;
    forwardMetadata.forward.inputParameters = {
        {"thickness", generation::LayerThickness{0}, 0, 100, std::nullopt, "arb"},
        {"energy", generation::BeamEnergy{"RBS"}, 0, 100, std::nullopt, "arb"}};
    forwardMetadata.forward.outputSpectra = {{"RBS", 1}};
    auto forwardPackage = ibeamlab::model::ModelPackage::fromOnnx(
        std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package" / "model.onnx",
        forwardMetadata);
    inference::ForwardModel forward(std::move(forwardPackage));
    auto forwardInput = simulator::SimulationInput{model, setup};
    forwardInput.sample.layers[0].thickness = 1.0;
    forwardInput.setup.detectors[0].beam.energy = 2.0;
    const auto forwardResults = forward.predict({forwardInput});
    assert(forwardResults[0].spectra[0].label == "RBS");
    assert(std::abs(forwardResults[0].spectra[0].counts[0] - 9.0F) < 1e-5F);
    auto dummy = std::make_shared<simulator::DummySimulator>(128);
    generation::DataGenerator generator(config, dummy);
    const std::vector<std::vector<double>> parameterRows{{91}, {97}, {103}, {109}};
    const auto summary =
        generator.generate(directory, parameterRows,
                           {.batchSize = 2,
                            .shardCount = 2,
                            .seed = 42,
                            .sampler = "native-test",
                            .samplerVersion = 1,
                            .samplingConfigToml = "strategy = \"explicit\"\n"});
    assert(summary.accepted == 4 && summary.failed == 0);
    datasets::DatasetReader reader(directory);
    assert(reader.metadata().complete && reader.readAll().size() == 4);
    assert(reader.metadata().generationConfigToml.find("species_concentration") ==
           std::string::npos);
    assert(reader.metadata().generationConfigToml.find("beam_energy") != std::string::npos);
    assert(reader.metadata().generationOptionsToml.find("batch_size") != std::string::npos);
    assert(reader.metadata().samplingConfigToml.find("explicit") != std::string::npos);
    assert(reader.metadata().provenance.sampler == "native-test");
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
