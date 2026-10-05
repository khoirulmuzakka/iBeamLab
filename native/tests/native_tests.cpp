#include <ibeamlab/data_generator.h>
#include <ibeamlab/datasets.h>
#include <ibeamlab/forward_model.h>
#include <ibeamlab/inference.h>
#include <ibeamlab/inverse_model.h>
#include <ibeamlab/sample.h>
#include <ibeamlab/sample_toml.h>
#include <ibeamlab/simulator.h>
#include <ibeamlab/spectrum_processing.h>
#include <ibeamlab/transforms.h>

#ifdef NDEBUG
#undef NDEBUG // Keep numerical and package checks active in Release builds.
#endif
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
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
    // AutoNRA formula uses seconds after converting its 0.4 microsecond setting.
    const double tau = 0.4e-6, real = 10, live = 8, n = 6;
    const double a = tau / real * std::exp(-n / real * tau);
    const auto pu = spectrum::pileup({2, 4}, real, live, tau);
    assert(pu.size() == 3);
    assert(std::abs(pu[0] - live / real * (2 - 2 * a * n * 2 + a * 4)) < 1e-12);
    assert(std::abs(pu[1] - live / real * (4 - 2 * a * n * 4 + a * 16)) < 1e-12);
    assert(std::abs(pu[2] - live / real * a * 16) < 1e-12);
    assert(throws([] { spectrum::pileup({1}, 0, 1, 0.4e-6); }));
    // Invert complete pileup spectra across support sizes and pileup strengths.
    for (std::size_t size : {1U, 2U, 7U, 64U, 257U}) {
        std::vector<double> original(size);
        for (std::size_t i = 0; i < size; ++i)
            original[i] = i % 5 == 0 && size > 1 ? 0 : 1 + (i * 17) % 101;
        double count = 0;
        for (double value : original)
            count += value;
        for (double rate : {0.0, 1e-9, 0.01, 0.5, 1.0, 2.0, 10.0, 1000.0}) {
            const double factor = rate * real / count;
            const auto measured = spectrum::pileup(original, real, live, factor);
            const auto recovered = spectrum::removePileup(measured, real, live, factor);
            assert(recovered.size() == size);
            for (std::size_t i = 0; i < size; ++i)
                assert(std::abs(recovered[i] - original[i]) < 1e-8);
            // Counts are stored as float32 at inference boundaries.
            auto rounded = measured;
            for (auto &value : rounded)
                value = static_cast<float>(value);
            const auto floatRecovered = spectrum::removePileup(rounded, real, live, factor);
            for (std::size_t i = 0; i < size; ++i)
                assert(std::abs(floatRecovered[i] - original[i]) < 1e-3);
        }
    }
    assert(spectrum::removePileup({}, real, live, tau).empty());
    assert(spectrum::removePileup({0, 0, 0}, real, live, tau) == std::vector<double>({0, 0}));
    assert(spectrum::removePileup({8, 16, 0}, 10, 8, 0) == std::vector<double>({10, 20}));
    assert(throws([&] { spectrum::removePileup(pu, 0, live, tau); }));
    assert(throws([&] { spectrum::removePileup(pu, real, 0, tau); }));
    assert(throws([&] { spectrum::removePileup(pu, real, live, -1); }));
    assert(throws([&] { spectrum::removePileup(pu, real, live, tau, 0); }));
    assert(throws([&] { spectrum::removePileup(pu, real, live, tau, 1); }));
    assert(throws([&] { spectrum::removePileup({1, 2}, real, live, tau); }));
    assert(throws([&] { spectrum::removePileup({-1}, real, live, tau); }));
    assert(throws([&] { spectrum::removePileup({NAN}, real, live, tau); }));
    assert(throws([&] { spectrum::removePileup({INFINITY}, real, live, tau); }));
    assert(throws([&] {
        const auto maximum = std::numeric_limits<double>::max();
        spectrum::removePileup({maximum, maximum, maximum}, real, live, tau);
    }));
    assert(throws([&] { spectrum::removePileup({1}, INFINITY, live, tau); }));
    assert(throws([&] { spectrum::removePileup({1}, real, live, NAN); }));
    assert(throws([&] { spectrum::removePileup({1}, real, live, tau, NAN); }));
    assert(throws([&] { spectrum::removePileup({1, 2, 3}, real, live, 0); }));
    // Cropping to another odd size must fail the model/tail check.
    auto truncated = spectrum::pileup({10, 20, 30, 40}, real, live, 0.01);
    truncated.resize(5);
    assert(throws([&] { spectrum::removePileup(truncated, real, live, 0.01); }));
    preprocessing::StandardScaler scaler({1, 2}, {2, 4});
    const auto scaled = scaler.apply({{3, 6}});
    assert(scaled[0][0] == 1 && scaled[0][1] == 1);
    assert(scaler.inverse(scaled)[0][0] == 3);
    preprocessing::LayerwiseConcentrationNormalizer concentrationNormalizer(6, {{1, 2}, {4, 5}});
    const auto normalizedConcentrations =
        concentrationNormalizer.apply({{100, 2, 6, 50000, -1, 3}});
    assert(std::abs(normalizedConcentrations[0][1] - 0.25F) < 1e-6F);
    assert(std::abs(normalizedConcentrations[0][2] - 0.75F) < 1e-6F);
    assert(normalizedConcentrations[0][4] == 0.0F);
    assert(normalizedConcentrations[0][5] == 1.0F);
    preprocessing::ParameterBoundMinMaxScaler parameterScaler({0, 0, 0, 0, 0, 0},
                                                              {1000, 1, 1, 100000, 1, 1});
    const auto parameterScaled = parameterScaler.apply(normalizedConcentrations);
    assert(std::abs(parameterScaled[0][0] - 0.1F) < 1e-6F);
    assert(std::abs(parameterScaled[0][3] - 0.5F) < 1e-6F);
    assert(std::abs(parameterScaled[0][1] - 0.25F) < 1e-6F);
    preprocessing::TransformPipeline inputPipeline;
    inputPipeline.add(
        std::make_shared<preprocessing::LayerwiseConcentrationNormalizer>(concentrationNormalizer));
    inputPipeline.add(std::make_shared<preprocessing::ParameterBoundMinMaxScaler>(parameterScaler));
    assert(std::abs(inputPipeline.apply({{100, 2, 6, 50000, -1, 3}})[0][5] - 1.0F) < 1e-6F);
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

    sample::SampleModel mixture{
        {sample::Layer{1.0,
                       0,
                       0,
                       0,
                       {sample::Species{"C", 0.4, {}}, sample::Species{"O", 0.4, {}},
                        sample::Species{"Zr", 0.2, {}}}}}};
    generation::GenerationConfig mixtureConfig{
        mixture,
        setup,
        {generation::MethodConfig{"RBS", "RBS", "reference.xnra"}},
        {
            generation::ParameterSpec{"C", generation::SpeciesConcentration{0, "C"}, 0, 0.8,
                                      std::nullopt, "fraction"},
            generation::ParameterSpec{"Zr", generation::SpeciesConcentration{0, "Zr"}, 0, 1, 0.2,
                                      "fraction"},
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
        writer.append({1, "long", {}, {{{"RBS", {3.0F, 4.0F, 5.0F, 6.0F}}}, {}, std::nullopt}});
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
    assert(throws([] {
        ibeamlab::model::ModelPackage::open(std::filesystem::path(IBEAMLAB_TEST_DATA) /
                                            "invalid-package");
    }));
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
    inverseMetadata.inverse.outputEdp = {1, {"Si"}};
    assert(!inverseMetadata.inverse.needPileupSubtraction);
    inverseMetadata.inverse.needPileupSubtraction = true;
    auto invalidTransformMetadata = inverseMetadata;
    invalidTransformMetadata.inputTransform.type = "min_max_scaler";
    invalidTransformMetadata.inputTransform.minimum = {0};
    invalidTransformMetadata.inputTransform.scale = {1};
    assert(throws([&] {
        model::ModelPackage::fromOnnx(std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package" /
                                          "model.onnx",
                                      invalidTransformMetadata);
    }));
    auto inversePackage = ibeamlab::model::ModelPackage::fromOnnx(
        std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package" / "model.onnx",
        inverseMetadata);
    inversePackage.write(writtenPackageDirectory);
    inversePackage.write(writtenPackageZip);
    const auto inverseFromDirectory = ibeamlab::model::ModelPackage::open(writtenPackageDirectory);
    const auto inverseFromZip = ibeamlab::model::ModelPackage::open(writtenPackageZip);
    assert(inverseFromDirectory.metadata().inverse.needPileupSubtraction);
    assert(inverseFromZip.metadata().inverse.needPileupSubtraction);
    // Older version 3 manifests omit this optional field; retain their behavior.
    const auto manifestPath = writtenPackageDirectory / "package.toml";
    std::ifstream manifestInput(manifestPath);
    std::ostringstream manifestBuffer;
    manifestBuffer << manifestInput.rdbuf();
    manifestInput.close();
    const auto originalManifest = manifestBuffer.str();
    const std::string flag = "need_pileup_subtraction = true";
    const auto flagOffset = originalManifest.find(flag);
    assert(flagOffset != std::string::npos);
    auto replaceFlag = [&](const std::string &replacement) {
        auto text = originalManifest;
        text.replace(flagOffset, flag.size(), replacement);
        std::ofstream output(manifestPath);
        output << text;
    };
    replaceFlag("");
    assert(!model::ModelPackage::open(writtenPackageDirectory)
                .metadata()
                .inverse.needPileupSubtraction);
    replaceFlag("need_pileup_subtraction = false");
    assert(!model::ModelPackage::open(writtenPackageDirectory)
                .metadata()
                .inverse.needPileupSubtraction);
    replaceFlag("need_pileup_subtraction = \"true\"");
    assert(throws([&] { model::ModelPackage::open(writtenPackageDirectory); }));
    replaceFlag(flag);
    assert(inverseFromZip.modelBytes() == inverseFromDirectory.modelBytes());
    inference::InverseModel inverse(inverseFromZip);
    assert(inverse.metadata().inverse.needPileupSubtraction);
    const auto inverseResults = inverse.predictPrepared({{{"RBS", {1.0F, 2.0F}}}});
    assert(std::abs(inverseResults[0].edp.values[0][0] - 9.0) < 1e-5);
    assert(inverseResults[0].edp.elements == std::vector<std::string>{"Si"});
    assert(std::abs(inverseResults[0].edp.toSample(model).layers[0].thickness - 9.0) < 1e-5);
    assert(throws([&] { inverse.predictPrepared({{{"RBS", {1.0F}}}}); }));
    std::filesystem::remove_all(writtenPackageDirectory);
    std::filesystem::remove(writtenPackageZip);

    // Multi-detector EDP: two active layers, two elements, one padded layer.
    auto edpMetadata = inverseMetadata;
    edpMetadata.inputDimension = 4;
    edpMetadata.outputDimension = 6;
    edpMetadata.inputTransform = {};
    edpMetadata.inputTransform.type = "constant_factor";
    edpMetadata.inputTransform.inputDimension = 4;
    edpMetadata.inputTransform.factor = 2;
    edpMetadata.outputTransform = {};
    edpMetadata.outputTransform.type = "constant_factor";
    edpMetadata.outputTransform.inputDimension = 6;
    edpMetadata.outputTransform.factor = 4;
    auto edpTemplate = model;
    edpTemplate.layers[0].species = {{"O", 0.5, {}}, {"Si", 0.5, {}}};
    edpTemplate.layers[0].roughness = 0.25;
    edpTemplate.layers.resize(3, edpTemplate.layers[0]);
    edpMetadata.inverse.sampleTemplate = edpTemplate;
    auto secondDetector = setup.detectors[0];
    secondDetector.label = "PIXE";
    edpMetadata.inverse.setupTemplate.detectors.push_back(secondDetector);
    edpMetadata.inverse.inputSpectra = {{"RBS", 2}, {"PIXE", 2}};
    edpMetadata.inverse.outputEdp = {3, {"Si", "O"}};
    edpMetadata.inverse.pileupFudgeFactorSeconds = 0.003;
    const auto edpFile = std::filesystem::path(IBEAMLAB_TEST_DATA) / "edp.onnx";
    auto edpPackage = model::ModelPackage::fromOnnx(edpFile, edpMetadata);
    edpPackage.write(writtenPackageDirectory);
    edpPackage.write(writtenPackageZip);
    const auto edpDirectory = model::ModelPackage::open(writtenPackageDirectory);
    const auto edpZip = model::ModelPackage::open(writtenPackageZip);
    assert(edpZip.metadata().formatVersion == 3);
    assert(edpZip.metadata().inverse.pileupFudgeFactorSeconds == 0.003);
    assert(edpZip.metadata().inverse.outputEdp.maxLayers == 3);
    assert(edpZip.metadata().inverse.outputEdp.elements == edpMetadata.inverse.outputEdp.elements);
    assert(edpZip.metadata().inverse.outputEdp.unit == "1e15 atoms/cm2");
    inference::InverseModel edpModel(edpZip);
    const auto profiles = edpModel.predictPrepared(
        {{{"PIXE", {6, 8}}, {"RBS", {2, 4}}}, {{"RBS", {4, 8}}, {"PIXE", {12, 16}}}});
    const std::vector<std::vector<float>> expectedEdp{{1, 2}, {3, 4}, {0, 0}};
    assert(profiles.size() == 2 && profiles[0].edp.values == expectedEdp);
    assert(profiles[1].edp.values[1][1] == 8);
    inference::InverseModel edpFromDirectory(edpDirectory);
    assert(edpFromDirectory.predictPrepared({{{"RBS", {2, 4}}, {"PIXE", {6, 8}}}})[0].edp.values ==
           expectedEdp);
    // Full inverse pipeline: experimental exposure/calibration/pileup -> reference
    // counts -> packaged input scaling -> ONNX -> physical output scaling.
    inference::InverseInput measurement;
    measurement.setup = edpMetadata.inverse.setupTemplate;
    std::vector<simulator::Spectrum> bareExperimental;
    for (auto &detector : measurement.setup.detectors) {
        detector.calibrationLinear = 0.5;
        const double exposure = detector.label == "RBS" ? 2 : 3;
        detector.particlesSr *= exposure;
        detector.realTime = 10;
        detector.liveTime = 8;
        const std::vector<double> referenceCounts =
            detector.label == "RBS" ? std::vector<double>{2, 4} : std::vector<double>{6, 8};
        const auto fine = spectrum::rebin({0, 1, 2}, {0, 0.5, 1, 1.5, 2}, referenceCounts);
        auto exposed = fine;
        for (auto &value : exposed)
            value *= exposure;
        bareExperimental.push_back(
            {detector.label, std::vector<float>(exposed.begin(), exposed.end())});
        const auto piled = spectrum::pileup(exposed, detector.realTime, detector.liveTime,
                                            edpMetadata.inverse.pileupFudgeFactorSeconds);
        measurement.spectra.insert(
            measurement.spectra.begin(),
            {detector.label, std::vector<float>(piled.begin(), piled.end())});
    }
    auto alreadyRemoved = measurement;
    alreadyRemoved.spectra = bareExperimental;
    alreadyRemoved.pileupAlreadyRemoved = true;
    for (auto &detector : alreadyRemoved.setup.detectors)
        detector.realTime = detector.liveTime = 0; // No timing needed when skipped.
    const auto correctedProfiles = edpModel.predict({measurement, alreadyRemoved});
    assert(correctedProfiles.size() == 2);
    for (const auto &profile : correctedProfiles)
        for (std::size_t l = 0; l < expectedEdp.size(); ++l)
            for (std::size_t e = 0; e < expectedEdp[l].size(); ++e)
                assert(std::abs(profile.edp.values[l][e] - expectedEdp[l][e]) < 1e-5);
    assert(edpModel.predict({}).empty());
    auto badMeasurement = measurement;
    badMeasurement.setup.detectors[0].particlesSr = 0;
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    badMeasurement = measurement;
    badMeasurement.setup.detectors[0].liveTime = 0;
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    badMeasurement = measurement;
    badMeasurement.setup.detectors.erase(badMeasurement.setup.detectors.begin());
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    badMeasurement = measurement;
    badMeasurement.spectra.push_back(badMeasurement.spectra[0]);
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    badMeasurement = alreadyRemoved;
    badMeasurement.spectra[0].counts = {1}; // Insufficient reference coverage.
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    badMeasurement = alreadyRemoved;
    badMeasurement.setup.detectors[0].calibrationLinear = -1;
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    badMeasurement = measurement;
    badMeasurement.spectra[0].counts.pop_back(); // Cropped/even pileup support.
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    badMeasurement = alreadyRemoved;
    badMeasurement.spectra[0].counts[0] = NAN;
    assert(throws([&] { edpModel.predict({badMeasurement}); }));
    // Quadratic calibration and offset on a constant spectral density.
    auto nonlinear = alreadyRemoved;
    for (auto &detector : nonlinear.setup.detectors) {
        detector.calibrationOffset = -0.25;
        detector.calibrationLinear = 0.75;
        detector.calibrationQuadratic = 0.25;
    }
    nonlinear.spectra = {{"RBS", {4, 6}}, {"PIXE", {12, 18}}};
    const auto nonlinearResult = edpModel.predict({nonlinear})[0].edp;
    assert(std::abs(nonlinearResult.values[0][0] - 1) < 1e-5);
    assert(std::abs(nonlinearResult.values[0][1] - 1) < 1e-5);
    assert(std::abs(nonlinearResult.values[1][0] - 2) < 1e-5);
    assert(std::abs(nonlinearResult.values[1][1] - 2) < 1e-5);
    auto mismatched = alreadyRemoved;
    mismatched.setup.detectors[0].beam.energy += 0.1;
    std::ostringstream inverseWarnings;
    auto *oldInverseWarnings = std::cerr.rdbuf(inverseWarnings.rdbuf());
    edpModel.predict({mismatched});
    const auto firstInverseWarnings = inverseWarnings.str();
    edpModel.predict({mismatched});
    std::cerr.rdbuf(oldInverseWarnings);
    assert(firstInverseWarnings.find("no correction is available") != std::string::npos);
    assert(inverseWarnings.str() == firstInverseWarnings);
    // Packages not requiring subtraction skip it even with raw input flag false.
    auto noSubtractionMetadata = edpMetadata;
    noSubtractionMetadata.inverse.needPileupSubtraction = false;
    inference::InverseModel noSubtraction(
        model::ModelPackage::fromOnnx(edpFile, noSubtractionMetadata));
    auto unpiled = alreadyRemoved;
    unpiled.pileupAlreadyRemoved = false;
    assert(noSubtraction.predict({unpiled})[0].edp.values == expectedEdp);
    auto invalidFudge = edpMetadata;
    invalidFudge.inverse.pileupFudgeFactorSeconds = -1;
    assert(throws([&] { model::ModelPackage::fromOnnx(edpFile, invalidFudge); }));
    invalidFudge.inverse.pileupFudgeFactorSeconds = NAN;
    assert(throws([&] { model::ModelPackage::fromOnnx(edpFile, invalidFudge); }));
    const auto reconstructed = profiles[0].edp.toSample(edpTemplate);
    assert(reconstructed.layers.size() == 2);
    assert(reconstructed.layers[0].thickness == 3);
    assert(reconstructed.layers[1].thickness == 7);
    assert(reconstructed.layers[0].species[0].element == "O");
    assert(std::abs(reconstructed.layers[0].species[0].concentration - 2.0 / 3) < 1e-12);
    assert(reconstructed.layers[0].roughness == 0.25);
    assert(edpModel.predictPrepared({}).empty());
    assert(throws([&] { edpModel.predictPrepared({{{"RBS", {2, 4}}}}); }));
    assert(throws([&] { edpModel.predictPrepared({{{"RBS", {2}}, {"PIXE", {6, 8}}}}); }));
    assert(throws([&] { edpModel.predictPrepared({{{"RBS", {2, 4}}, {"RBS", {6, 8}}}}); }));
    assert(throws([&] { edpModel.predictPrepared({{{"RBS", {2, 4}}, {"extra", {6, 8}}}}); }));
    assert(throws([&] { edpModel.predictPrepared({{{"RBS", {-2, 4}}, {"PIXE", {6, 8}}}}); }));
    assert(throws([&] { edpModel.predictPrepared({{{"RBS", {0, 0}}, {"PIXE", {6, 8}}}}); }));
    assert(throws([&] { edpModel.predictPrepared({{{"RBS", {NAN, 4}}, {"PIXE", {6, 8}}}}); }));
    const auto emptyProfile =
        edpModel.predictPrepared({{{"RBS", {0, 0}}, {"PIXE", {0, 0}}}})[0].edp;
    assert(throws([&] { emptyProfile.toSample(edpTemplate); }));
    for (int invalid = 0; invalid < 7; ++invalid) {
        auto bad = edpMetadata;
        switch (invalid) {
        case 0:
            bad.inverse.outputEdp.maxLayers = 0;
            break;
        case 1:
            bad.inverse.outputEdp.elements = {"Si", "Si"};
            break;
        case 2:
            bad.inverse.outputEdp.unit = "atoms/cm2";
            break;
        case 3:
            bad.outputDimension = 5;
            bad.outputTransform.inputDimension = 5;
            break;
        case 4:
            bad.inverse.inputSpectra[1].label = "RBS";
            break;
        case 5:
            bad.inverse.inputSpectra[1].label = "unknown";
            break;
        case 6:
            bad.inverse.outputEdp.elements = {"Si", "C"};
            break;
        }
        assert(throws([&] { model::ModelPackage::fromOnnx(edpFile, bad); }));
    }
    assert(throws([&] { inference::ForwardModel wrong(edpZip); }));
    std::filesystem::remove_all(writtenPackageDirectory);
    std::filesystem::remove(writtenPackageZip);

    ibeamlab::model::ModelMetadata forwardMetadata;
    forwardMetadata.modelType = ibeamlab::model::ModelType::Forward;
    forwardMetadata.className = "Linear";
    forwardMetadata.inputDimension = 2;
    forwardMetadata.outputDimension = 1;
    forwardMetadata.opsetVersion = 18;
    forwardMetadata.inputTransform.type = "pipeline";
    forwardMetadata.inputTransform.inputDimension = 2;
    model::TransformSpec concentrationTransform;
    concentrationTransform.type = "layerwise_concentration_normalizer";
    concentrationTransform.inputDimension = 2;
    concentrationTransform.concentrationGroups = {{0, 1}};
    model::TransformSpec parameterTransform;
    parameterTransform.type = "parameter_bound_min_max_scaler";
    parameterTransform.inputDimension = 2;
    parameterTransform.minimum = {0, 0};
    parameterTransform.scale = {1, 1};
    forwardMetadata.inputTransform.transforms = {concentrationTransform, parameterTransform};
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
    const auto forwardPackageDirectory =
        std::filesystem::temp_directory_path() / "ibeamlab-forward-transform-package";
    std::filesystem::remove_all(forwardPackageDirectory);
    forwardPackage.write(forwardPackageDirectory);
    const auto forwardRoundTrip = model::ModelPackage::open(forwardPackageDirectory);
    const std::vector<std::vector<std::size_t>> expectedConcentrationGroups{{0, 1}};
    assert(forwardRoundTrip.metadata().inputTransform.transforms[0].concentrationGroups ==
           expectedConcentrationGroups);
    inference::ForwardModel forward(forwardRoundTrip);
    assert(forward.metadata().className == "Linear");
    auto forwardInput = simulator::SimulationInput{model, setup};
    forwardInput.sample.layers[0].thickness = 1.0;
    forwardInput.setup.detectors[0].beam.energy = 2.0;
    const auto forwardResults = forward.predict({forwardInput});
    assert(forwardResults[0].spectra[0].label == "RBS");
    assert(std::abs(forwardResults[0].spectra[0].counts[0] - (11.0F / 3.0F)) < 1e-5F);
    auto correctedMetadata = forwardMetadata;
    correctedMetadata.inputTransform.type = "identity";
    correctedMetadata.inputTransform.transforms.clear();
    correctedMetadata.forward.inputParameters[1] = {
        "Si", generation::SpeciesConcentration{0, "Si"}, 0, 1, std::nullopt, "fraction"};
    correctedMetadata.forward.bareSpectrumCorrections = true;
    auto correctedPackage = model::ModelPackage::fromOnnx(
        std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package" / "model.onnx",
        correctedMetadata);
    std::filesystem::remove_all(forwardPackageDirectory);
    correctedPackage.write(forwardPackageDirectory);
    auto correctedRoundTrip = model::ModelPackage::open(forwardPackageDirectory);
    assert(correctedRoundTrip.metadata().forward.bareSpectrumCorrections);
    assert(correctedRoundTrip.metadata().forward.applyPileupOnInference);
    assert(correctedRoundTrip.metadata().forward.pileupFudgeFactorSeconds == 0.4e-6);
    inference::InferenceOptions noPileup;
    noPileup.applyPileupOnInference = false;
    inference::ForwardModel corrected(correctedRoundTrip, noPileup);
    auto requested = simulator::SimulationInput{model, setup};
    const auto bare = corrected.predict({requested})[0].spectra[0].counts;
    assert(bare.size() == 1);
    requested.setup.detectors[0].particlesSr *= 2;
    const auto doubled = corrected.predict({requested})[0].spectra[0].counts;
    assert(std::abs(doubled[0] - 2 * bare[0]) < 1e-5);
    requested.setup.detectors[0].calibrationLinear = 0.5;
    const auto split = corrected.predict({requested})[0].spectra[0].counts;
    assert(split.size() == 2 && std::abs(split[0] - bare[0]) < 1e-5 &&
           std::abs(split[1] - bare[0]) < 1e-5);
    requested.setup.detectors[0].calibrationOffset = 0.25;
    requested.setup.detectors[0].calibrationQuadratic = 0.25;
    const auto quadratic = corrected.predict({requested})[0].spectra[0].counts;
    assert(quadratic.size() == 1 && std::abs(quadratic[0] - 1.5 * bare[0]) < 1e-5);
    requested.setup.detectors[0].realTime = 10;
    requested.setup.detectors[0].liveTime = 8;
    inference::ForwardModel withPileup(correctedRoundTrip);
    const auto expectedPileup =
        spectrum::pileup(std::vector<double>(quadratic.begin(), quadratic.end()), 10, 8, 0.4e-6);
    const auto piled = withPileup.predict({requested})[0].spectra[0].counts;
    assert(piled.size() == expectedPileup.size());
    assert(std::abs(piled[0] - expectedPileup[0]) < 1e-5);
    inference::InferenceOptions parallelOptions;
    parallelOptions.correctionThreads = 4;
    inference::ForwardModel parallel(correctedRoundTrip, parallelOptions);
    std::vector<simulator::SimulationInput> candidates(32, requested);
    for (std::size_t c = 0; c < candidates.size(); ++c) {
        candidates[c].setup.detectors[0].particlesSr *= (c + 1);
        candidates[c].setup.detectors[0].calibrationLinear += c * 0.01;
    }
    const auto serialBatch = withPileup.predict(candidates);
    const auto parallelBatch = parallel.predict(candidates);
    for (std::size_t c = 0; c < candidates.size(); ++c)
        assert(serialBatch[c].spectra[0].counts == parallelBatch[c].spectra[0].counts);
    candidates[12].setup.detectors[0].realTime = 0;
    assert(throws([&] { parallel.predict(candidates); }));
    assert(parallel.predict({}).empty());
    parallelOptions.correctionThreads = 0;
    assert(throws([&] { inference::ForwardModel invalid(correctedRoundTrip, parallelOptions); }));
    requested.setup.detectors[0].realTime = 0;
    assert(throws([&] { withPileup.predict({requested}); }));
    requested.sample.layers[0].species = {{"C", 1, {}}};
    std::ostringstream warnings;
    auto *oldWarningBuffer = std::cerr.rdbuf(warnings.rdbuf());
    requested.setup.detectors[0].beam.energy = 200;
    assert(!corrected.predict({requested}).empty()); // Missing Si becomes zero.
    const auto firstWarnings = warnings.str();
    assert(firstWarnings.find("unsupported element C") != std::string::npos);
    assert(firstWarnings.find("beam or detector resolution") != std::string::npos);
    corrected.predict({requested});
    assert(warnings.str() == firstWarnings);
    std::cerr.rdbuf(oldWarningBuffer);
    auto invalidBare = correctedMetadata;
    invalidBare.forward.inputParameters[1] = forwardMetadata.forward.inputParameters[1];
    assert(throws([&] {
        model::ModelPackage::fromOnnx(std::filesystem::path(IBEAMLAB_TEST_DATA) / "model-package" /
                                          "model.onnx",
                                      invalidBare);
    }));
    std::filesystem::remove_all(forwardPackageDirectory);
    auto dummy = std::make_shared<simulator::DummySimulator>(128);
    generation::DataGenerator generator(config, dummy);
    const std::vector<std::vector<double>> parameterRows{{91}, {97}, {103}, {109}};
    const auto summary = generator.generate(directory, parameterRows,
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
