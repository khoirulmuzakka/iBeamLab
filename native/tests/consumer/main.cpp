#include <ibeamlab/forward_model.h>
#include <ibeamlab/inverse_model.h>
#include <ibeamlab/spectrum_processing.h>

#include <cmath>
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: ibeamlab_consumer MODEL_ONNX PACKAGE_DIRECTORY\n";
        return 2;
    }
    try {
        namespace fs = std::filesystem;
        using namespace ibeamlab;
        const auto measured = spectrum::pileup({10, 20}, 10, 8, 0.01);
        const auto recovered = spectrum::removePileup(measured, 10, 8, 0.01);
        if (recovered.size() != 2 || std::abs(recovered[1] - 20) > 1e-8)
            throw std::runtime_error("installed pileup removal returned unexpected counts");

        sample::SampleModel sample{
            {sample::Layer{1.0, 0.0, 0.0, 0.0, {sample::Species{"Si", 1.0, {}}}}}};
        sample::ExperimentalSetup setup{
            {sample::Detector{"RBS", sample::Beam{"He", 2.0, 0.0}, 1.0, 1.0}}};

        model::ModelMetadata metadata;
        metadata.modelType = model::ModelType::Forward;
        metadata.className = "InstalledConsumerProbe";
        metadata.inputDimension = 2;
        metadata.outputDimension = 1;
        metadata.opsetVersion = 18;
        metadata.inputTransform.inputDimension = 2;
        metadata.outputTransform.inputDimension = 1;
        metadata.forward.sampleTemplate = sample;
        metadata.forward.setupTemplate = setup;
        metadata.forward.inputParameters = {
            {"thickness", generation::LayerThickness{0}, 0, 100, std::nullopt, "arb"},
            {"energy", generation::BeamEnergy{"RBS"}, 0, 100, std::nullopt, "arb"},
        };
        metadata.forward.outputSpectra = {{"RBS", 1}};

        const fs::path packagePath = argv[2];
        std::error_code error;
        fs::remove_all(packagePath, error);
        auto package = model::ModelPackage::fromOnnx(argv[1], std::move(metadata));
        package.write(packagePath);

        inference::ForwardModel forward(packagePath);
        simulator::SimulationInput input{sample, setup};
        const auto result = forward.predict({input});
        const bool valid = forward.metadata().className == "InstalledConsumerProbe" &&
                           result.size() == 1 && result[0].spectra.size() == 1 &&
                           std::abs(result[0].spectra[0].counts[0] - 9.0F) < 1e-5F;
        fs::remove_all(packagePath, error);
        if (!valid) {
            std::cerr << "installed consumer inference returned an unexpected result\n";
            return 1;
        }
        model::ModelMetadata inverseMetadata;
        inverseMetadata.modelType = model::ModelType::Inverse;
        inverseMetadata.inputDimension = 2;
        inverseMetadata.outputDimension = 1;
        inverseMetadata.inputTransform.inputDimension = 2;
        inverseMetadata.outputTransform.inputDimension = 1;
        inverseMetadata.inverse.sampleTemplate = sample;
        inverseMetadata.inverse.setupTemplate = setup;
        inverseMetadata.inverse.inputSpectra = {{"RBS", 2}};
        inverseMetadata.inverse.outputEdp = {1, {"Si"}};
        inverseMetadata.inverse.needPileupSubtraction = true;
        auto inversePackage = model::ModelPackage::fromOnnx(argv[1], inverseMetadata);
        inversePackage.write(packagePath);
        inference::InverseModel inverse(packagePath);
        if (!inverse.metadata().inverse.needPileupSubtraction)
            throw std::runtime_error("installed inverse consumer lost pileup preparation flag");
        inference::InverseInput measuredInput;
        measuredInput.setup = setup;
        measuredInput.setup.detectors[0].particlesSr *= 2;
        measuredInput.setup.detectors[0].calibrationLinear = 0.5;
        measuredInput.setup.detectors[0].realTime = 10;
        measuredInput.setup.detectors[0].liveTime = 8;
        const auto measuredCounts = spectrum::pileup(
            {1, 1, 2, 2}, 10, 8, inverse.metadata().inverse.pileupFudgeFactorSeconds);
        measuredInput.spectra = {
            {"RBS", std::vector<float>(measuredCounts.begin(), measuredCounts.end())}};
        const auto corrected = inverse.predict({measuredInput})[0].edp;
        if (std::abs(corrected.values[0][0] - 9) > 1e-5)
            throw std::runtime_error(
                "installed inverse correction pipeline returned unexpected EDP");
        const auto profile = inverse.predictPrepared({{{"RBS", {1, 2}}}})[0].edp;
        const auto reconstructed = profile.toSample(sample);
        fs::remove_all(packagePath, error);
        if (profile.values.size() != 1 || profile.elements[0] != "Si" ||
            std::abs(reconstructed.layers[0].thickness - 9.0) > 1e-5)
            throw std::runtime_error("installed inverse consumer returned unexpected EDP");
        std::cout << "installed consumer inference passed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
