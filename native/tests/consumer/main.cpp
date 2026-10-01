#include <ibeamlab/forward_model.h>

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
        std::cout << "installed consumer inference passed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
