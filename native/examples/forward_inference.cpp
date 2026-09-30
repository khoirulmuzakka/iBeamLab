#include <ibeamlab/forward_model.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ibeamlab-forward MODEL_PACKAGE\n";
        return 2;
    }
    try {
        ibeamlab::inference::ForwardModel model(argv[1]);
        const auto& metadata = model.metadata();
        ibeamlab::simulator::SimulationInput input{
            metadata.forward.sampleTemplate,
            metadata.forward.setupTemplate,
        };
        // Applications may replace values targeted by metadata.forward.inputParameters.
        const auto results = model.predict({input});
        std::cout << results.front().spectra.size() << " spectrum/spectra\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
