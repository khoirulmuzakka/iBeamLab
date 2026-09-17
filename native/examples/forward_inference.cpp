#include <ibeamlab/forward_model.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ibeamlab-forward MODEL_PACKAGE\n";
        return 2;
    }
    try {
        ibeamlab::inference::ForwardModel model(argv[1]);
        ibeamlab::simulator::SimulationInput input;
        // Applications populate input.sample and input.setup using their own
        // adapter before invoking the model.
        const auto results = model.predict({input});
        std::cout << results.front().spectra.size() << " spectrum/spectra\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
