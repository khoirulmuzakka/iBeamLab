#include "bindings.h"
PYBIND11_MODULE(_ibeamlab_cpp, m) {
    m.doc() = "Native iBeamLab simulation and ML pipeline";
    bindSample(m);
    bindSimulator(m);
    bindGeneration(m);
    bindDatasets(m);
    bindPreprocessing(m);
    bindModel(m);
    bindInference(m);
}
