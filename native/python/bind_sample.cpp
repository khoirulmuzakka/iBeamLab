#include "bindings.h"
#include <ibeamlab/sample_toml.h>
#include <ibeamlab/sample.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace ibeamlab::sample;
void bindSample(py::module_ &root) {
    auto m = root.def_submodule("sample");
    m.def("sample_to_toml", py::overload_cast<const SampleModel &>(&toToml));
    m.def("sample_from_toml", &sampleModelFromToml);
    m.def("setup_to_toml", py::overload_cast<const ExperimentalSetup &>(&toToml));
    m.def("setup_from_toml", &experimentalSetupFromToml);
    py::class_<Isotope>(m, "Isotope")
        .def(py::init<>())
        .def_readwrite("mass_number", &Isotope::massNumber)
        .def_readwrite("exact_mass", &Isotope::exactMass)
        .def_readwrite("fraction", &Isotope::fraction);
    py::class_<Species>(m, "Species")
        .def(py::init<>())
        .def_readwrite("element", &Species::element)
        .def_readwrite("concentration", &Species::concentration)
        .def_readwrite("isotopes", &Species::isotopes);
    py::class_<Layer>(m, "Layer")
        .def(py::init<>())
        .def_readwrite("thickness", &Layer::thickness)
        .def_readwrite("roughness", &Layer::roughness)
        .def_readwrite("porosity_fraction", &Layer::porosityFraction)
        .def_readwrite("pore_diameter", &Layer::poreDiameter)
        .def_readwrite("species", &Layer::species);
    py::class_<SampleModel>(m, "SampleModel")
        .def(py::init<>())
        .def_readwrite("layers", &SampleModel::layers)
        .def("validate", &SampleModel::validate);
    py::class_<Beam>(m, "Beam")
        .def(py::init<>())
        .def_readwrite("particle", &Beam::particle)
        .def_readwrite("energy", &Beam::energy)
        .def_readwrite("spread", &Beam::spread);
    py::class_<Detector>(m, "Detector")
        .def(py::init<>())
        .def_readwrite("label", &Detector::label)
        .def_readwrite("beam", &Detector::beam)
        .def_readwrite("calibration_linear", &Detector::calibrationLinear)
        .def_readwrite("particles_sr", &Detector::particlesSr)
        .def_readwrite("calibration_offset", &Detector::calibrationOffset)
        .def_readwrite("calibration_quadratic", &Detector::calibrationQuadratic)
        .def_readwrite("resolution", &Detector::resolution)
        .def_readwrite("real_time", &Detector::realTime)
        .def_readwrite("live_time", &Detector::liveTime)
        .def_readwrite("info", &Detector::info);
    py::class_<ExperimentalSetup>(m, "ExperimentalSetup")
        .def(py::init<>())
        .def_readwrite("detectors", &ExperimentalSetup::detectors)
        .def("validate", &ExperimentalSetup::validate);
}
