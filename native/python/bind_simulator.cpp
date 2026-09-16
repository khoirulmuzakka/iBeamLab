#include "bindings.h"
#include <ibeamlab/simnra_simulator.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py = pybind11;
using namespace ibeamlab::simulator;
namespace {
py::array_t<float> counts(const Spectrum &value) {
    py::array_t<float> out(value.counts.size());
    auto view = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < value.counts.size(); ++i)
        view(static_cast<py::ssize_t>(i)) = value.counts[i];
    return out;
}
void setCounts(Spectrum &value,
               const py::array_t<float, py::array::c_style | py::array::forcecast> &input) {
    if (input.ndim() != 1)
        throw py::value_error("spectrum counts must be one-dimensional");
    auto view = input.unchecked<1>();
    value.counts.resize(static_cast<std::size_t>(input.shape(0)));
    for (py::ssize_t i = 0; i < input.shape(0); ++i)
        value.counts[static_cast<std::size_t>(i)] = view(i);
}
} // namespace
void bindSimulator(py::module_ &root) {
    auto m = root.def_submodule("simulator");
    py::class_<SimulationInput>(m, "SimulationInput")
        .def(py::init<>())
        .def_readwrite("sample", &SimulationInput::sample)
        .def_readwrite("setup", &SimulationInput::setup);
    py::class_<Spectrum>(m, "Spectrum")
        .def(py::init<>())
        .def_readwrite("label", &Spectrum::label)
        .def_property("counts", &counts, &setCounts);
    py::class_<SimulationFailure>(m, "SimulationFailure")
        .def_readonly("sample_index", &SimulationFailure::sampleIndex)
        .def_readonly("method_label", &SimulationFailure::methodLabel)
        .def_readonly("type", &SimulationFailure::type)
        .def_readonly("message", &SimulationFailure::message);
    py::class_<SimulationResult>(m, "SimulationResult")
        .def_readonly("spectra", &SimulationResult::spectra)
        .def_readonly("elemental_spectra", &SimulationResult::elementalSpectra)
        .def_readonly("failure", &SimulationResult::failure);
    py::class_<SimulationOptions>(m, "SimulationOptions")
        .def(py::init<>())
        .def_readwrite("capture_elemental_spectra", &SimulationOptions::captureElementalSpectra)
        .def_readwrite("continue_after_failure", &SimulationOptions::continueAfterFailure);
    py::class_<ISimulator, std::shared_ptr<ISimulator>>(m, "ISimulator")
        .def("request_stop", &ISimulator::requestStop)
        .def("reset_stop", &ISimulator::resetStop);
    py::class_<DummySimulator, ISimulator, std::shared_ptr<DummySimulator>>(m, "DummySimulator")
        .def(py::init<std::size_t>(), py::arg("channels") = 1024)
        .def("simulate_batch", &DummySimulator::simulateBatch,
             py::call_guard<py::gil_scoped_release>());
    py::class_<SimnraMethod>(m, "SimnraMethod")
        .def(py::init<>())
        .def_readwrite("label", &SimnraMethod::label)
        .def_readwrite("reference_file", &SimnraMethod::referenceFile);
    py::class_<SimnraSimulatorConfig>(m, "SimnraSimulatorConfig")
        .def(py::init<>())
        .def_readwrite("methods", &SimnraSimulatorConfig::methods)
        .def_readwrite("workers", &SimnraSimulatorConfig::workers)
        .def_readwrite("multithreaded_apartment", &SimnraSimulatorConfig::multithreadedApartment)
        .def_readwrite("thread_priority", &SimnraSimulatorConfig::threadPriority)
        .def_readwrite("fast_calculation", &SimnraSimulatorConfig::fastCalculation);
    py::class_<SimnraSimulator, ISimulator, std::shared_ptr<SimnraSimulator>>(m, "SimnraSimulator")
        .def(py::init<SimnraSimulatorConfig>())
        .def("simulate_batch", &SimnraSimulator::simulateBatch,
             py::call_guard<py::gil_scoped_release>())
        .def("request_stop", &SimnraSimulator::requestStop)
        .def("reset_stop", &SimnraSimulator::resetStop)
        .def("close", &SimnraSimulator::close);
}
