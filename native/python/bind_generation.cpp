#include "bindings.h"
#include <ibeamlab/data_generator.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py = pybind11;
using namespace ibeamlab::generation;
void bindGeneration(py::module_ &r) {
    auto m = r.def_submodule("generation");
    py::class_<LayerThickness>(m, "LayerThickness")
        .def(py::init<>())
        .def_readwrite("layer", &LayerThickness::layer);
    py::class_<SpeciesConcentration>(m, "SpeciesConcentration")
        .def(py::init<>())
        .def_readwrite("layer", &SpeciesConcentration::layer)
        .def_readwrite("element", &SpeciesConcentration::element);
#define DETECTOR_TARGET(Type, Name)                                                                \
    py::class_<Type>(m, Name).def(py::init<>()).def_readwrite("detector", &Type::detector)
    DETECTOR_TARGET(BeamEnergy, "BeamEnergy");
    DETECTOR_TARGET(BeamSpread, "BeamSpread");
    DETECTOR_TARGET(CalibrationLinear, "CalibrationLinear");
    DETECTOR_TARGET(CalibrationOffset, "CalibrationOffset");
    DETECTOR_TARGET(CalibrationQuadratic, "CalibrationQuadratic");
    DETECTOR_TARGET(DetectorResolution, "DetectorResolution");
    DETECTOR_TARGET(ParticlesSr, "ParticlesSr");
#undef DETECTOR_TARGET
    py::class_<ParameterSpec>(m, "ParameterSpec")
        .def(py::init<>())
        .def_readwrite("name", &ParameterSpec::name)
        .def_readwrite("target", &ParameterSpec::target)
        .def_readwrite("lower_bound", &ParameterSpec::lowerBound)
        .def_readwrite("upper_bound", &ParameterSpec::upperBound)
        .def_readwrite("fixed_value", &ParameterSpec::fixedValue)
        .def_readwrite("unit", &ParameterSpec::unit);
    py::class_<MethodConfig>(m, "MethodConfig")
        .def(py::init<>())
        .def_readwrite("label", &MethodConfig::label)
        .def_readwrite("iba_method", &MethodConfig::ibaMethod)
        .def_readwrite("reference_file", &MethodConfig::referenceFile);
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("sample", &GenerationConfig::sample)
        .def_readwrite("setup", &GenerationConfig::setup)
        .def_readwrite("methods", &GenerationConfig::methods)
        .def_readwrite("parameters", &GenerationConfig::parameters)
        .def("validate", &GenerationConfig::validate)
        .def("materialize", &GenerationConfig::materialize);
    py::enum_<FailurePolicy>(m, "FailurePolicy")
        .value("STOP", FailurePolicy::Stop)
        .value("RECORD", FailurePolicy::Record)
        .value("DISCARD", FailurePolicy::Discard);
    py::class_<GenerationOptions>(m, "GenerationOptions")
        .def(py::init<>())
        .def_readwrite("samples", &GenerationOptions::samples)
        .def_readwrite("batch_size", &GenerationOptions::batchSize)
        .def_readwrite("shard_count", &GenerationOptions::shardCount)
        .def_readwrite("seed", &GenerationOptions::seed)
        .def_readwrite("failure_policy", &GenerationOptions::failurePolicy);
    py::class_<GenerationProgress>(m, "GenerationProgress")
        .def_readonly("attempted", &GenerationProgress::attempted)
        .def_readonly("accepted", &GenerationProgress::accepted)
        .def_readonly("invalid", &GenerationProgress::invalid)
        .def_readonly("failed", &GenerationProgress::failed)
        .def_readonly("total", &GenerationProgress::total);
    py::class_<GenerationSummary>(m, "GenerationSummary")
        .def_readonly("requested", &GenerationSummary::requested)
        .def_readonly("accepted", &GenerationSummary::accepted)
        .def_readonly("invalid", &GenerationSummary::invalid)
        .def_readonly("failed", &GenerationSummary::failed)
        .def_readonly("cancelled", &GenerationSummary::cancelled);
    py::class_<DataGenerator>(m, "DataGenerator")
        .def(py::init<GenerationConfig, std::shared_ptr<ibeamlab::simulator::ISimulator>>())
        .def(
            "generate",
            [](DataGenerator &self, const std::filesystem::path &path,
               const GenerationOptions &options, py::object callback) {
                ProgressCallback progress;
                if (!callback.is_none()) {
                    progress = [callback = std::move(callback)](const GenerationProgress &value) {
                        py::gil_scoped_acquire acquire;
                        callback(value);
                    };
                }
                py::gil_scoped_release release;
                return self.generate(path, options, std::move(progress));
            },
            py::arg("path"), py::arg("options"), py::arg("progress") = py::none())
        .def("request_stop", &DataGenerator::requestStop);
}
