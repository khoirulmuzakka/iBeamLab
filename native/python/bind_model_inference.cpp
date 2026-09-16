#include "bindings.h"
#include <ibeamlab/inference.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py = pybind11;
void bindModel(py::module_ &r) {
    auto m = r.def_submodule("model");
    py::class_<ibeamlab::model::ModelMetadata>(m, "ModelMetadata")
        .def_readonly("format_version", &ibeamlab::model::ModelMetadata::formatVersion)
        .def_readonly("task", &ibeamlab::model::ModelMetadata::task)
        .def_readonly("input_dimension", &ibeamlab::model::ModelMetadata::inputDimension)
        .def_readonly("output_dimension", &ibeamlab::model::ModelMetadata::outputDimension)
        .def_readonly("methods", &ibeamlab::model::ModelMetadata::methodNames)
        .def_readonly("spectrum_lengths", &ibeamlab::model::ModelMetadata::spectrumLengths)
        .def_readonly("input_features", &ibeamlab::model::ModelMetadata::inputFeatures)
        .def_readonly("output_features", &ibeamlab::model::ModelMetadata::outputFeatures)
        .def_readonly("output_units", &ibeamlab::model::ModelMetadata::outputUnits)
        .def_readonly("model_size", &ibeamlab::model::ModelMetadata::modelSize)
        .def_readonly("model_crc32", &ibeamlab::model::ModelMetadata::modelChecksum);
    py::class_<ibeamlab::model::ModelPackage>(m, "ModelPackage")
        .def_static("open", &ibeamlab::model::ModelPackage::open)
        .def_property_readonly("metadata", &ibeamlab::model::ModelPackage::metadata,
                               py::return_value_policy::reference_internal);
}
void bindInference(py::module_ &r) {
    auto m = r.def_submodule("inference");
    using namespace ibeamlab::inference;
    py::class_<InferenceOptions>(m, "InferenceOptions")
        .def(py::init<>())
        .def_readwrite("intra_op_threads", &InferenceOptions::intraOpThreads)
        .def_readwrite("inter_op_threads", &InferenceOptions::interOpThreads)
        .def_readwrite("execution_provider", &InferenceOptions::executionProvider)
        .def_readwrite("enable_graph_optimizations", &InferenceOptions::enableGraphOptimizations);
    py::class_<NamedValue>(m, "NamedValue")
        .def_readonly("name", &NamedValue::name)
        .def_readonly("value", &NamedValue::value)
        .def_readonly("unit", &NamedValue::unit);
    py::class_<InferenceResult>(m, "InferenceResult")
        .def_readonly("values", &InferenceResult::values);
    py::class_<OnnxInferenceEngine>(m, "OnnxInferenceEngine")
        .def(py::init([](const ibeamlab::model::ModelPackage &package, InferenceOptions options) {
                 return std::make_unique<OnnxInferenceEngine>(package, std::move(options));
             }),
             py::arg("package"), py::arg("options") = InferenceOptions{})
        .def("predict", &OnnxInferenceEngine::predict, py::call_guard<py::gil_scoped_release>());
}
