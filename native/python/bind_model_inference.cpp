#include "bindings.h"
#include <ibeamlab/forward_model.h>
#include <ibeamlab/inverse_model.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py=pybind11;
void bindModel(py::module_&root){using namespace ibeamlab::model;auto m=root.def_submodule("model");
 py::class_<TransformSpec>(m,"TransformSpec").def(py::init<>()).def_readwrite("type",&TransformSpec::type).def_readwrite("input_dimension",&TransformSpec::inputDimension).def_readwrite("factor",&TransformSpec::factor).def_readwrite("offset",&TransformSpec::offset).def_readwrite("low",&TransformSpec::low).def_readwrite("high",&TransformSpec::high).def_readwrite("minimum",&TransformSpec::minimum).def_readwrite("scale",&TransformSpec::scale).def_readwrite("mean",&TransformSpec::mean).def_readwrite("deviation",&TransformSpec::deviation).def_readwrite("transforms",&TransformSpec::transforms);
 py::enum_<ModelType>(m,"ModelType").value("INVERSE",ModelType::Inverse).value("FORWARD",ModelType::Forward);
 py::class_<SpectrumSpec>(m,"SpectrumSpec").def(py::init<>()).def_readwrite("label",&SpectrumSpec::label).def_readwrite("length",&SpectrumSpec::length);
 py::class_<InverseModelMetadata>(m,"InverseModelMetadata").def(py::init<>()).def_readwrite("sample_template",&InverseModelMetadata::sampleTemplate).def_readwrite("setup_template",&InverseModelMetadata::setupTemplate).def_readwrite("input_spectra",&InverseModelMetadata::inputSpectra).def_readwrite("output_parameters",&InverseModelMetadata::outputParameters);
 py::class_<ForwardModelMetadata>(m,"ForwardModelMetadata").def(py::init<>()).def_readwrite("sample_template",&ForwardModelMetadata::sampleTemplate).def_readwrite("setup_template",&ForwardModelMetadata::setupTemplate).def_readwrite("input_parameters",&ForwardModelMetadata::inputParameters).def_readwrite("output_spectra",&ForwardModelMetadata::outputSpectra);
 py::class_<ModelMetadata>(m,"ModelMetadata").def(py::init<>()).def_readwrite("format_version",&ModelMetadata::formatVersion).def_readwrite("created_utc",&ModelMetadata::createdUtc).def_readwrite("model_type",&ModelMetadata::modelType).def_readwrite("class_name",&ModelMetadata::className).def_readwrite("input_name",&ModelMetadata::inputName).def_readwrite("output_name",&ModelMetadata::outputName).def_readwrite("input_dimension",&ModelMetadata::inputDimension).def_readwrite("output_dimension",&ModelMetadata::outputDimension).def_readwrite("opset_version",&ModelMetadata::opsetVersion).def_readwrite("input_transform",&ModelMetadata::inputTransform).def_readwrite("output_transform",&ModelMetadata::outputTransform).def_readwrite("inverse",&ModelMetadata::inverse).def_readwrite("forward",&ModelMetadata::forward).def_readonly("model_size",&ModelMetadata::modelSize).def_readonly("model_crc32",&ModelMetadata::modelChecksum);
 py::class_<ModelPackage>(m,"ModelPackage").def_static("open",&ModelPackage::open).def_static("from_onnx",&ModelPackage::fromOnnx).def("write",&ModelPackage::write).def_property_readonly("metadata",&ModelPackage::metadata,py::return_value_policy::reference_internal);
}
void bindInference(py::module_&root){using namespace ibeamlab::inference;auto m=root.def_submodule("inference");
 py::class_<InferenceOptions>(m,"InferenceOptions").def(py::init<>()).def_readwrite("intra_op_threads",&InferenceOptions::intraOpThreads).def_readwrite("inter_op_threads",&InferenceOptions::interOpThreads).def_readwrite("execution_provider",&InferenceOptions::executionProvider).def_readwrite("enable_graph_optimizations",&InferenceOptions::enableGraphOptimizations);
 py::class_<NamedValue>(m,"NamedValue").def_readonly("name",&NamedValue::name).def_readonly("value",&NamedValue::value).def_readonly("unit",&NamedValue::unit);
 py::class_<InverseResult>(m,"InverseResult").def_readonly("sample",&InverseResult::sample).def_readonly("parameters",&InverseResult::parameters);
 py::class_<ForwardResult>(m,"ForwardResult").def_readonly("spectra",&ForwardResult::spectra);
 py::class_<InverseModel>(m,"InverseModel").def(py::init<const std::filesystem::path&,InferenceOptions>(),py::arg("package"),py::arg("options")=InferenceOptions{}).def("predict",&InverseModel::predict,py::call_guard<py::gil_scoped_release>());
 py::class_<ForwardModel>(m,"ForwardModel").def(py::init<const std::filesystem::path&,InferenceOptions>(),py::arg("package"),py::arg("options")=InferenceOptions{}).def("predict",&ForwardModel::predict,py::call_guard<py::gil_scoped_release>());
}
