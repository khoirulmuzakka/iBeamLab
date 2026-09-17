#include "bindings.h"
#include <ibeamlab/datasets.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py = pybind11;
using namespace ibeamlab::datasets;
void bindDatasets(py::module_ &r) {
    auto m = r.def_submodule("datasets");
    py::class_<FailureRecord>(m, "FailureRecord")
        .def_readonly("sample_index", &FailureRecord::sampleIndex)
        .def_readonly("sample_id", &FailureRecord::sampleId)
        .def_readonly("method_label", &FailureRecord::methodLabel)
        .def_readonly("type", &FailureRecord::type)
        .def_readonly("message", &FailureRecord::message);
    py::class_<DatasetMetadata>(m, "DatasetMetadata")
        .def_readonly("format_version", &DatasetMetadata::formatVersion)
        .def_readonly("complete", &DatasetMetadata::complete)
        .def_readonly("requested", &DatasetMetadata::requested)
        .def_readonly("accepted", &DatasetMetadata::accepted)
        .def_readonly("invalid", &DatasetMetadata::invalid)
        .def_readonly("failed", &DatasetMetadata::failed)
        .def_readonly("generation_config_toml", &DatasetMetadata::generationConfigToml)
        .def_readonly("generation_options_toml", &DatasetMetadata::generationOptionsToml)
        .def_readonly("sampling_config_toml", &DatasetMetadata::samplingConfigToml)
        .def_readonly("simulator_config_toml", &DatasetMetadata::simulatorConfigToml)
        .def_readonly("parameter_names", &DatasetMetadata::parameterNames)
        .def_readonly("spectrum_labels", &DatasetMetadata::spectrumLabels)
        .def_readonly("spectrum_lengths", &DatasetMetadata::spectrumLengths);
    py::class_<DatasetRecord>(m, "DatasetRecord")
        .def_readonly("sample_index", &DatasetRecord::sampleIndex)
        .def_readonly("sample_id", &DatasetRecord::sampleId)
        .def_readonly("open_parameters", &DatasetRecord::openParameters)
        .def_readonly("result", &DatasetRecord::result);
    py::class_<DatasetReader>(m, "DatasetReader")
        .def(py::init<std::filesystem::path>())
        .def_property_readonly("metadata", &DatasetReader::metadata,
                               py::return_value_policy::reference_internal)
        .def("read_all", &DatasetReader::readAll, py::call_guard<py::gil_scoped_release>())
        .def("read_failures", &DatasetReader::readFailures,
             py::call_guard<py::gil_scoped_release>());
}
