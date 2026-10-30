#include "inference/session.h"
#include <ibeamlab/core/isotopes.h>
#include <ibeamlab/core/validation.h>
#include <ibeamlab/generation/generator.h>
#include <ibeamlab/inference/models.h>
#include <ibeamlab/processing/pileup.h>
#include <ibeamlab/processing/rebin.h>
#include <ibeamlab/simulation/simnra.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py = pybind11;
using namespace ibeamlab;
using Array = py::array_t<double, py::array::c_style>;
std::span<const double> data(const Array &a) {
    return {a.data(), static_cast<std::size_t>(a.size())};
}
std::span<double> writable(Array &a) {
    if (!a.writeable())
        throw std::invalid_argument("output array must be writable");
    return {a.mutable_data(), static_cast<std::size_t>(a.size())};
}
py::array_t<double> readonly_array(py::object owner, std::span<const double> values,
                                   std::vector<py::ssize_t> shape) {
    py::array_t<double> result(shape, values.data(), owner);
    result.attr("setflags")(false);
    return result;
}
#define FIELD(T, N) .def_readwrite(#N, &T::N)
PYBIND11_MODULE(_ibeamlab, m) {
    m.doc() = "iBeamLab C++ library; fitting is not implemented.";
    py::class_<core::Isotope>(m, "Isotope")
        .def(py::init<>()) FIELD(core::Isotope, massNumber)
            FIELD(core::Isotope, fraction);
    py::class_<core::IsotopeData>(m, "IsotopeData")
        .def_readonly("massNumber", &core::IsotopeData::massNumber)
        .def_readonly("exactMass", &core::IsotopeData::exactMass);
    m.def("natural_isotopes", [](const std::string &element) {
        auto values = core::natural_isotopes(element);
        return std::vector<core::Isotope>(values.begin(), values.end());
    });
    m.def("isotope_data", [](const std::string &element) {
        auto values = core::isotope_data(element);
        return std::vector<core::IsotopeData>(values.begin(), values.end());
    });
    m.def("validate_isotopes", [](const std::string &element, const std::vector<core::Isotope> &values) {
        core::validate_isotopes(element, values);
    });
    py::class_<core::Species>(m, "Species")
        .def(py::init<>()) FIELD(core::Species, element) FIELD(core::Species, concentration)
            FIELD(core::Species, isotopes);
    py::class_<core::Layer>(m, "Layer")
        .def(py::init<>()) FIELD(core::Layer, thickness) FIELD(core::Layer, roughness)
            FIELD(core::Layer, porosityFraction) FIELD(core::Layer, poreDiameter)
                FIELD(core::Layer, species);
    py::class_<core::Sample>(m, "Sample")
        .def(py::init<>()) FIELD(core::Sample, layers)
        .def("validate", &core::Sample::validate);
    py::class_<core::Beam>(m, "Beam").def(py::init<>()) FIELD(core::Beam, particle)
        FIELD(core::Beam, energy) FIELD(core::Beam, spread);
    py::class_<core::Detector>(m, "Detector")
        .def(py::init<>()) FIELD(core::Detector, calibrationLinear)
            FIELD(core::Detector, calibrationOffset) FIELD(core::Detector, calibrationQuadratic)
                FIELD(core::Detector, resolution) FIELD(core::Detector, info);
    py::class_<core::Acquisition>(m, "Acquisition")
        .def(py::init<>()) FIELD(core::Acquisition, particlesSr) FIELD(core::Acquisition, realTime)
            FIELD(core::Acquisition, liveTime);
    py::class_<core::Measurement>(m, "Measurement")
        .def(py::init<>()) FIELD(core::Measurement, label) FIELD(core::Measurement, beam)
            FIELD(core::Measurement, detector) FIELD(core::Measurement, acquisition);
    py::class_<core::ExperimentalSetup>(m, "ExperimentalSetup")
        .def(py::init<>()) FIELD(core::ExperimentalSetup, measurements)
        .def("validate", &core::ExperimentalSetup::validate);
    py::class_<core::Experiment>(m, "Experiment")
        .def(py::init<>()) FIELD(core::Experiment, sample) FIELD(core::Experiment, setup)
        .def("validate", &core::Experiment::validate);
    py::class_<core::ValidationIssue>(m, "ValidationIssue")
        .def_readonly("path", &core::ValidationIssue::path)
        .def_readonly("message", &core::ValidationIssue::message);
    m.def("validate", &core::validate);
    py::class_<core::LayerThickness>(m, "LayerThickness")
        .def(py::init<std::size_t>()) FIELD(core::LayerThickness, layer);
    py::class_<core::SpeciesConcentration>(m, "SpeciesConcentration")
        .def(py::init<std::size_t, std::string>()) FIELD(core::SpeciesConcentration, layer)
            FIELD(core::SpeciesConcentration, element);
#define MEASUREMENT_TARGET(T)                                                                      \
    py::class_<core::T>(m, #T).def(py::init<std::string>()) FIELD(core::T, measurement)
    MEASUREMENT_TARGET(BeamEnergy);
    MEASUREMENT_TARGET(BeamSpread);
    MEASUREMENT_TARGET(CalibrationLinear);
    MEASUREMENT_TARGET(CalibrationOffset);
    MEASUREMENT_TARGET(CalibrationQuadratic);
    MEASUREMENT_TARGET(DetectorResolution);
    MEASUREMENT_TARGET(ParticlesSr);
    py::class_<core::ParameterBounds>(m, "ParameterBounds")
        .def(py::init<double, double>()) FIELD(core::ParameterBounds, minimum)
            FIELD(core::ParameterBounds, maximum)
        .def("validate", &core::ParameterBounds::validate)
        .def("contains", &core::ParameterBounds::contains);
    py::class_<core::ParameterUpdate>(m, "ParameterUpdate")
        .def(py::init<core::ParameterTarget, double>()) FIELD(core::ParameterUpdate, target)
            FIELD(core::ParameterUpdate, value);
    m.def("read_parameter", &core::read_parameter);
    m.def("write_parameter", &core::write_parameter);
    m.def("parameter_key", &core::parameter_key);
    m.def("apply", [](core::Experiment &x, const std::vector<core::ParameterUpdate> &updates) {
        core::apply(x, updates);
    });
    py::class_<core::Spectrum>(m, "Spectrum")
        .def(py::init<>())
        .def(py::init<std::string, core::Counts>()) FIELD(core::Spectrum, measurement)
            FIELD(core::Spectrum, counts)
        .def_property_readonly(
            "counts_array",
            [](py::object owner) {
                const auto &s = owner.cast<const core::Spectrum &>();
                return readonly_array(owner, s.counts, {static_cast<py::ssize_t>(s.counts.size())});
            },
            "Read-only NumPy view retaining its C++ owner; do not replace counts while a view "
            "exists.");
    py::enum_<processing::Coverage>(m, "Coverage")
        .value("ZERO_OUTSIDE", processing::Coverage::ZeroOutside)
        .value("REQUIRE_FULL", processing::Coverage::RequireFull);
    m.def(
        "rebin",
        [](const Array &counts, const Array &old_edges, const Array &new_edges,
           processing::Coverage coverage) {
            return processing::rebin(data(counts), data(old_edges), data(new_edges), coverage);
        },
        py::arg("counts").noconvert(), py::arg("old_edges").noconvert(),
        py::arg("new_edges").noconvert(), py::arg("coverage") = processing::Coverage::ZeroOutside);
    m.def(
        "rebin_into",
        [](const Array &counts, const Array &old_edges, const Array &new_edges, Array output,
           processing::Coverage coverage) {
            auto out = writable(output);
            py::gil_scoped_release release;
            processing::rebin(data(counts), data(old_edges), data(new_edges), out, coverage);
        },
        py::arg("counts").noconvert(), py::arg("old_edges").noconvert(),
        py::arg("new_edges").noconvert(), py::arg("output").noconvert(),
        py::arg("coverage") = processing::Coverage::ZeroOutside);
    py::class_<processing::PileupWorkspace>(m, "PileupWorkspace").def(py::init<>());
    m.def(
        "apply_pileup_into",
        [](const Array &input, Array output, double real, double live, double tau,
           processing::PileupWorkspace &w) {
            auto out = writable(output);
            py::gil_scoped_release release;
            processing::apply_pileup(data(input), out, real, live, tau, w);
        },
        py::arg("input").noconvert(), py::arg("output").noconvert(), py::arg("real_time"),
        py::arg("live_time"), py::arg("tau"), py::arg("workspace"));
    m.def(
        "remove_pileup_into",
        [](const Array &input, Array output, double real, double live, double tau) {
            auto out = writable(output);
            py::gil_scoped_release release;
            processing::remove_pileup(data(input), out, real, live, tau);
        },
        py::arg("input").noconvert(), py::arg("output").noconvert(), py::arg("real_time"),
        py::arg("live_time"), py::arg("tau"));
    py::class_<simulation::Failure>(m, "Failure")
        .def(py::init<>()) FIELD(simulation::Failure, measurement)
            FIELD(simulation::Failure, message);
    py::class_<simulation::Result>(m, "SimulationResult")
        .def(py::init<>()) FIELD(simulation::Result, spectra) FIELD(simulation::Result, failure);
    py::class_<simulation::Simulator>(m, "Simulator")
        .def("request_stop", &simulation::Simulator::request_stop)
        .def("reset_stop", &simulation::Simulator::reset_stop)
        .def("provenance", &simulation::Simulator::provenance)
        .def("simulate",
             [](simulation::Simulator &s, const core::Experiment &x) {
                 simulation::Result r;
                 {
                     py::gil_scoped_release release;
                     s.simulate(x, r);
                 }
                 return r;
             })
        .def("simulate_batch",
             [](simulation::Simulator &s, const std::vector<core::Experiment> &x) {
                 py::gil_scoped_release release;
                 return s.simulate_batch(x);
             });
    py::class_<simulation::DummySimulator, simulation::Simulator>(m, "DummySimulator")
        .def(py::init<std::size_t>(), py::arg("channels") = 5000);
    py::class_<simulation::SimnraMethod>(m, "SimnraMethod")
        .def(py::init<>()) FIELD(simulation::SimnraMethod, measurement)
            FIELD(simulation::SimnraMethod, reference) FIELD(simulation::SimnraMethod, particle);
    py::class_<simulation::SimnraConfig>(m, "SimnraConfig")
        .def(py::init<>()) FIELD(simulation::SimnraConfig, methods)
            FIELD(simulation::SimnraConfig, workers)
                FIELD(simulation::SimnraConfig, multithreaded_apartment)
                    FIELD(simulation::SimnraConfig, fast_calculation)
                        FIELD(simulation::SimnraConfig, reuse_target_topology);
    py::class_<simulation::SimnraSimulator, simulation::Simulator>(m, "SimnraSimulator")
        .def(py::init<simulation::SimnraConfig>())
        .def("close", &simulation::SimnraSimulator::close,
             py::call_guard<py::gil_scoped_release>());
    py::class_<datasets::Parameter>(m, "Parameter")
        .def(py::init<std::string, core::ParameterTarget, std::string>())
            FIELD(datasets::Parameter, name) FIELD(datasets::Parameter, target)
                FIELD(datasets::Parameter, unit);
    py::class_<datasets::Metadata>(m, "DatasetMetadata")
        .def(py::init<>()) FIELD(datasets::Metadata, reference)
            FIELD(datasets::Metadata, parameters) FIELD(datasets::Metadata, provenance)
                FIELD(datasets::Metadata, requested) FIELD(datasets::Metadata, accepted)
                    FIELD(datasets::Metadata, failed) FIELD(datasets::Metadata, discarded);
    py::class_<datasets::Record>(m, "Record")
        .def(py::init<>()) FIELD(datasets::Record, sample_index) FIELD(datasets::Record, sample_id)
            FIELD(datasets::Record, parameters) FIELD(datasets::Record, result);
    py::class_<datasets::Reader>(m, "DatasetReader")
        .def(py::init<const std::filesystem::path &>())
        .def_property_readonly("metadata", &datasets::Reader::metadata,
                               py::return_value_policy::copy)
        .def("rewind", &datasets::Reader::rewind)
        .def("maximum_spectrum_lengths", &datasets::Reader::maximum_spectrum_lengths,
             py::call_guard<py::gil_scoped_release>())
        .def(
            "__iter__", [](datasets::Reader &r) -> datasets::Reader & { return r; },
            py::return_value_policy::reference_internal)
        .def("__next__", [](datasets::Reader &r) {
            datasets::Record record;
            bool present;
            {
                py::gil_scoped_release release;
                present = r.next(record);
            }
            if (!present)
                throw py::stop_iteration();
            return record;
        });
    py::class_<datasets::Writer>(m, "DatasetWriter")
        .def(py::init<const std::filesystem::path &, datasets::Metadata, std::size_t>(),
             py::arg("path"), py::arg("metadata"), py::arg("records_per_shard") = 64)
        .def("append", &datasets::Writer::append)
        .def("discard", &datasets::Writer::discard)
        .def("finish", &datasets::Writer::finish);
    py::enum_<generation::FailurePolicy>(m, "FailurePolicy")
        .value("STOP", generation::FailurePolicy::Stop)
        .value("RECORD", generation::FailurePolicy::Record)
        .value("DISCARD", generation::FailurePolicy::Discard);
    py::class_<generation::GenerationPlan>(m, "GenerationPlan")
        .def(py::init<>()) FIELD(generation::GenerationPlan, reference)
            FIELD(generation::GenerationPlan, parameters)
        .def("validate", &generation::GenerationPlan::validate);
    py::class_<generation::Summary>(m, "GenerationSummary")
        .def_readonly("accepted", &generation::Summary::accepted)
        .def_readonly("failed", &generation::Summary::failed)
        .def_readonly("discarded", &generation::Summary::discarded)
        .def_readonly("cancelled", &generation::Summary::cancelled);
    m.def(
        "generate",
        [](const generation::GenerationPlan &plan, const Array &rows, simulation::Simulator &sim,
           const std::filesystem::path &path, generation::FailurePolicy policy, py::object progress,
           std::string provenance, std::size_t shard_size, std::size_t batch_size) {
            if (rows.ndim() != 2)
                throw std::invalid_argument("rows must be a two-dimensional float64 array");
            if (static_cast<std::size_t>(rows.shape(1)) != plan.parameters.size())
                throw std::invalid_argument("parameter column count mismatch");
            auto samples = static_cast<std::size_t>(rows.shape(0));
            simulation::Progress callback;
            if (!progress.is_none())
                callback = [&progress](std::size_t completed, std::size_t total) {
                    py::gil_scoped_acquire lock;
                    progress(completed, total);
                };
            py::gil_scoped_release release;
            return generation::generate(plan, data(rows), samples, sim, path, policy, callback,
                                        provenance, shard_size, batch_size);
        },
        py::arg("plan"), py::arg("rows").noconvert(), py::arg("simulator"), py::arg("path"),
        py::arg("failure_policy") = generation::FailurePolicy::Stop,
        py::arg("progress") = py::none(), py::arg("provenance") = "",
        py::arg("records_per_shard") = 64, py::arg("batch_size") = 256);
    py::enum_<model::Direction>(m, "Direction")
        .value("FORWARD", model::Direction::Forward)
        .value("INVERSE", model::Direction::Inverse);
    py::enum_<model::SpectrumState>(m, "SpectrumState")
        .value("BARE", model::SpectrumState::Bare).value("PILED_UP", model::SpectrumState::PiledUp);
    py::class_<model::SpectrumSpec>(m, "SpectrumSpec")
        .def(py::init<std::string, std::size_t, model::SpectrumState, std::string>(),
             py::arg("measurement"), py::arg("channels"), py::arg("state") = model::SpectrumState::Bare,
             py::arg("unit") = "counts") FIELD(model::SpectrumSpec, measurement)
            FIELD(model::SpectrumSpec, channels) FIELD(model::SpectrumSpec, state) FIELD(model::SpectrumSpec, unit);
    py::class_<model::SpectraSchema>(m, "SpectraSchema")
        .def(py::init<>())
        .def(py::init<std::vector<model::SpectrumSpec>>()) FIELD(model::SpectraSchema, spectra)
        .def("width", &model::SpectraSchema::width)
        .def("offset", &model::SpectraSchema::offset);
    py::class_<model::Log1p>(m, "Log1p").def(py::init<>());
    py::class_<model::MinMaxScale>(m, "MinMaxScale")
        .def(py::init<core::Counts, core::Counts, double, double, bool>(), py::arg("minimum"),
             py::arg("maximum"), py::arg("low") = 0., py::arg("high") = 1., py::arg("clip") = false)
            FIELD(model::MinMaxScale, minimum) FIELD(model::MinMaxScale, maximum)
                FIELD(model::MinMaxScale, low) FIELD(model::MinMaxScale, high)
                    FIELD(model::MinMaxScale, clip);
    py::class_<model::StandardScale>(m, "StandardScale")
        .def(py::init<core::Counts, core::Counts>()) FIELD(model::StandardScale, mean)
            FIELD(model::StandardScale, deviation);
    py::class_<model::MultiplyScale>(m, "MultiplyScale")
        .def(py::init<core::Counts>()) FIELD(model::MultiplyScale, factors);
    py::class_<model::CustomTransform>(m, "CustomTransform")
        .def(py::init<>()) FIELD(model::CustomTransform, identifier)
            FIELD(model::CustomTransform, version) FIELD(model::CustomTransform, parameters);
    py::enum_<model::TransformDirection>(m, "TransformDirection")
        .value("FORWARD", model::TransformDirection::Forward).value("INVERSE", model::TransformDirection::Inverse);
    py::enum_<model::TransformLocation>(m, "TransformLocation")
        .value("RUNTIME", model::TransformLocation::Runtime).value("GRAPH", model::TransformLocation::Graph);
    py::class_<model::TransformPipeline>(m, "TransformPipeline")
        .def(py::init<model::TensorPipeline, model::TransformDirection, model::TransformLocation>(),
            py::arg("steps") = model::TensorPipeline{}, py::arg("direction") = model::TransformDirection::Forward,
            py::arg("location") = model::TransformLocation::Runtime)
        FIELD(model::TransformPipeline, steps) FIELD(model::TransformPipeline, direction) FIELD(model::TransformPipeline, location);
    py::class_<model::SpectrumTransform>(m, "SpectrumTransform")
        .def(py::init<std::string, model::TransformPipeline>())
        FIELD(model::SpectrumTransform, measurement) FIELD(model::SpectrumTransform, transform);
    py::class_<model::ExposureScale>(m, "ExposureScale").def(py::init<>());
    py::class_<model::Rebin>(m, "Rebin")
        .def(py::init<std::size_t, bool>(), py::arg("channels") = 0,
             py::arg("require_full_coverage") = true) FIELD(model::Rebin, channels)
            FIELD(model::Rebin, require_full_coverage);
    py::enum_<model::AcquisitionSource>(m, "AcquisitionSource")
        .value("REQUESTED", model::AcquisitionSource::Requested)
        .value("REFERENCE", model::AcquisitionSource::Reference);
    py::class_<model::Pileup>(m, "Pileup")
        .def(py::init<double, model::AcquisitionSource>(), py::arg("tau_seconds"),
             py::arg("acquisition") = model::AcquisitionSource::Requested)
            FIELD(model::Pileup, tau_seconds) FIELD(model::Pileup, acquisition);
    py::class_<model::RemovePileup>(m, "RemovePileup")
        .def(py::init<double, double, model::AcquisitionSource>(), py::arg("tau_seconds"),
             py::arg("relative_tolerance") = 1e-7,
             py::arg("acquisition") = model::AcquisitionSource::Requested)
            FIELD(model::RemovePileup, tau_seconds) FIELD(model::RemovePileup, relative_tolerance)
                FIELD(model::RemovePileup, acquisition);
    py::class_<model::SpectrumProcessing>(m, "SpectrumProcessing")
        .def(py::init<std::string, std::vector<model::SpectrumOperation>>())
            FIELD(model::SpectrumProcessing, measurement) FIELD(model::SpectrumProcessing, steps);
    py::class_<model::EdpSchema>(m, "EdpSchema")
        .def(py::init<std::vector<std::string>, std::size_t, bool, std::string>(),
             py::arg("elements"), py::arg("layers"), py::arg("uncertainty") = false,
             py::arg("unit") = "1e15 atoms/cm2")
        FIELD(model::EdpSchema, elements) FIELD(model::EdpSchema, layers)
        FIELD(model::EdpSchema, uncertainty) FIELD(model::EdpSchema, unit);
    py::class_<model::ParameterInputSchema>(m, "ParameterInputSchema")
        .def(py::init<>()) FIELD(model::ParameterInputSchema, reference_sample)
            FIELD(model::ParameterInputSchema, parameters);
    py::class_<model::ElementComposition>(m, "ElementComposition")
        .def(py::init<>()) FIELD(model::ElementComposition, element) FIELD(model::ElementComposition, isotopes);
    py::class_<model::LayerInputSchema>(m, "LayerInputSchema")
        .def(py::init<>()) FIELD(model::LayerInputSchema, elements) FIELD(model::LayerInputSchema, isotope_compositions);
    py::class_<model::ForwardModelMetadata>(m, "ForwardModelMetadata")
        .def(py::init<>()) FIELD(model::ForwardModelMetadata, reference_setup)
            FIELD(model::ForwardModelMetadata, input_preparation)
                FIELD(model::ForwardModelMetadata, output_spectra)
                        FIELD(model::ForwardModelMetadata, input_transform)
            FIELD(model::ForwardModelMetadata, output_transform) FIELD(model::ForwardModelMetadata, output_preparation);
    py::class_<model::InverseModelMetadata>(m, "InverseModelMetadata")
        .def(py::init<>()) FIELD(model::InverseModelMetadata, reference_setup)
        FIELD(model::InverseModelMetadata, input_spectra) FIELD(model::InverseModelMetadata, input_preparation)
        FIELD(model::InverseModelMetadata, input_transform) FIELD(model::InverseModelMetadata, output_transform)
        FIELD(model::InverseModelMetadata, output_preparation);
    py::class_<model::ModelMetadata>(m, "ModelMetadata")
        .def(py::init<>())
        .def(py::init<model::ForwardModelMetadata>())
        .def(py::init<model::InverseModelMetadata>()) FIELD(model::ModelMetadata, model)
        .def_property_readonly("direction", &model::ModelMetadata::direction)
            FIELD(model::ModelMetadata, input_name) FIELD(model::ModelMetadata, output_name)
                FIELD(model::ModelMetadata, provenance)
        .def("validate", &model::ModelMetadata::validate)
        .def("input_width", &model::ModelMetadata::input_width)
        .def("input_rank", &model::ModelMetadata::input_rank)
        .def("layer_input", &model::ModelMetadata::layer_input)
        .def("output_width", &model::ModelMetadata::output_width);
    py::class_<model::Package>(m, "Package")
        .def_static("open", &model::Package::open, py::call_guard<py::gil_scoped_release>())
        .def_static("from_onnx", &model::Package::from_onnx)
        .def("write", &model::Package::write) FIELD(model::Package, metadata);
    py::enum_<inference::ExecutionProvider>(m, "ExecutionProvider")
        .value("CPU", inference::ExecutionProvider::Cpu)
        .value("CUDA", inference::ExecutionProvider::Cuda);
    py::class_<inference::InferenceOptions>(m, "InferenceOptions")
        .def(py::init<>()) FIELD(inference::InferenceOptions, execution_provider)
            FIELD(inference::InferenceOptions, device_id) FIELD(inference::InferenceOptions, intra_op_threads)
                FIELD(inference::InferenceOptions, inter_op_threads) FIELD(inference::InferenceOptions, enable_graph_optimizations)
                    FIELD(inference::InferenceOptions, allow_tf32) FIELD(inference::InferenceOptions, apply_pileup)
        .def("validate", &inference::InferenceOptions::validate);
    m.def("available_execution_providers", &inference::available_execution_providers);
    py::enum_<inference::PredictionDomain>(m, "PredictionDomain")
        .value("MEASUREMENT", inference::PredictionDomain::Measurement)
        .value("REFERENCE", inference::PredictionDomain::Reference);
    py::class_<inference::ForwardModel>(m, "ForwardModel")
        .def(py::init<model::Package, inference::InferenceOptions>(), py::arg("package"),
             py::arg("options") = inference::InferenceOptions{}, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("options", &inference::ForwardModel::options, py::return_value_policy::copy)
        .def("predict",
             py::overload_cast<const core::Experiment &, inference::PredictionDomain>(&inference::ForwardModel::predict),
             py::arg("experiment"), py::arg("domain") = inference::PredictionDomain::Measurement,
             py::call_guard<py::gil_scoped_release>())
        .def("predict_reference", py::overload_cast<const core::Experiment &>(&inference::ForwardModel::predict_reference),
             py::call_guard<py::gil_scoped_release>())
        .def("predict_areal_density", [](inference::ForwardModel &model,
             py::array_t<float, py::array::c_style> values, const core::ExperimentalSetup *setup, inference::PredictionDomain domain) {
            if (values.ndim() != 2 || static_cast<std::size_t>(values.shape(1)) != model.metadata().input_width())
                throw std::invalid_argument("AD must have shape [layers,elements]");
            py::gil_scoped_release release;
            return model.predict_areal_density({values.data(), static_cast<std::size_t>(values.size())},
                                               static_cast<std::size_t>(values.shape(0)), setup, domain);
        }, py::arg("densities").noconvert(), py::arg("setup") = nullptr,
           py::arg("domain") = inference::PredictionDomain::Measurement)
        .def("predict_areal_density_batch", [](inference::ForwardModel &model,
             py::array_t<float, py::array::c_style> values, const core::ExperimentalSetup *setup,
             py::object destination, inference::PredictionDomain domain) {
            if (values.ndim() != 3 || static_cast<std::size_t>(values.shape(2)) != model.metadata().input_width())
                throw std::invalid_argument("AD must have shape [batch,layers,elements]");
            const auto channels = model.spectrum_channels(domain);
            std::size_t width = 0;
            for (auto count : channels) width += count;
            py::array_t<double, py::array::c_style> output;
            if (destination.is_none())
                output = py::array_t<double>({values.shape(0), static_cast<py::ssize_t>(width)});
            else {
                if (!py::isinstance<py::array_t<double>>(destination))
                    throw std::invalid_argument("out must be a writable contiguous float64 array");
                auto array = destination.cast<py::array>();
                if (!(array.flags() & py::array::c_style) || !array.writeable())
                    throw std::invalid_argument("out must be a writable contiguous float64 array");
                output = destination.cast<py::array_t<double, py::array::c_style>>();
                if (output.ndim() != 2 || output.shape(0) != values.shape(0) ||
                    output.shape(1) != static_cast<py::ssize_t>(width))
                    throw std::invalid_argument("out must have shape [batch,total_processed_channels]");
            }
            const auto input_begin = reinterpret_cast<std::uintptr_t>(values.data());
            const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data());
            if (values.nbytes() && output.nbytes() && input_begin < output_begin + output.nbytes() &&
                output_begin < input_begin + values.nbytes())
                throw std::invalid_argument("AD input and out must not overlap");
            {
                py::gil_scoped_release release;
                model.predict_areal_density_batch({values.data(), static_cast<std::size_t>(values.size())},
                    static_cast<std::size_t>(values.shape(0)), static_cast<std::size_t>(values.shape(1)),
                    {output.mutable_data(), static_cast<std::size_t>(output.size())}, setup, domain);
            }
            py::dict result;
            const auto &f = std::get<model::ForwardModelMetadata>(model.metadata().model);
            std::size_t offset = 0;
            for (std::size_t i = 0; i < channels.size(); ++i) {
                result[py::str(f.output_spectra.spectra[i].measurement)] = py::array_t<double>(
                    {values.shape(0), static_cast<py::ssize_t>(channels[i])},
                    {static_cast<py::ssize_t>(width * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
                    output.data() + offset, output);
                offset += channels[i];
            }
            return result;
        }, py::arg("densities").noconvert(), py::arg("setup") = nullptr, py::arg("out") = py::none(), py::arg("domain") = inference::PredictionDomain::Measurement)
        .def("predict_batch",
             [](inference::ForwardModel &model, const std::vector<core::Experiment> &inputs, inference::PredictionDomain domain) {
                 py::gil_scoped_release release;
                 return model.predict_batch(inputs, domain);
             }, py::arg("inputs"), py::arg("domain") = inference::PredictionDomain::Measurement);
    py::class_<inference::InverseInput>(m, "InverseInput")
        .def(py::init<>()) FIELD(inference::InverseInput, setup)
            FIELD(inference::InverseInput, spectra)
                FIELD(inference::InverseInput, pileup_already_removed);
    py::class_<inference::Prediction>(m, "Prediction")
        .def_readonly("elements", &inference::Prediction::elements)
        .def_readonly("layers", &inference::Prediction::layers)
        .def_readonly("values", &inference::Prediction::values)
        .def_readonly("presence_probability", &inference::Prediction::presence_probability)
        .def_readonly("posterior_std", &inference::Prediction::posterior_std)
        .def_readonly("uncertainty_predicted", &inference::Prediction::uncertainty_predicted)
        .def_readonly("unit", &inference::Prediction::unit)
        .def_property_readonly("values_array",
                               [](py::object owner) {
                                   const auto &p = owner.cast<const inference::Prediction &>();
                                   return readonly_array(
                                       owner, p.values,
                                       {static_cast<py::ssize_t>(p.layers),
                                        static_cast<py::ssize_t>(p.elements.size())});
                               })
        .def_property_readonly("presence_probability_array",
                               [](py::object owner) {
                                   const auto &p = owner.cast<const inference::Prediction &>();
                                   return readonly_array(
                                       owner, p.presence_probability,
                                       {static_cast<py::ssize_t>(p.layers),
                                        static_cast<py::ssize_t>(p.elements.size())});
                               })
        .def_property_readonly("posterior_std_array", [](py::object owner) {
            const auto &p = owner.cast<const inference::Prediction &>();
            return readonly_array(
                owner, p.posterior_std,
                {static_cast<py::ssize_t>(p.layers), static_cast<py::ssize_t>(p.elements.size())});
        });
    py::class_<inference::InverseModel>(m, "InverseModel")
        .def(py::init<model::Package, inference::InferenceOptions>(), py::arg("package"),
             py::arg("options") = inference::InferenceOptions{}, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("options", &inference::InverseModel::options, py::return_value_policy::copy)
        .def("predict",
             py::overload_cast<const inference::InverseInput &>(&inference::InverseModel::predict),
             py::call_guard<py::gil_scoped_release>())
        .def(
            "predict_batch",
            [](inference::InverseModel &model, const std::vector<inference::InverseInput> &inputs, inference::PredictionDomain domain) {
                py::gil_scoped_release release;
                return model.predict_batch(inputs, domain);
            }, py::arg("inputs"), py::arg("domain") = inference::PredictionDomain::Measurement)
        .def("predict_reference",
             [](inference::InverseModel &model, const std::vector<core::Spectrum> &spectra) {
                 inference::Prediction out;
                 {
                     py::gil_scoped_release release;
                     model.predict_reference(spectra, out);
                 }
                 return out;
             });
    m.def(
        "run_graph",
        [](const model::Package &p, py::array_t<float, py::array::c_style> input, inference::InferenceOptions options) {
            if (input.ndim() != static_cast<py::ssize_t>(p.metadata.input_rank()))
                throw std::invalid_argument("graph input rank mismatch");
            auto batch = static_cast<std::size_t>(input.shape(0));
            if (static_cast<std::size_t>(input.shape(input.ndim() - 1)) != p.metadata.input_width())
                throw std::invalid_argument("graph input width mismatch");
            py::array_t<float> output({static_cast<py::ssize_t>(batch),
                                       static_cast<py::ssize_t>(p.metadata.output_width())});
            {
                py::gil_scoped_release release;
                inference::detail::run_graph(
                    p, {const_cast<float *>(input.data()), static_cast<std::size_t>(input.size())},
                    {output.mutable_data(), static_cast<std::size_t>(output.size())}, batch, options,
                    p.metadata.layer_input() ? static_cast<std::size_t>(input.shape(1)) : 1);
            }
            return output;
        },
        py::arg("package"), py::arg("input").noconvert(), py::arg("options") = inference::InferenceOptions{});
}
