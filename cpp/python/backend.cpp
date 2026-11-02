#include <ibeamlab/analysis/executor.h>
#include <ibeamlab/analysis/predictor.h>
#include <ibeamlab/core/isotopes.h>
#include <ibeamlab/io/parser.h>
#include <ibeamlab/io/results.h>
#include <ibeamlab/profile/discretization.h>
#include <ibeamlab/simulation/simnra.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py = pybind11;
namespace p = ibeamlab::profile;
namespace f = ibeamlab::fitting;
namespace a = ibeamlab::analysis;
namespace io = ibeamlab::io;
void bind_backend(py::module_ &m) {
    py::enum_<p::FunctionKind>(m, "ProfileFunctionKind")
        .value("CONSTANT", p::FunctionKind::Constant)
        .value("POLYNOMIAL", p::FunctionKind::Polynomial)
        .value("ERF_POLYNOMIAL", p::FunctionKind::ErfPolynomial);
    py::class_<p::Function>(m, "ProfileFunction",
                            "Nonnegative composition weight in normalized local depth.")
        .def(py::init<>())
        .def_readwrite("kind", &p::Function::kind)
        .def_readwrite("coefficients", &p::Function::coefficients)
        .def("validate", &p::Function::validate)
        .def("evaluate", &p::Function::evaluate);
    py::enum_<p::Spacing>(m, "ProfileSpacing")
        .value("LINEAR", p::Spacing::Linear)
        .value("LOGARITHMIC", p::Spacing::Logarithmic);
    py::class_<p::Layer>(m, "ProfileLayer")
        .def(py::init<>())
        .def_readwrite("reference", &p::Layer::reference)
        .def_readwrite("functions", &p::Layer::functions)
        .def_readwrite("sublayers", &p::Layer::sublayers)
        .def_readwrite("spacing", &p::Layer::spacing);
    py::class_<p::Model>(m, "ProfileModel")
        .def(py::init<>())
        .def_readwrite("layers", &p::Model::layers)
        .def("validate", &p::Model::validate)
        .def("reference_sample", &p::Model::reference_sample);
    py::class_<p::Discretizer>(m, "ProfileDiscretizer")
        .def(py::init<const p::Model &>())
        .def("apply", &p::Discretizer::apply, py::call_guard<py::gil_scoped_release>());
    m.def("discretize_profile", &p::discretize, py::call_guard<py::gil_scoped_release>());
    py::class_<f::ProfileLink>(m, "FitProfileLink")
        .def(py::init<>())
        .def_readwrite("layer", &f::ProfileLink::layer)
        .def_readwrite("species", &f::ProfileLink::species)
        .def_readwrite("coefficient", &f::ProfileLink::coefficient)
        .def_readwrite("ratio", &f::ProfileLink::ratio);
    py::class_<f::ProfileParameter>(m, "FitProfileParameter")
        .def(py::init<>())
        .def_readwrite("name", &f::ProfileParameter::name)
        .def_readwrite("layer", &f::ProfileParameter::layer)
        .def_readwrite("species", &f::ProfileParameter::species)
        .def_readwrite("coefficient", &f::ProfileParameter::coefficient)
        .def_readwrite("bounds", &f::ProfileParameter::bounds)
        .def_readwrite("links", &f::ProfileParameter::links);
    py::class_<f::LinkedTarget>(m, "FitLinkedTarget")
        .def(py::init<>())
        .def_readwrite("target", &f::LinkedTarget::target)
        .def_readwrite("ratio", &f::LinkedTarget::ratio);
    py::enum_<f::LayerProperty>(m, "FitLayerProperty")
        .value("ROUGHNESS", f::LayerProperty::Roughness)
        .value("POROSITY_FRACTION", f::LayerProperty::PorosityFraction)
        .value("PORE_DIAMETER", f::LayerProperty::PoreDiameter);
    py::class_<f::LayerPropertyTarget>(m, "FitLayerPropertyTarget")
        .def(py::init<>())
        .def_readwrite("layer", &f::LayerPropertyTarget::layer)
        .def_readwrite("property", &f::LayerPropertyTarget::property);
    py::enum_<f::RegionNormalization>(m, "FitRegionNormalization")
        .value("SUM", f::RegionNormalization::Sum)
        .value("WEIGHT_SUM", f::RegionNormalization::WeightSum);
    py::enum_<f::MissingPrediction>(m, "FitMissingPrediction")
        .value("REJECT", f::MissingPrediction::Reject)
        .value("ZERO", f::MissingPrediction::Zero);
    py::class_<a::Route>(m, "AnalysisRoute")
        .def(py::init<>())
        .def_readwrite("measurements", &a::Route::measurements)
        .def_readwrite("predictor", &a::Route::predictor);
    py::class_<a::Plan>(m, "AnalysisPlan")
        .def(py::init<>())
        .def_readwrite("routes", &a::Plan::routes)
        .def("validate", &a::Plan::validate);
    py::class_<a::Predictor, f::Predictor, std::shared_ptr<a::Predictor>>(m, "AnalysisPredictor")
        .def(py::init<a::Plan>(), py::keep_alive<1, 2>());
    py::class_<a::Task>(m, "AnalysisTask")
        .def(py::init<>())
        .def_readwrite("id", &a::Task::id)
        .def_readwrite("problem", &a::Task::problem)
        .def_readwrite("options", &a::Task::options)
        .def_readwrite("stages", &a::Task::stages)
        .def_readwrite("initial_from", &a::Task::initial_from);
    py::class_<a::Outcome>(m, "AnalysisOutcome")
        .def_readonly("id", &a::Outcome::id)
        .def_readonly("result", &a::Outcome::result)
        .def_readonly("error", &a::Outcome::error);
    py::class_<a::Executor>(m, "AnalysisExecutor")
        .def(py::init([](std::vector<std::shared_ptr<f::Predictor>> workers) {
                 auto count = workers.size();
                 return std::make_unique<a::Executor>(
                     count, [workers = std::move(workers)](std::size_t i) { return workers[i]; });
             }),
             py::keep_alive<1, 2>())
        .def(
            "run",
            [](a::Executor &executor, const std::vector<a::Task> &tasks, py::object callback) {
                auto progress = [&](const std::string &id, std::size_t worker,
                                    const f::Progress &p) {
                    py::gil_scoped_acquire lock;
                    if (PyErr_CheckSignals() != 0)
                        throw py::error_already_set();
                    if (callback.is_none())
                        return false;
                    try {
                        auto value = callback(id, worker, py::cast(f::Progress(p)));
                        int stop = PyObject_IsTrue(value.ptr());
                        if (stop < 0)
                            throw py::error_already_set();
                        return stop != 0;
                    } catch (py::error_already_set &e) {
                        std::string message = e.what();
                        e.restore();
                        PyErr_Clear();
                        throw std::runtime_error(message);
                    }
                };
                py::gil_scoped_release release;
                return executor.run(tasks, progress);
            },
            py::arg("tasks"), py::arg("progress") = py::none())
        .def("request_stop", &a::Executor::request_stop);
    py::class_<io::TimeField>(m, "SpectrumTimeField")
        .def(py::init<>())
        .def_readwrite("marker", &io::TimeField::marker)
        .def_readwrite("seconds_per_unit", &io::TimeField::seconds_per_unit)
        .def_readwrite("fallback", &io::TimeField::fallback);
    py::class_<io::Parser>(m, "SpectrumParser")
        .def(py::init<>())
        .def_readwrite("records", &io::Parser::records)
        .def_readwrite("start_marker", &io::Parser::start_marker)
        .def_readwrite("end_marker", &io::Parser::end_marker)
        .def_readwrite("real_time", &io::Parser::real_time)
        .def_readwrite("live_time", &io::Parser::live_time)
        .def_readwrite("missing_channels_are_zero", &io::Parser::missing_channels_are_zero);
    py::class_<io::ImportedSpectrum>(m, "ImportedSpectrum")
        .def_readonly("spectrum", &io::ImportedSpectrum::spectrum)
        .def_readonly("real_time", &io::ImportedSpectrum::real_time)
        .def_readonly("live_time", &io::ImportedSpectrum::live_time);
    m.def(
        "read_instrument_spectrum",
        [](const std::filesystem::path &path, std::string label, const io::Parser &p) {
            return io::read_spectrum(path, std::move(label), p);
        },
        py::call_guard<py::gil_scoped_release>());
    m.def("write_fit_result", &io::write_result, py::arg("path"), py::arg("result"),
          py::arg("provenance") = "", py::call_guard<py::gil_scoped_release>());
    py::class_<ibeamlab::simulation::Capabilities>(m, "SimulatorCapabilities")
        .def_readonly("elemental_spectra", &ibeamlab::simulation::Capabilities::elemental_spectra)
        .def_readonly("project_export", &ibeamlab::simulation::Capabilities::project_export);
    py::class_<ibeamlab::simulation::ElementSpectrum>(m, "ElementSpectrum")
        .def_readonly("measurement", &ibeamlab::simulation::ElementSpectrum::measurement)
        .def_readonly("element", &ibeamlab::simulation::ElementSpectrum::element)
        .def_readonly("counts", &ibeamlab::simulation::ElementSpectrum::counts);
    m.def("atomic_number", &ibeamlab::core::atomic_number);
}
