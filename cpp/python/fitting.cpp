#include <ibeamlab/fitting/fit.h>
#include <ibeamlab/io/reader.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
namespace py = pybind11;
namespace f = ibeamlab::fitting;
namespace c = ibeamlab::core;
void bind_fitting(py::module_ &m) {
    py::class_<c::MeasuredSpectrum>(
        m, "MeasuredSpectrum",
        "Observed counts, optional validity mask and variances; independent of a sample.")
        .def(py::init<>(), "Create an empty observation; populate fields before validation.")
        .def_readwrite("measurement", &c::MeasuredSpectrum::measurement)
        .def_readwrite("counts", &c::MeasuredSpectrum::counts)
        .def_readwrite("valid", &c::MeasuredSpectrum::valid)
        .def_readwrite("variances", &c::MeasuredSpectrum::variances)
        .def_readwrite("source", &c::MeasuredSpectrum::source)
        .def("validate", &c::MeasuredSpectrum::validate,
             "Validate labels, dimensions, counts and masks.");
    py::class_<c::MeasuredData>(
        m, "MeasuredData",
        "Measured spectra and measurement configurations, without a sample assumption.")
        .def(py::init<>())
        .def_readwrite("setup", &c::MeasuredData::setup)
        .def_readwrite("spectra", &c::MeasuredData::spectra)
        .def("validate", &c::MeasuredData::validate,
             "Validate one observation per configured measurement.");
    using namespace ibeamlab::io;
    py::enum_<SpectrumFormat>(m, "SpectrumFormat")
        .value("COUNTS", SpectrumFormat::Counts)
        .value("CHANNEL_COUNTS", SpectrumFormat::ChannelCounts)
        .value("CSV", SpectrumFormat::Csv);
    py::class_<ReadOptions>(
        m, "SpectrumReadOptions",
        "Explicit text parsing options; column indices and channel base are independent.")
        .def(py::init<>())
        .def_readwrite("format", &ReadOptions::format)
        .def_readwrite("counts_column", &ReadOptions::counts_column)
        .def_readwrite("channel_column", &ReadOptions::channel_column)
        .def_readwrite("variance_column", &ReadOptions::variance_column)
        .def_readwrite("channel_base", &ReadOptions::channel_base)
        .def_readwrite("skip_rows", &ReadOptions::skip_rows)
        .def_readwrite("maximum_channels", &ReadOptions::maximum_channels)
        .def_readwrite("header", &ReadOptions::header)
        .def_readwrite("delimiter", &ReadOptions::delimiter)
        .def_readwrite("comment", &ReadOptions::comment)
        .def_readwrite("stop_row", &ReadOptions::stop_row);
    m.def("read_spectrum", &read_spectrum, py::arg("path"), py::arg("measurement"),
          py::arg("options") = ReadOptions{}, py::call_guard<py::gil_scoped_release>(),
          "Read explicit channels/counts without inferring geometry or calibration.");
    py::class_<f::ElementArealDensity>(m, "ElementArealDensity")
        .def(py::init<std::size_t, std::string>(), py::arg("layer"), py::arg("element"));
    py::class_<f::Parameter>(m, "FitParameter",
                             "Bounded physical coordinate with optional tied targets.")
        .def(py::init<>())
        .def_readwrite("name", &f::Parameter::name)
        .def_readwrite("target", &f::Parameter::target)
        .def_readwrite("bounds", &f::Parameter::bounds)
        .def_readwrite("tied_targets", &f::Parameter::tied_targets)
        .def_readwrite("linked_targets", &f::Parameter::linked_targets);
    py::class_<f::Composition>(m, "FitComposition",
                               "Bounded composition encoded using N-1 stick-breaking coordinates.")
        .def(py::init<>())
        .def_readwrite("layer", &f::Composition::layer)
        .def_readwrite("elements", &f::Composition::elements)
        .def_readwrite("bounds", &f::Composition::bounds)
        .def_readwrite("name_prefix", &f::Composition::name_prefix);
    py::class_<f::Region>(m, "FitRegion")
        .def(py::init<std::size_t, std::size_t, double>(), py::arg("start"), py::arg("stop"),
             py::arg("weight") = 1.)
        .def_readwrite("start", &f::Region::start)
        .def_readwrite("stop", &f::Region::stop)
        .def_readwrite("weight", &f::Region::weight);
    py::class_<f::Selection>(m, "FitSelection")
        .def(py::init<>())
        .def_readwrite("measurement", &f::Selection::measurement)
        .def_readwrite("weight", &f::Selection::weight)
        .def_readwrite("regions", &f::Selection::regions);
    py::enum_<f::Loss>(m, "FitLoss")
        .value("DATA_CHI_SQUARE", f::Loss::DataChiSquare)
        .value("VARIANCE_CHI_SQUARE", f::Loss::VarianceChiSquare)
        .value("NEYMAN", f::Loss::Neyman)
        .value("PEARSON", f::Loss::Pearson)
        .value("MSE", f::Loss::Mse)
        .value("POISSON", f::Loss::Poisson)
        .value("GAUSSIAN", f::Loss::Gaussian);
    py::class_<f::Problem>(m, "FitProblem",
                           "Fixed measured data, sample assumption and physical parameter bounds.")
        .def(py::init<>())
        .def_readwrite("data", &f::Problem::data)
        .def_readwrite("initial_sample", &f::Problem::initial_sample)
        .def_readwrite("parameters", &f::Problem::parameters)
        .def_readwrite("compositions", &f::Problem::compositions)
        .def_readwrite("selections", &f::Problem::selections)
        .def_readwrite("loss", &f::Problem::loss)
        .def_readwrite("data_floor", &f::Problem::data_floor)
        .def_readwrite("profile", &f::Problem::profile)
        .def_readwrite("profile_parameters", &f::Problem::profile_parameters)
        .def_readwrite("missing_observations_as_zero", &f::Problem::missing_observations_as_zero)
        .def_readwrite("allow_overlapping_regions", &f::Problem::allow_overlapping_regions)
        .def_readwrite("region_normalization", &f::Problem::region_normalization)
        .def_readwrite("missing_prediction", &f::Problem::missing_prediction);
    py::class_<f::Predictor, std::shared_ptr<f::Predictor>>(
        m, "FitPredictor", "Nonconcurrent batch prediction adapter.");
    py::class_<f::ForwardPredictor, f::Predictor, std::shared_ptr<f::ForwardPredictor>>(
        m, "ForwardFitPredictor")
        .def(py::init<ibeamlab::inference::ForwardModel &>(), py::keep_alive<1, 2>());
    py::class_<f::SimulatorPredictor, f::Predictor, std::shared_ptr<f::SimulatorPredictor>>(
        m, "SimulatorFitPredictor")
        .def(py::init<ibeamlab::simulation::Simulator &>(), py::keep_alive<1, 2>());
    py::class_<f::Contribution>(m, "FitContribution")
        .def_readonly("measurement", &f::Contribution::measurement)
        .def_readonly("cost", &f::Contribution::cost)
        .def_readonly("channels", &f::Contribution::channels)
        .def_property_readonly("mean", &f::Contribution::mean);
    py::class_<f::Progress>(m, "FitProgress")
        .def_readonly("evaluations", &f::Progress::evaluations)
        .def_readonly("invalid", &f::Progress::invalid)
        .def_readonly("best_cost", &f::Progress::best_cost);
    py::class_<f::Result>(m, "FitResult", "Best evaluated experiment and its matching prediction.")
        .def_readonly("experiment", &f::Result::experiment)
        .def_readonly("prediction", &f::Result::prediction)
        .def_readonly("coordinates", &f::Result::coordinates)
        .def_readonly("parameter_names", &f::Result::parameter_names)
        .def_readonly("parameter_values", &f::Result::parameter_values)
        .def_readonly("contributions", &f::Result::contributions)
        .def_readonly("history", &f::Result::history)
        .def_readonly("cost", &f::Result::cost)
        .def_readonly("evaluations", &f::Result::evaluations)
        .def_readonly("invalid", &f::Result::invalid)
        .def_readonly("status", &f::Result::status)
        .def_readonly("message", &f::Result::message)
        .def_property_readonly("feasible", &f::Result::feasible);
    py::class_<f::Objective>(m, "FitObjective",
                             "Reusable batch objective; one instance per concurrent fit.")
        .def(py::init<f::Problem, std::shared_ptr<f::Predictor>>(), py::keep_alive<1, 3>(),
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("dimension", &f::Objective::dimension)
        .def_property_readonly("bounds",
                               [](const f::Objective &o) {
                                   std::vector<std::pair<double, double>> r;
                                   for (const auto &b : o.bounds())
                                       r.emplace_back(b.minimum, b.maximum);
                                   return r;
                               })
        .def_property_readonly("coordinate_names",
                               [](const f::Objective &o) {
                                   return std::vector<std::string>(o.coordinate_names().begin(),
                                                                   o.coordinate_names().end());
                               })
        .def_property_readonly("initial_coordinates",
                               [](const f::Objective &o) {
                                   return std::vector<double>(o.initial_coordinates().begin(),
                                                              o.initial_coordinates().end());
                               })
        .def(
            "evaluate_batch",
            [](f::Objective &o, py::array_t<double, py::array::c_style> rows,
               py::array_t<double, py::array::c_style> out) {
                if (rows.ndim() != 2 || std::size_t(rows.shape(1)) != o.dimension() ||
                    out.ndim() != 1 || out.shape(0) != rows.shape(0) || !out.writeable())
                    throw py::value_error("expected candidates [population,dimension] and writable "
                                          "costs [population]");
                py::gil_scoped_release release;
                o.evaluate_batch({rows.data(), std::size_t(rows.size())},
                                 std::size_t(rows.shape(0)),
                                 {out.mutable_data(), std::size_t(out.size())});
            },
            py::arg("candidates").noconvert(), py::arg("out").noconvert(),
            "Score a contiguous float64 population into caller-owned costs.")
        .def(
            "decode",
            [](const f::Objective &o, const std::vector<double> &x) { return o.decode(x); },
            "Decode optimizer coordinates to a physical experiment.")
        .def("best_result", &f::Objective::best_result,
             "Copy the best evaluated experiment and prediction.")
        .def("progress", &f::Objective::progress, "Return evaluation counters and best cost.")
        .def("request_stop", &f::Objective::request_stop, "Request cooperative cancellation.")
        .def_property_readonly("stopped", &f::Objective::stopped);
    py::class_<f::Options>(m, "FitOptions")
        .def(py::init<>())
        .def_readwrite("max_evaluations", &f::Options::max_evaluations)
        .def_readwrite("population_size", &f::Options::population_size)
        .def_readwrite("seed", &f::Options::seed)
        .def_readwrite("record_history", &f::Options::record_history)
        .def_readwrite("algorithm", &f::Options::algorithm)
        .def_readwrite("settings", &f::Options::settings)
        .def_readwrite("initial_guesses", &f::Options::initial_guesses);
    m.def("arrde_available", &f::arrde_available,
          "Return whether native ARRDE was enabled at build time.");
    m.def(
        "fit_arrde",
        [](f::Objective &o, f::Options options, py::object callback,
           std::vector<f::Options> stages) {
            auto progress = [&](const f::Progress &p) {
                py::gil_scoped_acquire lock;
                if (PyErr_CheckSignals() != 0) {
                    if (PyErr_ExceptionMatches(PyExc_KeyboardInterrupt)) {
                        PyErr_Clear();
                        return true;
                    }
                    throw py::error_already_set();
                }
                if (callback.is_none())
                    return false;
                try {
                    auto returned =
                        callback(py::cast(f::Progress(p), py::return_value_policy::move));
                    const auto stop = PyObject_IsTrue(returned.ptr());
                    if (stop < 0)
                        throw py::error_already_set();
                    return stop != 0;
                } catch (py::error_already_set &error) {
                    if (error.matches(PyExc_KeyboardInterrupt)) {
                        error.restore();
                        PyErr_Clear();
                        return true;
                    }
                    throw;
                }
            };
            py::gil_scoped_release release;
            return stages.empty() ? f::fit(o, options, progress)
                                  : f::fit_pipeline(o, stages, progress);
        },
        py::arg("objective"), py::arg("options"), py::arg("progress") = py::none(),
        py::arg("stages") = std::vector<f::Options>{},
        "Run native optimization with true population prediction; callbacks return True to stop.");
}
