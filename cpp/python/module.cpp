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
// Python backends use public Python value objects via the facade callbacks.
// Native simulators retain their existing path and never acquire the GIL here.
class PythonSimulator final : public simulation::Simulator {
    py::function single_, batch_, provenance_;
  public:
    PythonSimulator(py::function single, py::function batch, py::function provenance)
        : single_(std::move(single)), batch_(std::move(batch)), provenance_(std::move(provenance)) {}
    void simulate(const core::Experiment &experiment, simulation::Result &output) override {
        py::gil_scoped_acquire lock;
        auto result = single_(py::cast(&experiment, py::return_value_policy::reference));
        output = std::move(result.cast<simulation::Result &>());
    }
    std::string provenance() const override {
        py::gil_scoped_acquire lock;
        return provenance_().cast<std::string>();
    }
    std::vector<simulation::Result> simulate_batch(std::span<const core::Experiment> inputs,
                                                 simulation::Progress progress) override {
        if (stopped()) return {};
        py::gil_scoped_acquire lock;
        py::list experiments;
        for (const auto &input : inputs)
            experiments.append(py::cast(&input, py::return_value_policy::reference));
        auto output = batch_(experiments).cast<py::list>();
        if (static_cast<std::size_t>(output.size()) != inputs.size() && !stopped())
            throw std::invalid_argument("Python simulator returned the wrong number of batch results");
        std::vector<simulation::Result> results;
        results.reserve(output.size());
        for (auto result : output) results.push_back(std::move(result.cast<simulation::Result &>()));
        for (std::size_t i = 0; progress && i < results.size() && !stopped(); ++i)
            progress(i + 1, inputs.size());
        return results;
    }
};

#define FIELD(T, N) .def_readwrite(#N, &T::N)
PYBIND11_MODULE(_ibeamlab, m) {
    m.doc() = "Native iBeamLab execution engine. Use the ibeamlab facade for Python descriptions, simulation, generation and inference. Units are keV, seconds, degrees and 1e15 atoms/cm2.";
    py::class_<core::Isotope>(m, "Isotope", R"ibdoc(Native isotope identity and atomic fraction.

Attributes
----------
massNumber : int
    Positive mass number; element identity is supplied by the containing Species.
fraction : float
    Finite nonnegative fraction within the element. Explicit mixtures sum to one.

Notes
-----
Use the Python Isotope facade for immutable descriptions. An empty Species
isotope list requests natural composition; massNumber=0 is not a natural marker.)ibdoc")
        .def(py::init<>()) FIELD(core::Isotope, massNumber)
            FIELD(core::Isotope, fraction);
    py::class_<core::IsotopeData>(m, "IsotopeData", R"ibdoc(Read-only isotope database record.

Attributes
----------
massNumber : int
    Isotope mass number.
exactMass : float
    Atomic mass in atomic mass units (u).)ibdoc")
        .def_readonly("massNumber", &core::IsotopeData::massNumber)
        .def_readonly("exactMass", &core::IsotopeData::exactMass);
    m.def("natural_isotopes", [](const std::string &element) {
        auto values = core::natural_isotopes(element);
        return std::vector<core::Isotope>(values.begin(), values.end());
    }, R"ibdoc(Return native natural-isotope fraction records for an element.

Parameters
----------
element : str
    Canonical symbol.

Returns
-------
list[Isotope]
    Owned records; empty when no natural default exists. Unknown symbols raise ValueError.)ibdoc");
    m.def("isotope_data", [](const std::string &element) {
        auto values = core::isotope_data(element);
        return std::vector<core::IsotopeData>(values.begin(), values.end());
    }, R"ibdoc(Return read-only native isotope identity/mass records.

Parameters
----------
element : str
    Canonical element symbol.

Returns
-------
list[IsotopeData]
    Supported masses in atomic mass units. Unknown elements raise ValueError.)ibdoc");
    m.def("validate_isotopes", [](const std::string &element, const std::vector<core::Isotope> &values) {
        core::validate_isotopes(element, values);
    }, R"ibdoc(Validate an explicit or default isotope composition.

Parameters
----------
element : str
    Canonical symbol.
values : sequence of Isotope
    Unique supported mass numbers and normalized finite nonnegative fractions.
    Empty requests natural composition.

Raises
------
ValueError
    Composition is invalid or no requested default is available.)ibdoc");
    py::class_<core::Species>(m, "Species", R"ibdoc(Native element composition record.

Attributes
----------
element : str
    Canonical element symbol.
concentration : float
    Atomic fraction within a layer.
isotopes : list[Isotope]
    Explicit mixture; empty selects the framework natural default.)ibdoc")
        .def(py::init<>()) FIELD(core::Species, element) FIELD(core::Species, concentration)
            FIELD(core::Species, isotopes);
    py::class_<core::Layer>(m, "Layer", R"ibdoc(Native layer in surface-to-depth order.

Attributes
----------
thickness : float
    Areal thickness in 1e15 atoms/cm2.
roughness : float
    Backend roughness width.
porosityFraction : float
    Pore volume fraction in [0,1].
poreDiameter : float
    Pore diameter in backend areal-length units.
species : list[Species]
    Unique elements with atomic concentrations summing to one.)ibdoc")
        .def(py::init<>()) FIELD(core::Layer, thickness) FIELD(core::Layer, roughness)
            FIELD(core::Layer, porosityFraction) FIELD(core::Layer, poreDiameter)
                FIELD(core::Layer, species);
    py::class_<core::Sample>(m, "Sample", R"ibdoc(Native owning stack of physical layers.

Attributes
----------
layers : list[Layer]
    Surface-to-depth layer sequence.

Notes
-----
validate() checks dimensions, element/isotope identities and normalized
compositions. Use ibeamlab.Sample for immutable Python descriptions.)ibdoc")
        .def(py::init<>()) FIELD(core::Sample, layers)
        .def("validate", &core::Sample::validate, R"ibdoc(Validate dimensions, element/isotope identities and normalized compositions.
Raises ValueError when the physical sample contract is invalid.)ibdoc");
    py::class_<core::Beam>(m, "Beam", R"ibdoc(Native beam description.

Attributes
----------
particle : str
    Backend particle identifier.
energy : float
    Positive beam energy in keV.
spread : float
    Beam energy spread in keV FWHM.)ibdoc").def(py::init<>()) FIELD(core::Beam, particle)
        FIELD(core::Beam, energy) FIELD(core::Beam, spread);
    py::class_<core::Geometry>(m, "Geometry", R"ibdoc(Native detector measurement geometry in degrees.

Attributes
----------
incidenceAngle : float
    Alpha from the inward surface normal, in [0,90).
exitAngle : float
    Beta from the outward normal, in [0,180].
scatteringAngle : float
    Theta between incident and outgoing trajectories, in [0,180].

Notes
-----
validate() checks directional consistency, including non-coplanar geometry.
Default (0,0,180) means normal incidence and exact backscattering.)ibdoc")
        .def(py::init<>()) FIELD(core::Geometry, incidenceAngle) FIELD(core::Geometry, exitAngle)
            FIELD(core::Geometry, scatteringAngle).def("validate", &core::Geometry::validate, R"ibdoc(Validate angle ranges and coexistence of the three directions.
Angles use degrees; non-coplanar and transmission geometries are supported.
Raises ValueError for inconsistent geometry.)ibdoc");
    py::class_<core::Detector>(m, "Detector", R"ibdoc(Native detector response, calibration and geometry.

Attributes
----------
calibrationLinear : float
    Linear calibration in keV/channel.
calibrationOffset : float
    Channel-center offset in keV.
calibrationQuadratic : float
    Quadratic term in keV/channel squared.
resolution : float
    Energy resolution in keV FWHM.
info : str
    Descriptive information.
geometry : Geometry
    Complete detector-specific measurement geometry.)ibdoc")
        .def(py::init<>()) FIELD(core::Detector, calibrationLinear)
            FIELD(core::Detector, calibrationOffset) FIELD(core::Detector, calibrationQuadratic)
                FIELD(core::Detector, resolution) FIELD(core::Detector, info) FIELD(core::Detector, geometry);
    py::class_<core::Acquisition>(m, "Acquisition", R"ibdoc(Native measurement exposure and acquisition times.

Attributes
----------
particlesSr : float
    Incident particle count times solid angle in sr.
realTime : float
    Real time in seconds.
liveTime : float
    Live time in seconds.)ibdoc")
        .def(py::init<>()) FIELD(core::Acquisition, particlesSr) FIELD(core::Acquisition, realTime)
            FIELD(core::Acquisition, liveTime);
    py::class_<core::Measurement>(m, "Measurement", R"ibdoc(Native labelled measurement of the shared sample.

Attributes
----------
label : str
    Measurement identity.
beam : Beam
    Particle and energy settings.
detector : Detector
    Response, calibration and geometry.
acquisition : Acquisition
    Exposure and times.)ibdoc")
        .def(py::init<>()) FIELD(core::Measurement, label) FIELD(core::Measurement, beam)
            FIELD(core::Measurement, detector) FIELD(core::Measurement, acquisition);
    py::class_<core::ExperimentalSetup>(m, "ExperimentalSetup", R"ibdoc(Native owning collection of independently configured measurements.

Attributes
----------
measurements : list[Measurement]
    Nonempty measurements with unique labels.

Notes
-----
validate() includes geometry and physical setup checks.)ibdoc")
        .def(py::init<>()) FIELD(core::ExperimentalSetup, measurements)
        .def("validate", &core::ExperimentalSetup::validate, R"ibdoc(Validate measurement identities, beam values and detector/acquisition settings.
Includes per-detector geometry validation. Raises ValueError on invalid input.)ibdoc");
    py::class_<core::Experiment>(m, "Experiment", R"ibdoc(Native mutable sample and experimental setup.

Attributes
----------
sample : Sample
    Shared sample.
setup : ExperimentalSetup
    Measurement configurations.

Notes
-----
validate() checks the complete experiment. Python facade experiments cache
this representation; mutating a cached value bypasses their immutability.)ibdoc")
        .def(py::init<>()) FIELD(core::Experiment, sample) FIELD(core::Experiment, setup)
        .def("validate", &core::Experiment::validate, R"ibdoc(Validate the complete physical sample and experimental setup.
Raises ValueError on the first invalid physical field.)ibdoc");
    py::class_<core::ValidationIssue>(m, "ValidationIssue", R"ibdoc(Structured scientific validation diagnostic.

Attributes
----------
path : str
    Location of the invalid field.
message : str
    Diagnostic explaining the invalid value.)ibdoc")
        .def_readonly("path", &core::ValidationIssue::path)
        .def_readonly("message", &core::ValidationIssue::message);
    m.def("validate", &core::validate, R"ibdoc(Collect validation issues for a native Experiment.

Returns
-------
list[ValidationIssue]
    Empty when valid. At most the first sample and first setup diagnostic are
    collected; this is not an exhaustive listing of every invalid field.)ibdoc");
    py::class_<core::LayerThickness>(m, "LayerThickness", R"ibdoc(Select the areal thickness of a zero-based sample layer.

Parameters
----------
layer : int
    Zero-based layer index; values use 1e15 atoms/cm2.)ibdoc")
        .def(py::init<std::size_t>()) FIELD(core::LayerThickness, layer);
    py::class_<core::SpeciesConcentration>(m, "SpeciesConcentration", R"ibdoc(Select an elemental concentration in a zero-based layer.

Parameters
----------
layer : int
    Zero-based layer index.
element : str
    Element symbol; values are atomic fractions.)ibdoc")
        .def(py::init<std::size_t, std::string>()) FIELD(core::SpeciesConcentration, layer)
            FIELD(core::SpeciesConcentration, element);
#define MEASUREMENT_TARGET(T, DESCRIPTION)                                                                      \
    py::class_<core::T>(m, #T, DESCRIPTION).def(py::init<std::string>()) FIELD(core::T, measurement)
    MEASUREMENT_TARGET(IncidenceAngle, R"ibdoc(Select the detector incidence angle alpha of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use degree.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(ExitAngle, R"ibdoc(Select the detector exit angle beta of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use degree.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(ScatteringAngle, R"ibdoc(Select the detector scattering angle theta of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use degree.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(BeamEnergy, R"ibdoc(Select the beam energy of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use keV.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(BeamSpread, R"ibdoc(Select the beam energy spread of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use keV FWHM.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(CalibrationLinear, R"ibdoc(Select the linear calibration of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use keV/channel.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(CalibrationOffset, R"ibdoc(Select the calibration offset of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use keV.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(CalibrationQuadratic, R"ibdoc(Select the quadratic calibration of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use keV/channel squared.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(DetectorResolution, R"ibdoc(Select the detector resolution of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use keV FWHM.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    MEASUREMENT_TARGET(ParticlesSr, R"ibdoc(Select the exposure of a labelled measurement.

Parameters
----------
measurement : str
    Exact measurement label. Physical values use particles*sr.

Notes
-----
The target selects a physical field; it does not store a numeric value or bounds.)ibdoc");
    py::class_<core::ParameterBounds>(m, "ParameterBounds", R"ibdoc(Declare an inclusive finite interval.

Parameters
----------
minimum : float
    Lower bound.
maximum : float
    Upper bound, not smaller than minimum.

Notes
-----
validate() checks the interval; contains(value) tests a finite scalar.)ibdoc")
        .def(py::init<double, double>()) FIELD(core::ParameterBounds, minimum)
            FIELD(core::ParameterBounds, maximum)
        .def("validate", &core::ParameterBounds::validate, R"ibdoc(Validate a finite inclusive interval; raise ValueError if invalid.)ibdoc")
        .def("contains", &core::ParameterBounds::contains, R"ibdoc(Return whether a scalar is finite and lies in this inclusive interval.)ibdoc");
    py::class_<core::ParameterUpdate>(m, "ParameterUpdate", R"ibdoc(Assign one physical target value.

Parameters
----------
target : ParameterTarget
    Sample or labelled setup location.
value : float
    Finite scalar in the target physical units.

Notes
-----
apply() validates a complete set of updates atomically using an experiment copy.)ibdoc")
        .def(py::init<core::ParameterTarget, double>()) FIELD(core::ParameterUpdate, target)
            FIELD(core::ParameterUpdate, value);
    m.def("read_parameter", &core::read_parameter, R"ibdoc(Read one native physical parameter target from an Experiment.

Parameters
----------
input : Experiment
    Physical input.
target : ParameterTarget
    Layer or labelled measurement location.

Returns
-------
float
    Value in the physical units of the target. Invalid locations raise ValueError or IndexError.)ibdoc");
    m.def("write_parameter", &core::write_parameter, R"ibdoc(Assign a finite native physical value without copying an Experiment.

Parameters
----------
input : Experiment
    Mutable experiment.
target : ParameterTarget
    Physical location.
value : float
    Finite value in the target units.

Notes
-----
Intermediate composition/geometry may be inconsistent. Apply all coordinated
updates, then validate. This operation does not provide atomic multi-field updates.)ibdoc");
    m.def("parameter_key", &core::parameter_key, R"ibdoc(Return a stable identity string for a physical parameter target.

Parameters
----------
target : ParameterTarget
    Physical location.

Returns
-------
str
    Identity used to reject duplicate targets; it is not a display name or unit converter.)ibdoc");
    m.def("apply", [](core::Experiment &x, const std::vector<core::ParameterUpdate> &updates) {
        core::apply(x, updates);
    }, R"ibdoc(Apply native parameter updates atomically after complete validation.

Parameters
----------
experiment : Experiment
    Mutable experiment receiving successful updates.
updates : sequence of ParameterUpdate
    Unique targets and finite physical values.

Notes
-----
Creates a candidate copy, validates it, then commits. Failure leaves the original
unchanged. For hot loops use coordinated write_parameter calls and validate once.)ibdoc");
    py::class_<core::Spectrum>(m, "Spectrum", R"ibdoc(Own one labelled native count spectrum.

Parameters
----------
measurement : str
    Measurement label.
counts : array_like
    Contiguous float64 counts copied into native storage.

Notes
-----
counts_array is a read-only NumPy view retaining its native owner. The
counts field is a convenience vector conversion and can allocate a Python list.)ibdoc")
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
    py::enum_<processing::Coverage>(m, "Coverage", R"ibdoc(Choose behavior for destination energy bins outside source coverage.

Notes
-----
ZERO_OUTSIDE fills uncovered destination regions with zero. REQUIRE_FULL
requires the source interval to cover the entire destination interval.)ibdoc")
        .value("ZERO_OUTSIDE", processing::Coverage::ZeroOutside)
        .value("REQUIRE_FULL", processing::Coverage::RequireFull);
    m.def(
        "rebin",
        [](const Array &counts, const Array &old_edges, const Array &new_edges,
           processing::Coverage coverage) {
            return processing::rebin(data(counts), data(old_edges), data(new_edges), coverage);
        },
        py::arg("counts").noconvert(), py::arg("old_edges").noconvert(),
        py::arg("new_edges").noconvert(), py::arg("coverage") = processing::Coverage::ZeroOutside, R"ibdoc(Return rebinned counts from contiguous float64 arrays.

Input counts have N bins, old_edges N+1 edges and new_edges M+1 edges.
Edges must increase and use the same energy units. Coverage chooses whether
uncovered destination regions are zero-filled or rejected. Returns a native
vector converted to a Python list; rebin_into provides caller-owned output.)ibdoc");
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
        py::arg("coverage") = processing::Coverage::ZeroOutside, R"ibdoc(Rebin integrated counts into a writable contiguous float64 output.

Input counts have N bins, old_edges N+1 edges, new_edges M+1 edges and output M
bins. Arrays must be one-dimensional and valid. No implicit output conversion
is performed. Output must not overlap input arrays. Returns None.)ibdoc");
    py::class_<processing::PileupWorkspace>(m, "PileupWorkspace", R"ibdoc(Own reusable FFT scratch for pair-sum pileup.

Notes
-----
Pass one workspace to repeated apply_pileup calls to retain allocated
capacity. A workspace must not be used concurrently by multiple callers.)ibdoc").def(py::init<>());
    m.def(
        "apply_pileup_into",
        [](const Array &input, Array output, double real, double live, double tau,
           processing::PileupWorkspace &w) {
            auto out = writable(output);
            py::gil_scoped_release release;
            processing::apply_pileup(data(input), out, real, live, tau, w);
        },
        py::arg("input").noconvert(), py::arg("output").noconvert(), py::arg("real_time"),
        py::arg("live_time"), py::arg("tau"), py::arg("workspace"), R"ibdoc(Apply pair-sum pileup into caller-owned float64 storage.

Input is a nonempty contiguous float64 array of length N; output has 2*N-1
entries. Real/live times are positive seconds with live <= real; tau is
nonnegative seconds. Supply one PileupWorkspace per serialized caller to retain
FFT scratch. Input/output overlap and invalid values are rejected. Returns None.)ibdoc");
    m.def(
        "remove_pileup_into",
        [](const Array &input, Array output, double real, double live, double tau) {
            auto out = writable(output);
            py::gil_scoped_release release;
            processing::remove_pileup(data(input), out, real, live, tau);
        },
        py::arg("input").noconvert(), py::arg("output").noconvert(), py::arg("real_time"),
        py::arg("live_time"), py::arg("tau"), R"ibdoc(Invert complete pair-sum pileup into caller-owned float64 storage.

A complete odd-length 2*N-1 input reconstructs N output bins. Arrays must be
contiguous float64 and nonoverlapping. Times/tau must match the forward model.
Truncated or incompatible spectra are rejected. Returns None.)ibdoc");
    py::class_<simulation::Failure>(m, "Failure", R"ibdoc(Native backend failure diagnostic.

Attributes
----------
measurement : str
    Affected label, or empty for an experiment-wide failure.
message : str
    Human-readable diagnostic.)ibdoc")
        .def(py::init<>()) FIELD(simulation::Failure, measurement)
            FIELD(simulation::Failure, message);
    py::class_<simulation::Result>(m, "SimulationResult", R"ibdoc(Native owning spectra and optional failure.

Attributes
----------
spectra : list[Spectrum]
    Labelled spectra.
failure : Failure or None
    Optional failure; successful results have no diagnostic.)ibdoc")
        .def(py::init<>()) FIELD(simulation::Result, spectra) FIELD(simulation::Result, failure);
    // Copy each Python spectrum into its final owning native buffer once. Going
    // through individual bound Spectrum objects and the vector setter copies twice.
    m.def("_simulation_result", [](const py::list &spectra, py::object failure) {
        simulation::Result output;
        output.spectra.resize(spectra.size());
        for (py::ssize_t i = 0; i < spectra.size(); ++i) {
            auto item = spectra[i];
            auto &spectrum = output.spectra[static_cast<std::size_t>(i)];
            spectrum.measurement = item.attr("measurement").cast<std::string>();
            auto counts = item.attr("counts").cast<Array>();
            if (counts.ndim() != 1) throw std::invalid_argument("spectrum counts must be one-dimensional");
            const auto values = data(counts);
            spectrum.counts.assign(values.begin(), values.end());
        }
        if (!failure.is_none()) output.failure = simulation::Failure{
            failure.attr("measurement").cast<std::string>(), failure.attr("message").cast<std::string>()};
        return output;
    });
    py::class_<simulation::Simulator>(m, "Simulator", R"ibdoc(Native simulator interface with cooperative cancellation.

Notes
-----
Use the public Python Simulator facade to define Python backends. Native
simulate() returns an owning result; simulate_batch() preserves input order.
Calls on an instance must be serialized. Each backend owns an atomic stop flag.)ibdoc")
        .def("request_stop", &simulation::Simulator::request_stop, R"ibdoc(Set the cooperative cancellation flag; does not forcibly stop external work.)ibdoc")
        .def("reset_stop", &simulation::Simulator::reset_stop, R"ibdoc(Clear the cooperative cancellation flag before reusing an active backend.)ibdoc")
        .def("stopped", &simulation::Simulator::stopped, R"ibdoc(Return the current atomic cooperative cancellation flag.)ibdoc")
        .def("provenance", &simulation::Simulator::provenance, R"ibdoc(Return the backend identity recorded in generated datasets.)ibdoc")
        .def("simulate",
             [](simulation::Simulator &s, const core::Experiment &x) {
                 simulation::Result r;
                 {
                     py::gil_scoped_release release;
                     s.simulate(x, r);
                 }
                 return r;
             }, R"ibdoc(Simulate one native Experiment and return an owning SimulationResult.
Inspect failure as well as handling raised errors. Calls must be serialized.)ibdoc")
        .def("simulate_batch",
             [](simulation::Simulator &s, const std::vector<core::Experiment> &x) {
                 py::gil_scoped_release release;
                 return s.simulate_batch(x);
             }, R"ibdoc(Simulate native experiments in one backend batch, returning ordered results.
Cancellation may shorten the result sequence. Calls must be serialized.)ibdoc");
    py::class_<PythonSimulator, simulation::Simulator>(m, "_PythonSimulator")
        .def(py::init<py::function, py::function, py::function>());
    py::class_<simulation::DummySimulator, simulation::Simulator>(m, "DummySimulator", R"ibdoc(Create a deterministic synthetic native simulation backend.

Parameters
----------
channels : int, default=5000
    Positive channel count.)ibdoc")
        .def(py::init<std::size_t>(), py::arg("channels") = 5000);
    py::class_<simulation::SimnraMethod>(m, "SimnraMethod", R"ibdoc(Native SIMNRA reference mapping.

Attributes
----------
measurement : str
    Measurement label.
reference : pathlib.Path
    Reference project path.
particle : str
    Expected particle identifier.)ibdoc")
        .def(py::init<>()) FIELD(simulation::SimnraMethod, measurement)
            FIELD(simulation::SimnraMethod, reference) FIELD(simulation::SimnraMethod, particle);
    py::class_<simulation::SimnraConfig>(m, "SimnraConfig", R"ibdoc(Configure native Windows COM simulation workers.

Attributes
----------
methods : list[SimnraMethod]
    One reference per measurement.
workers : int, default=1
    Worker count.
multithreaded_apartment : bool, default=True
    Use MTA COM initialization.
fast_calculation : bool, default=False
    Select CalculateSpectrumFast rather than the standard algorithm.
reuse_target_topology : bool, default=True
    Reuse compatible targets and retain cross-section choices.)ibdoc")
        .def(py::init<>()) FIELD(simulation::SimnraConfig, methods)
            FIELD(simulation::SimnraConfig, workers)
                FIELD(simulation::SimnraConfig, multithreaded_apartment)
                    FIELD(simulation::SimnraConfig, fast_calculation)
                        FIELD(simulation::SimnraConfig, reuse_target_topology);
    py::class_<simulation::SimnraSimulator, simulation::Simulator>(m, "SimnraSimulator", R"ibdoc(Create a native backend with persistent SIMNRA workers.

Parameters
----------
config : SimnraConfig
    Reference mappings and worker configuration.

Notes
-----
Requires a SIMNRA-enabled Windows build and installed runtime. close()
releases workers. Interactive/already-owned applications are skipped using OLEUser.)ibdoc")
        .def(py::init<simulation::SimnraConfig>())
        .def("close", &simulation::SimnraSimulator::close,
             py::call_guard<py::gil_scoped_release>(), R"ibdoc(Request cancellation and release COM workers. Safe to call repeatedly.)ibdoc");
    py::class_<datasets::Parameter>(m, "Parameter", R"ibdoc(Declare one named native physical parameter column.

Parameters
----------
name : str
    Unique column name.
target : ParameterTarget
    Physical target location.
unit : str
    Descriptive unit label; values use the target physical convention.)ibdoc")
        .def(py::init<std::string, core::ParameterTarget, std::string>())
            FIELD(datasets::Parameter, name) FIELD(datasets::Parameter, target)
                FIELD(datasets::Parameter, unit);
    py::class_<datasets::Metadata>(m, "DatasetMetadata", R"ibdoc(Native dataset reference, columns, provenance and outcome counts.

Attributes
----------
reference : Experiment
    Physical reference including geometry.
parameters : list[Parameter]
    Stored column order.
provenance : str
    Generation provenance.
requested : int
    Requested row count.
accepted : int
    Successful stored count.
failed : int
    Stored failure count.
discarded : int
    Omitted failure count.)ibdoc")
        .def(py::init<>()) FIELD(datasets::Metadata, reference)
            FIELD(datasets::Metadata, parameters) FIELD(datasets::Metadata, provenance)
                FIELD(datasets::Metadata, requested) FIELD(datasets::Metadata, accepted)
                    FIELD(datasets::Metadata, failed) FIELD(datasets::Metadata, discarded);
    py::class_<datasets::Record>(m, "Record", R"ibdoc(Native dataset record.

Attributes
----------
sample_index : int
    Original generation row index.
sample_id : str
    Stored identifier.
parameters : list[float]
    Physical row values.
result : SimulationResult
    Stored spectra or failure.)ibdoc")
        .def(py::init<>()) FIELD(datasets::Record, sample_index) FIELD(datasets::Record, sample_id)
            FIELD(datasets::Record, parameters) FIELD(datasets::Record, result);
    py::class_<datasets::Reader>(m, "DatasetReader", R"ibdoc(Stream records from a completed checksummed dataset.

Parameters
----------
path : pathlib.Path
    Directory containing manifest.ibd.

Notes
-----
Each reader owns its cursor. Use separate readers for independent iteration.
The public Dataset facade provides Python records and explicit matrix helpers.)ibdoc")
        .def(py::init<const std::filesystem::path &>())
        .def_property_readonly("metadata", &datasets::Reader::metadata,
                               py::return_value_policy::copy, R"ibdoc(Return the physical reference, column order and completed dataset counters.)ibdoc")
        .def("rewind", &datasets::Reader::rewind)
        .def("maximum_spectrum_lengths", &datasets::Reader::maximum_spectrum_lengths,
             py::call_guard<py::gil_scoped_release>(), R"ibdoc(Scan successful native records for maximum per-label channel lengths.
Uses reusable streaming buffers without allocating full spectral matrices.)ibdoc")
        .def(
            "__iter__", [](datasets::Reader &r) -> datasets::Reader & { return r; },
            py::return_value_policy::reference_internal, R"ibdoc(Create an iterator for native stored records. Use independent readers for
independent cursor state.)ibdoc")
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
    py::class_<datasets::Writer>(m, "DatasetWriter", R"ibdoc(Publish a new sharded native dataset.

Parameters
----------
path : pathlib.Path
    New destination directory.
metadata : DatasetMetadata
    Validated reference and requested outcome metadata.
records_per_shard : int
    Buffered records per shard.

Notes
-----
append() writes stored records, discard() counts omitted inputs, and finish()
publishes the final manifest only when counters are complete. Serial use only.)ibdoc")
        .def(py::init<const std::filesystem::path &, datasets::Metadata, std::size_t>(),
             py::arg("path"), py::arg("metadata"), py::arg("records_per_shard") = 64)
        .def("append", &datasets::Writer::append, R"ibdoc(Append one stored native Record, validating identity and outcome.)ibdoc")
        .def("discard", &datasets::Writer::discard, R"ibdoc(Count a failed input omitted under the discard policy.)ibdoc")
        .def("finish", &datasets::Writer::finish, R"ibdoc(Validate complete outcome counters and atomically publish the final manifest.
An incomplete dataset cannot be committed.)ibdoc");
    py::enum_<generation::FailurePolicy>(m, "FailurePolicy", R"ibdoc(Select native generation behavior for failed inputs.

Notes
-----
STOP aborts with incomplete output; RECORD stores failures; DISCARD omits
failed inputs while retaining source indices for successful records.)ibdoc")
        .value("STOP", generation::FailurePolicy::Stop)
        .value("RECORD", generation::FailurePolicy::Record)
        .value("DISCARD", generation::FailurePolicy::Discard);
    py::class_<generation::GenerationPlan>(m, "GenerationPlan", R"ibdoc(Native reference experiment and physical parameter columns.

Attributes
----------
reference : Experiment
    Reference values for unvaried fields.
parameters : list[Parameter]
    Ordered input columns.)ibdoc")
        .def(py::init<>()) FIELD(generation::GenerationPlan, reference)
            FIELD(generation::GenerationPlan, parameters)
        .def("validate", &generation::GenerationPlan::validate, R"ibdoc(Validate reference experiment and unique ordered physical parameter columns.)ibdoc");
    py::class_<generation::Summary>(m, "GenerationSummary", R"ibdoc(Native counters from a generation invocation.

Attributes
----------
requested : int
    Requested input count.
accepted : int
    Successful stored records.
failed : int
    Stored failures.
discarded : int
    Omitted failures.
cancelled : bool
    Whether cooperative cancellation stopped generation.)ibdoc")
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
        py::arg("records_per_shard") = 64, py::arg("batch_size") = 256, R"ibdoc(Run native sharded dataset generation from a physical row matrix.

Rows must be a contiguous float64 matrix with one column per plan parameter;
this low-level binding does not convert incompatible arrays. Simulator is a
native backend or the adapter returned by a Python Simulator. Progress receives
completed and total counts. Returns a native GenerationSummary. Use the public
generate facade for option conversion, contextual errors and an output path.)ibdoc");
    py::enum_<model::Direction>(m, "Direction", R"ibdoc(Identify a model scientific mapping.

Notes
-----
FORWARD maps sample parameters to spectra; INVERSE maps spectra to an
elemental depth profile.)ibdoc")
        .value("FORWARD", model::Direction::Forward)
        .value("INVERSE", model::Direction::Inverse);
    py::enum_<model::SpectrumState>(m, "SpectrumState", R"ibdoc(Declare the physical pileup state of a reference spectrum.

Notes
-----
BARE denotes pileup-free counts. PILED_UP denotes counts with pileup.
Metadata validation uses this state to reject inconsistent preparation.)ibdoc")
        .value("BARE", model::SpectrumState::Bare).value("PILED_UP", model::SpectrumState::PiledUp);
    py::class_<model::SpectrumSpec>(m, "SpectrumSpec", R"ibdoc(Declare one reference-condition graph spectrum.

Parameters
----------
measurement : str
    Label matching a reference measurement.
channels : int
    Positive reference channel count.
state : SpectrumState, default=SpectrumState.BARE
    Physical pileup state.
unit : str, default="counts"
    Integrated-count unit.

Notes
-----
Energy calibration and exposure come from the matching reference setup.
Physical output preparation may change the final grid, length and pileup state.)ibdoc")
        .def(py::init<std::string, std::size_t, model::SpectrumState, std::string>(),
             py::arg("measurement"), py::arg("channels"), py::arg("state") = model::SpectrumState::Bare,
             py::arg("unit") = "counts") FIELD(model::SpectrumSpec, measurement)
            FIELD(model::SpectrumSpec, channels) FIELD(model::SpectrumSpec, state) FIELD(model::SpectrumSpec, unit);
    py::class_<model::SpectraSchema>(m, "SpectraSchema", R"ibdoc(Pack labelled spectra consecutively along a tensor feature axis.

Parameters
----------
spectra : sequence of SpectrumSpec, optional
    Ordered unique measurement descriptors.

Notes
-----
width() returns total channels. offset(label) returns the first feature index
for a measurement, independently of caller spectrum ordering.)ibdoc")
        .def(py::init<>())
        .def(py::init<std::vector<model::SpectrumSpec>>()) FIELD(model::SpectraSchema, spectra)
        .def("width", &model::SpectraSchema::width, R"ibdoc(Return the summed channel count of all packed reference spectra.)ibdoc")
        .def("offset", &model::SpectraSchema::offset, R"ibdoc(Return the first packed feature index for a measurement label.
Raises ValueError for an unknown label.)ibdoc");
    py::class_<model::Log1p>(m, "Log1p", R"ibdoc(Describe the fitted numerical transform log(1 + x).

Notes
-----
Requires nonnegative inputs. Inverse direction uses expm1. This is a tensor
transform descriptor; it does not perform physical detector corrections.)ibdoc").def(py::init<>());
    py::class_<model::MinMaxScale>(m, "MinMaxScale", R"ibdoc(Describe fitted per-feature min-max scaling.

Parameters
----------
minimum : sequence of float
    Fitted input minima; one scalar or one value per feature.
maximum : sequence of float
    Fitted maxima, larger than corresponding minima.
low : float, default=0.0
    Destination lower bound.
high : float, default=1.0
    Destination upper bound.
clip : bool, default=False
    Clamp transformed values; clipping prevents exact inversion.

Notes
-----
Maps x to low + (x-minimum)*(high-low)/(maximum-minimum). Coefficients are
recorded from training and never refitted during inference.)ibdoc")
        .def(py::init<core::Counts, core::Counts, double, double, bool>(), py::arg("minimum"),
             py::arg("maximum"), py::arg("low") = 0., py::arg("high") = 1., py::arg("clip") = false)
            FIELD(model::MinMaxScale, minimum) FIELD(model::MinMaxScale, maximum)
                FIELD(model::MinMaxScale, low) FIELD(model::MinMaxScale, high)
                    FIELD(model::MinMaxScale, clip);
    py::class_<model::StandardScale>(m, "StandardScale", R"ibdoc(Describe fitted standardization (x - mean) / deviation.

Parameters
----------
mean : sequence of float
    Fitted means; scalar or one per feature.
deviation : sequence of float
    Positive finite fitted deviations with compatible broadcasting.

Notes
-----
Inverse direction restores x*deviation + mean. Layer schemas broadcast
coefficients independently over each nonempty layer.)ibdoc")
        .def(py::init<core::Counts, core::Counts>()) FIELD(model::StandardScale, mean)
            FIELD(model::StandardScale, deviation);
    py::class_<model::MultiplyScale>(m, "MultiplyScale", R"ibdoc(Describe numerical multiplication by fitted factors.

Parameters
----------
factors : sequence of float
    One broadcast factor or one per feature.

Notes
-----
Inverse direction divides by factors and requires nonzero factors. This is
a numerical transform, distinct from physical particlesSr exposure scaling.)ibdoc")
        .def(py::init<core::Counts>()) FIELD(model::MultiplyScale, factors);
    py::class_<model::CustomTransform>(m, "CustomTransform", R"ibdoc(Describe an application-registered numerical transform.

Attributes
----------
identifier : str
    Kernel registration identifier.
version : int, default=1
    Kernel contract version.
parameters : dict[str, list[float]]
    Named numeric coefficient arrays.

Notes
-----
Packages contain descriptors, not executable code. A runtime transform
requires a matching C++ kernel registered by the consuming application.)ibdoc")
        .def(py::init<>()) FIELD(model::CustomTransform, identifier)
            FIELD(model::CustomTransform, version) FIELD(model::CustomTransform, parameters);
    py::enum_<model::TransformDirection>(m, "TransformDirection", R"ibdoc(Choose the direction of stored numerical training operations.

Notes
-----
FORWARD applies steps in stored order. INVERSE applies inverse operations
in reverse order. Non-invertible descriptors are rejected in inverse pipelines.)ibdoc")
        .value("FORWARD", model::TransformDirection::Forward).value("INVERSE", model::TransformDirection::Inverse);
    py::enum_<model::TransformLocation>(m, "TransformLocation", R"ibdoc(Declare where a numerical transform executes.

Notes
-----
RUNTIME executes outside ONNX. GRAPH declares operations already embedded
in ONNX; the runtime skips them to avoid double processing.)ibdoc")
        .value("RUNTIME", model::TransformLocation::Runtime).value("GRAPH", model::TransformLocation::Graph);
    py::class_<model::TransformPipeline>(m, "TransformPipeline", R"ibdoc(Describe fitted numerical operations and their execution boundary.

Parameters
----------
steps : sequence of tensor transforms, optional
    Ordered fitted operations.
direction : TransformDirection, default=FORWARD
    Apply or invert the stored operations.
location : TransformLocation, default=RUNTIME
    Execute in native processing or declare graph-embedded operations.

Notes
-----
Numerical transforms preserve feature width. Physical spectrum operations
are declared separately through SpectrumProcessing.)ibdoc")
        .def(py::init<model::TensorPipeline, model::TransformDirection, model::TransformLocation>(),
            py::arg("steps") = model::TensorPipeline{}, py::arg("direction") = model::TransformDirection::Forward,
            py::arg("location") = model::TransformLocation::Runtime)
        FIELD(model::TransformPipeline, steps) FIELD(model::TransformPipeline, direction) FIELD(model::TransformPipeline, location);
    py::class_<model::SpectrumTransform>(m, "SpectrumTransform", R"ibdoc(Associate one numerical transform pipeline with a detector label.

Parameters
----------
measurement : str
    Measurement identifier.
transform : TransformPipeline
    Fitted numerical operations for this spectrum.)ibdoc")
        .def(py::init<std::string, model::TransformPipeline>())
        FIELD(model::SpectrumTransform, measurement) FIELD(model::SpectrumTransform, transform);
    py::class_<model::ExposureScale>(m, "ExposureScale", R"ibdoc(Describe physical scaling by the ratio of particlesSr exposures.

Notes
-----
Requires positive reference and requested exposures. Forward preparation maps
reference to requested exposure; inverse preparation maps measured to reference.
Ordering relative to nonlinear pileup matters.)ibdoc").def(py::init<>());
    py::class_<model::Rebin>(m, "Rebin", R"ibdoc(Describe physical integrated-count rebinning between detector grids.

Parameters
----------
channels : int, default=0
    Destination channels; zero selects the measurement schema channel count.
require_full_coverage : bool, default=True
    Require the source energy range to cover the destination grid.

Notes
-----
Forward preparation targets the requested calibration; inverse preparation
targets reference calibration. No angle change is represented by this operation.)ibdoc")
        .def(py::init<std::size_t, bool>(), py::arg("channels") = 0,
             py::arg("require_full_coverage") = true) FIELD(model::Rebin, channels)
            FIELD(model::Rebin, require_full_coverage);
    py::enum_<model::AcquisitionSource>(m, "AcquisitionSource", R"ibdoc(Select the times used by a physical pileup operation.

Notes
-----
REQUESTED uses inference-request real/live times. REFERENCE uses package
reference times. This does not change the exposure-scaling policy.)ibdoc")
        .value("REQUESTED", model::AcquisitionSource::Requested)
        .value("REFERENCE", model::AcquisitionSource::Reference);
    py::class_<model::Pileup>(m, "Pileup", R"ibdoc(Describe pair-sum pileup applied to a bare reference spectrum.

Parameters
----------
tau_seconds : float
    Nonnegative resolving time in seconds.
acquisition : AcquisitionSource, default=REQUESTED
    Source of positive real/live times.

Notes
-----
Expands N channels to the complete 2*N-1 support and changes physical state
to piled-up. Execution follows the declared per-detector operation order.)ibdoc")
        .def(py::init<double, model::AcquisitionSource>(), py::arg("tau_seconds"),
             py::arg("acquisition") = model::AcquisitionSource::Requested)
            FIELD(model::Pileup, tau_seconds) FIELD(model::Pileup, acquisition);
    py::class_<model::RemovePileup>(m, "RemovePileup", R"ibdoc(Describe inversion of a complete pair-sum piled-up spectrum.

Parameters
----------
tau_seconds : float
    Resolving time in seconds, matching the forward pileup model.
relative_tolerance : float, default=1e-7
    Forward-reconstruction consistency tolerance.
acquisition : AcquisitionSource, default=REQUESTED
    Source of real/live acquisition times.

Notes
-----
Requires complete 2*N-1 support and reconstructs N bare bins. Truncated or
incompatible spectra are rejected; arbitrary pileup deconvolution is unsupported.)ibdoc")
        .def(py::init<double, double, model::AcquisitionSource>(), py::arg("tau_seconds"),
             py::arg("relative_tolerance") = 1e-7,
             py::arg("acquisition") = model::AcquisitionSource::Requested)
            FIELD(model::RemovePileup, tau_seconds) FIELD(model::RemovePileup, relative_tolerance)
                FIELD(model::RemovePileup, acquisition);
    py::class_<model::SpectrumProcessing>(m, "SpectrumProcessing", R"ibdoc(Associate ordered physical spectrum operations with one measurement.

Parameters
----------
measurement : str
    Detector measurement label.
steps : sequence of spectrum operations
    Ordered exposure, rebin and pileup operations.

Notes
-----
Different detectors can use different pipelines. Operations are not
automatically reordered; nonlinear corrections make order scientifically relevant.)ibdoc")
        .def(py::init<std::string, std::vector<model::SpectrumOperation>>())
            FIELD(model::SpectrumProcessing, measurement) FIELD(model::SpectrumProcessing, steps);
    py::class_<model::EdpSchema>(m, "EdpSchema", R"ibdoc(Declare a fixed-maximum-layer elemental depth-profile output.

Parameters
----------
elements : sequence of str
    Ordered element symbols.
layers : int
    Positive fixed maximum number of output layers.
uncertainty : bool, default=False
    Pack means, presence probabilities and posterior standard deviations.
unit : str, default="1e15 atoms/cm2"
    Physical elemental areal-density unit.

Notes
-----
Unlike a variable-layer forward schema, this declares a finite output layer
count. Values are ordered surface to depth, then by element.)ibdoc")
        .def(py::init<std::vector<std::string>, std::size_t, bool, std::string>(),
             py::arg("elements"), py::arg("layers"), py::arg("uncertainty") = false,
             py::arg("unit") = "1e15 atoms/cm2")
        FIELD(model::EdpSchema, elements) FIELD(model::EdpSchema, layers)
        FIELD(model::EdpSchema, uncertainty) FIELD(model::EdpSchema, unit);
    py::class_<model::ParameterInputSchema>(m, "ParameterInputSchema", R"ibdoc(Native fixed sample and physical parameter packing.

Attributes
----------
reference_sample : Sample
    Reference sample for unvaried properties.
parameters : list[Parameter]
    Ordered physical graph columns.

Notes
-----
The public Python metadata facade offers dataclass construction and field
context for conversion errors. Complete validation occurs at metadata/package boundaries.)ibdoc")
        .def(py::init<>()) FIELD(model::ParameterInputSchema, reference_sample)
            FIELD(model::ParameterInputSchema, parameters);
    py::class_<model::ElementComposition>(m, "ElementComposition", R"ibdoc(Native fixed isotope assumption for one element.

Attributes
----------
element : str
    Supported element symbol.
isotopes : list[Isotope]
    Fixed mixture; empty resolves to natural composition.

Notes
-----
The public Python metadata facade offers dataclass construction and field
context for conversion errors. Complete validation occurs at metadata/package boundaries.)ibdoc")
        .def(py::init<>()) FIELD(model::ElementComposition, element) FIELD(model::ElementComposition, isotopes);
    py::class_<model::LayerInputSchema>(m, "LayerInputSchema", R"ibdoc(Native unbounded layer-sequence input packing.

Attributes
----------
elements : list[str]
    Ordered elemental AD features in 1e15 atoms/cm2.
isotope_compositions : list[ElementComposition]
    Fixed per-element mixtures; omitted entries resolve to natural composition.

Notes
-----
The public Python metadata facade offers dataclass construction and field
context for conversion errors. Complete validation occurs at metadata/package boundaries.)ibdoc")
        .def(py::init<>()) FIELD(model::LayerInputSchema, elements) FIELD(model::LayerInputSchema, isotope_compositions);
    py::class_<model::ForwardModelMetadata>(m, "ForwardModelMetadata", R"ibdoc(Native forward scientific model contract.

Attributes
----------
reference_setup : ExperimentalSetup
    Reference measurements including geometry.
input_preparation : ParameterInputSchema or LayerInputSchema
    Physical sample packing.
output_spectra : SpectraSchema
    Packed reference spectra.
input_transform : TransformPipeline
    Fitted graph input transforms.
output_transform : list[SpectrumTransform]
    Per-detector numerical output transforms.
output_preparation : list[SpectrumProcessing]
    Per-detector physical output corrections.

Notes
-----
The public Python metadata facade offers dataclass construction and field
context for conversion errors. Complete validation occurs at metadata/package boundaries.)ibdoc")
        .def(py::init<>()) FIELD(model::ForwardModelMetadata, reference_setup)
            FIELD(model::ForwardModelMetadata, input_preparation)
                FIELD(model::ForwardModelMetadata, output_spectra)
                        FIELD(model::ForwardModelMetadata, input_transform)
            FIELD(model::ForwardModelMetadata, output_transform) FIELD(model::ForwardModelMetadata, output_preparation);
    py::class_<model::InverseModelMetadata>(m, "InverseModelMetadata", R"ibdoc(Native inverse scientific model contract.

Attributes
----------
reference_setup : ExperimentalSetup
    Reference setup and geometry.
input_spectra : SpectraSchema
    Packed reference input spectra.
input_preparation : list[SpectrumProcessing]
    Measured-to-reference physical corrections.
input_transform : list[SpectrumTransform]
    Per-detector numerical input transforms.
output_transform : TransformPipeline
    Numerical EDP output processing.
output_preparation : EdpSchema
    Fixed-layer EDP layout and units.

Notes
-----
The public Python metadata facade offers dataclass construction and field
context for conversion errors. Complete validation occurs at metadata/package boundaries.)ibdoc")
        .def(py::init<>()) FIELD(model::InverseModelMetadata, reference_setup)
        FIELD(model::InverseModelMetadata, input_spectra) FIELD(model::InverseModelMetadata, input_preparation)
        FIELD(model::InverseModelMetadata, input_transform) FIELD(model::InverseModelMetadata, output_transform)
        FIELD(model::InverseModelMetadata, output_preparation);
    py::class_<model::ModelMetadata>(m, "ModelMetadata", R"ibdoc(Native directional scientific contract and graph identifiers.

Attributes
----------
model : ForwardModelMetadata or InverseModelMetadata
    Directional contract.
input_name : str, default="inputs"
    Single graph input name.
output_name : str, default="outputs"
    Single graph output name.
provenance : str
    User/model provenance.

Notes
-----
The public Python metadata facade offers dataclass construction and field
context for conversion errors. Complete validation occurs at metadata/package boundaries.)ibdoc")
        .def(py::init<>())
        .def(py::init<model::ForwardModelMetadata>())
        .def(py::init<model::InverseModelMetadata>()) FIELD(model::ModelMetadata, model)
        .def_property_readonly("direction", &model::ModelMetadata::direction, R"ibdoc(Return FORWARD or INVERSE from the active directional metadata.)ibdoc")
            FIELD(model::ModelMetadata, input_name) FIELD(model::ModelMetadata, output_name)
                FIELD(model::ModelMetadata, provenance)
        .def("validate", &model::ModelMetadata::validate, R"ibdoc(Validate the directional scientific contract without executing ONNX.
Checks schemas, mixtures, geometry, units and processing; raises ValueError.)ibdoc")
        .def("input_width", &model::ModelMetadata::input_width, R"ibdoc(Return physical-parameter, per-layer element or packed spectral width.)ibdoc")
        .def("input_rank", &model::ModelMetadata::input_rank, R"ibdoc(Return 3 for a layer sequence and 2 for other graph inputs.)ibdoc")
        .def("layer_input", &model::ModelMetadata::layer_input, R"ibdoc(Return True when this is a forward variable-layer input contract.)ibdoc")
        .def("output_width", &model::ModelMetadata::output_width, R"ibdoc(Return packed graph output width before physical spectrum preparation.)ibdoc");
    py::class_<model::Package>(m, "Package", R"ibdoc(Native owning ONNX graph and scientific metadata.

Attributes
----------
metadata : ModelMetadata
    Directional scientific contract.

Notes
-----
open() validates checksums and metadata without initializing ONNX.
from_onnx() associates standalone graph bytes with metadata; write() refuses
an existing destination. Independent immutable-file loads own separate storage.)ibdoc")
        .def_static("open", &model::Package::open, py::call_guard<py::gil_scoped_release>(), R"ibdoc(Read checksummed graph bytes and validate native scientific metadata.
Does not initialize ONNX. Invalid/unavailable files raise native exceptions.)ibdoc")
        .def_static("from_onnx", &model::Package::from_onnx, R"ibdoc(Read a standalone graph and associate validated native ModelMetadata.
Resolves default isotope assumptions explicitly; does not execute the graph.)ibdoc")
        .def("write", &model::Package::write, R"ibdoc(Publish a new checksummed format-1 package.
The parent directory must exist; an existing destination is rejected.)ibdoc") FIELD(model::Package, metadata);
    py::enum_<inference::ExecutionProvider>(m, "ExecutionProvider", R"ibdoc(Choose the native ONNX provider.

Notes
-----
CPU selects CPU execution. CUDA requires a GPU-enabled native build and
working compatible device/runtime libraries.)ibdoc")
        .value("CPU", inference::ExecutionProvider::Cpu)
        .value("CUDA", inference::ExecutionProvider::Cuda);
    py::class_<inference::InferenceOptions>(m, "InferenceOptions", R"ibdoc(Native per-session execution settings.

Attributes
----------
execution_provider : ExecutionProvider
    CPU or CUDA.
device_id : int
    Nonnegative CUDA device index.
intra_op_threads : int
    Positive operator-level thread count.
inter_op_threads : int
    Positive graph-level thread count.
enable_graph_optimizations : bool
    Enable graph optimizations.
allow_tf32 : bool
    Allow TensorFloat-32 CUDA operations.
apply_pileup : bool
    Enable declared forward pileup operations.

Notes
-----
Use the Python InferenceOptions facade for validated keyword construction.
These settings do not change the model scientific contract.)ibdoc")
        .def(py::init<>()) FIELD(inference::InferenceOptions, execution_provider)
            FIELD(inference::InferenceOptions, device_id) FIELD(inference::InferenceOptions, intra_op_threads)
                FIELD(inference::InferenceOptions, inter_op_threads) FIELD(inference::InferenceOptions, enable_graph_optimizations)
                    FIELD(inference::InferenceOptions, allow_tf32) FIELD(inference::InferenceOptions, apply_pileup)
        .def("validate", &inference::InferenceOptions::validate, R"ibdoc(Validate provider enumeration, nonnegative device and positive thread counts.
Does not initialize a device; raises ValueError for invalid scalar options.)ibdoc");
    m.def("available_execution_providers", &inference::available_execution_providers, R"ibdoc(Return provider names exposed by this native runtime.

Returns a list containing supported cpu/cuda names, or an empty list when
ONNX support is disabled. Enumeration does not guarantee device initialization.)ibdoc");
    py::enum_<inference::PredictionDomain>(m, "PredictionDomain", R"ibdoc(Choose requested-measurement or reference-condition inference.

Notes
-----
MEASUREMENT executes physical preparation. REFERENCE skips physical
preparation while retaining numerical training transforms and output decoding.)ibdoc")
        .value("MEASUREMENT", inference::PredictionDomain::Measurement)
        .value("REFERENCE", inference::PredictionDomain::Reference);
    py::class_<inference::ForwardModel>(m, "ForwardModel", R"ibdoc(Create a native forward ONNX session.

Parameters
----------
package : Package
    Owning forward package.
options : InferenceOptions, optional
    Provider, device and execution settings.

Notes
-----
Calls on one instance must be serialized. Native overloads support reusable
output buffers and direct elemental AD. The Python ForwardModel facade provides
labelled results and input conversion. Use one instance per concurrent worker.)ibdoc")
        .def(py::init<model::Package, inference::InferenceOptions>(), py::arg("package"),
             py::arg("options") = inference::InferenceOptions{}, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("options", &inference::ForwardModel::options, py::return_value_policy::copy, R"ibdoc(Return the resolved execution settings for this session.)ibdoc")
        .def("predict",
             py::overload_cast<const core::Experiment &, inference::PredictionDomain>(&inference::ForwardModel::predict),
             py::arg("experiment"), py::arg("domain") = inference::PredictionDomain::Measurement,
             py::call_guard<py::gil_scoped_release>(), R"ibdoc(Predict requested-condition spectra for one native Experiment.
Applies packing, fitted transforms and physical detector preparation.)ibdoc")
        .def("predict_reference", py::overload_cast<const core::Experiment &>(&inference::ForwardModel::predict_reference),
             py::call_guard<py::gil_scoped_release>(), R"ibdoc(Predict physical reference-condition spectra for one native Experiment.
Skips physical output preparation while retaining numerical transforms.)ibdoc")
        .def("predict_areal_density", [](inference::ForwardModel &model,
             py::array_t<float, py::array::c_style> values, const core::ExperimentalSetup *setup, inference::PredictionDomain domain) {
            if (values.ndim() != 2 || static_cast<std::size_t>(values.shape(1)) != model.metadata().input_width())
                throw std::invalid_argument("AD must have shape [layers,elements]");
            py::gil_scoped_release release;
            return model.predict_areal_density({values.data(), static_cast<std::size_t>(values.size())},
                                               static_cast<std::size_t>(values.shape(0)), setup, domain);
        }, py::arg("densities").noconvert(), py::arg("setup") = nullptr,
           py::arg("domain") = inference::PredictionDomain::Measurement, R"ibdoc(Predict from contiguous float32 elemental AD shaped (layers,elements).
Features use 1e15 atoms/cm2 and fixed package isotope assumptions. Optional
setup must satisfy the model contract. Returns an owning native result.)ibdoc")
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
        }, py::arg("densities").noconvert(), py::arg("setup") = nullptr, py::arg("out") = py::none(), py::arg("domain") = inference::PredictionDomain::Measurement, R"ibdoc(Predict one float32 AD population shaped (batch,layers,elements).
Optional out is writable contiguous float64 (batch,total_output_channels),
nonoverlapping with input. Returned per-detector arrays view output storage;
reusing out changes previous views. Domain selects physical or reference output.)ibdoc")
        .def("predict_batch",
             [](inference::ForwardModel &model, const std::vector<core::Experiment> &inputs, inference::PredictionDomain domain) {
                 py::gil_scoped_release release;
                 return model.predict_batch(inputs, domain);
             }, py::arg("inputs"), py::arg("domain") = inference::PredictionDomain::Measurement, R"ibdoc(Predict an ordered native experiment population in one ONNX call.
Layer-input graphs zero-pad differing layer counts. Calls must be serialized.)ibdoc");
    py::class_<inference::InverseInput>(m, "InverseInput", R"ibdoc(Native measured spectra and acquisition setup.

Attributes
----------
setup : ExperimentalSetup
    Requested measurement conditions.
spectra : list[Spectrum]
    Labelled count spectra.
pileup_already_removed : bool
    Skip the first applicable requested-time inverse pileup step.)ibdoc")
        .def(py::init<>()) FIELD(inference::InverseInput, setup)
            FIELD(inference::InverseInput, spectra)
                FIELD(inference::InverseInput, pileup_already_removed);
    py::class_<inference::Prediction>(m, "Prediction", R"ibdoc(Native inverse EDP values and graph uncertainty.

Attributes
----------
elements : list[str]
    Element order.
layers : int
    Fixed output layer count.
unit : str
    Physical AD units.
uncertainty_predicted : bool
    Whether the graph supplies uncertainty.

Notes
-----
values_array, presence_probability_array and posterior_std_array are
read-only (layers, elements) NumPy views retaining the native owner. Vector
fields are convenience conversions and can copy.)ibdoc")
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
    py::class_<inference::InverseModel>(m, "InverseModel", R"ibdoc(Create a native spectra-to-EDP ONNX session.

Parameters
----------
package : Package
    Owning inverse package.
options : InferenceOptions, optional
    Session execution controls.

Notes
-----
Input preparation, fitted transforms and physical EDP decoding follow metadata.
Use one instance per concurrent worker; calls on an instance must serialize.)ibdoc")
        .def(py::init<model::Package, inference::InferenceOptions>(), py::arg("package"),
             py::arg("options") = inference::InferenceOptions{}, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("options", &inference::InverseModel::options, py::return_value_policy::copy, R"ibdoc(Return this session execution settings.)ibdoc")
        .def("predict",
             py::overload_cast<const inference::InverseInput &>(&inference::InverseModel::predict),
             py::call_guard<py::gil_scoped_release>(), R"ibdoc(Prepare measured native input spectra and return a physical EDP prediction.)ibdoc")
        .def(
            "predict_batch",
            [](inference::InverseModel &model, const std::vector<inference::InverseInput> &inputs, inference::PredictionDomain domain) {
                py::gil_scoped_release release;
                return model.predict_batch(inputs, domain);
            }, py::arg("inputs"), py::arg("domain") = inference::PredictionDomain::Measurement, R"ibdoc(Prepare each native InverseInput, then execute one ordered ONNX batch.
Calls on the model must be serialized.)ibdoc")
        .def("predict_reference",
             [](inference::InverseModel &model, const std::vector<core::Spectrum> &spectra) {
                 inference::Prediction out;
                 {
                     py::gil_scoped_release release;
                     model.predict_reference(spectra, out);
                 }
                 return out;
             }, R"ibdoc(Predict from labelled reference-grid spectra, skipping physical preparation.
Fitted numerical transforms and physical EDP processing still run.)ibdoc");
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
        py::arg("package"), py::arg("input").noconvert(), py::arg("options") = inference::InferenceOptions{}, R"ibdoc(Execute the raw ONNX boundary for export verification.

Parameters
----------
package : Package
    Native package containing a standalone graph.
input : numpy.ndarray
    Contiguous float32 graph input with rank/feature dimensions matching metadata.

Returns
-------
numpy.ndarray
    Raw float32 graph output.

Notes
-----
Requires an ONNX-enabled build and uses a CPU session. This bypasses physical
packing, numerical transform pipelines and physical detector preparation. It is
an export/parity diagnostic, not the ordinary physical inference API.)ibdoc");
}
