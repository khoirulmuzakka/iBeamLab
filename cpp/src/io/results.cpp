#include <fstream>
#include <ibeamlab/io/results.h>
#include <iomanip>
#include <stdexcept>
namespace ibeamlab::io {
namespace {
std::string quote(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += '"';
        out += c;
    }
    return out + '"';
}
std::ofstream file(const std::filesystem::path &path) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17);
    return out;
}
} // namespace
void write_result(const std::filesystem::path &path, const fitting::Result &r,
                  std::string provenance) {
    if (!r.feasible() || r.parameter_names.size() != r.parameter_values.size())
        throw std::invalid_argument("cannot export infeasible/inconsistent result");
    if (!std::filesystem::create_directory(path))
        throw std::runtime_error("result directory already exists");
    auto sample = file(path / "sample.csv");
    sample << "layer,thickness,roughness,porosity_fraction,pore_diameter,element,fraction,mass_"
              "number,isotope_fraction\n";
    for (std::size_t i = 0; i < r.experiment.sample.layers.size(); ++i) {
        const auto &l = r.experiment.sample.layers[i];
        for (const auto &s : l.species) {
            auto prefix = [&] {
                sample << i << ',' << l.thickness << ',' << l.roughness << ',' << l.porosityFraction
                       << ',' << l.poreDiameter << ',' << quote(s.element) << ',' << s.concentration
                       << ',';
            };
            if (s.isotopes.empty()) {
                prefix();
                sample << ",\n";
            }
            for (const auto &iso : s.isotopes) {
                prefix();
                sample << iso.massNumber << ',' << iso.fraction << '\n';
            }
        }
    }
    auto params = file(path / "parameters.csv");
    params << "name,value\n";
    for (std::size_t i = 0; i < r.parameter_names.size(); ++i)
        params << quote(r.parameter_names[i]) << ',' << r.parameter_values[i] << '\n';
    auto spectra = file(path / "spectra.csv");
    spectra << "measurement,channel,counts\n";
    for (const auto &s : r.prediction.spectra)
        for (std::size_t j = 0; j < s.counts.size(); ++j)
            spectra << quote(s.measurement) << ',' << j << ',' << s.counts[j] << '\n';
    auto costs = file(path / "costs.csv");
    costs << "measurement,cost,channels\n";
    for (const auto &c : r.contributions)
        costs << quote(c.measurement) << ',' << c.cost << ',' << c.channels << '\n';
    auto setup = file(path / "setup.csv");
    setup
        << "measurement,particle,energy,spread,incidence,exit,scattering,calibration_linear,"
           "calibration_offset,calibration_quadratic,resolution,particles_sr,real_time,live_time\n";
    for (const auto &m : r.experiment.setup.measurements) {
        const auto &d = m.detector;
        setup << quote(m.label) << ',' << quote(m.beam.particle) << ',' << m.beam.energy << ','
              << m.beam.spread << ',' << d.geometry.incidenceAngle << ',' << d.geometry.exitAngle
              << ',' << d.geometry.scatteringAngle << ',' << d.calibrationLinear << ','
              << d.calibrationOffset << ',' << d.calibrationQuadratic << ',' << d.resolution << ','
              << m.acquisition.particlesSr << ',' << m.acquisition.realTime << ','
              << m.acquisition.liveTime << '\n';
    }
    auto coordinates = file(path / "coordinates.csv");
    coordinates << "index,value\n";
    for (std::size_t i = 0; i < r.coordinates.size(); ++i)
        coordinates << i << ',' << r.coordinates[i] << '\n';
    auto history = file(path / "history.csv");
    history << "evaluations,invalid,best_cost\n";
    for (const auto &v : r.history)
        history << v.evaluations << ',' << v.invalid << ',' << v.best_cost << '\n';
    auto elemental = file(path / "elemental.csv");
    elemental << "measurement,element,channel,counts\n";
    for (const auto &s : r.prediction.elemental_spectra)
        for (std::size_t i = 0; i < s.counts.size(); ++i)
            elemental << quote(s.measurement) << ',' << quote(s.element) << ',' << i << ','
                      << s.counts[i] << '\n';
    auto info = file(path / "provenance.txt");
    info << "cost=" << r.cost << "\nstatus=" << r.status << "\nmessage=" << r.message
         << "\nevaluations=" << r.evaluations << "\n"
         << provenance << '\n';
}
} // namespace ibeamlab::io
