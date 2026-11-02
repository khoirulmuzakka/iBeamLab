#include <algorithm>
#include <atomic>
#include <cmath>
#include <ibeamlab/fitting/objective.h>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
namespace ibeamlab::fitting {
namespace {
constexpr auto absent = std::size_t(-1);
// Bindings compile the closed core target variant to scalar field indices.
// Extending ParameterTarget requires extending this dispatch explicitly.
static_assert(std::variant_size_v<core::ParameterTarget> == 12);
struct Binding {
    std::size_t kind, first = 0, second = 0;
    double &ref(core::Experiment &e) const {
        if (kind >= 13) {
            auto &l = e.sample.layers[first];
            return kind == 13 ? l.roughness : kind == 14 ? l.porosityFraction : l.poreDiameter;
        }
        if (kind == 0)
            return e.sample.layers[first].thickness;
        if (kind == 1 || kind == 12)
            return e.sample.layers[first].species[second].concentration;
        auto &m = e.setup.measurements[first];
        switch (kind) {
        case 2:
            return m.beam.energy;
        case 3:
            return m.beam.spread;
        case 4:
            return m.detector.calibrationLinear;
        case 5:
            return m.detector.calibrationOffset;
        case 6:
            return m.detector.calibrationQuadratic;
        case 7:
            return m.detector.resolution;
        case 8:
            return m.acquisition.particlesSr;
        case 9:
            return m.detector.geometry.incidenceAngle;
        case 10:
            return m.detector.geometry.exitAngle;
        default:
            return m.detector.geometry.scatteringAngle;
        }
    }
};
Binding compile_binding(core::Experiment &e, const Target &t) {
    if (auto property = std::get_if<LayerPropertyTarget>(&t)) {
        (void)e.sample.layers.at(property->layer);
        if (property->property != LayerProperty::Roughness &&
            property->property != LayerProperty::PorosityFraction &&
            property->property != LayerProperty::PoreDiameter)
            throw std::invalid_argument("unknown layer property");
        return {13 + static_cast<std::size_t>(property->property), property->layer, 0};
    }
    if (auto ad = std::get_if<ElementArealDensity>(&t)) {
        auto b = compile_binding(
            e, Target{core::ParameterTarget{core::SpeciesConcentration{ad->layer, ad->element}}});
        b.kind = 12;
        return b;
    }
    const auto &v = std::get<core::ParameterTarget>(t);
    core::read_parameter(e, v);
    Binding b{v.index()};
    std::visit(
        [&](const auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, core::LayerThickness> ||
                          std::is_same_v<T, core::SpeciesConcentration>) {
                b.first = x.layer;
                if constexpr (std::is_same_v<T, core::SpeciesConcentration>) {
                    const auto &s = e.sample.layers[x.layer].species;
                    b.second = std::size_t(
                        std::find_if(s.begin(), s.end(),
                                     [&](const auto &z) { return z.element == x.element; }) -
                        s.begin());
                }
            } else
                b.first = std::size_t(
                    std::find_if(e.setup.measurements.begin(), e.setup.measurements.end(),
                                 [&](const auto &m) { return m.label == x.measurement; }) -
                    e.setup.measurements.begin());
        },
        v);
    return b;
}
bool physical(const core::Experiment &e) {
    for (const auto &l : e.sample.layers) {
        if (!std::isfinite(l.thickness) || l.thickness < 0 || !std::isfinite(l.roughness) ||
            l.roughness < 0 || !std::isfinite(l.porosityFraction) || l.porosityFraction < 0 ||
            l.porosityFraction > 1 || !std::isfinite(l.poreDiameter) || l.poreDiameter < 0)
            return false;
        double sum = 0;
        for (const auto &s : l.species) {
            if (!std::isfinite(s.concentration) || s.concentration < 0 || s.concentration > 1)
                return false;
            sum += s.concentration;
        }
        if (std::abs(sum - 1) > 1e-6)
            return false;
    }
    for (const auto &m : e.setup.measurements) {
        if (!std::isfinite(m.beam.energy) || !std::isfinite(m.beam.spread) ||
            !std::isfinite(m.detector.resolution) || !std::isfinite(m.acquisition.particlesSr) ||
            !std::isfinite(m.acquisition.realTime) || m.acquisition.realTime < 0 ||
            !std::isfinite(m.acquisition.liveTime) || m.acquisition.liveTime < 0 ||
            !std::isfinite(m.detector.calibrationLinear) ||
            !std::isfinite(m.detector.calibrationOffset) ||
            !std::isfinite(m.detector.calibrationQuadratic) || !(m.beam.energy > 0) ||
            m.beam.spread < 0 || m.detector.resolution < 0 || m.acquisition.particlesSr < 0 ||
            !(m.detector.calibrationLinear > 0))
            return false;
        try {
            m.detector.geometry.validate();
        } catch (const std::invalid_argument &) {
            return false;
        }
    }
    return true;
}
} // namespace
struct Objective::Impl {
    Problem problem;
    std::unique_ptr<Parameterization> parameterization;
    std::vector<std::size_t> profile_coordinates;
    core::Counts profile_values;
    core::Experiment macro_reference;
    std::shared_ptr<Predictor> predictor;
    core::Experiment reference;
    struct Scalar {
        std::vector<Binding> bindings;
        std::vector<double> ratios;
        std::size_t coordinate;
        double fixed;
    };
    struct Group {
        std::vector<Binding> bindings;
        std::vector<core::ParameterBounds> bounds;
        std::vector<std::size_t> coordinates;
        double total;
    };
    struct Score {
        std::vector<std::size_t> channels;
        core::Counts coefficients;
        std::size_t spectrum = absent;
    };
    std::vector<Scalar> scalars;
    std::vector<Group> groups;
    std::set<std::size_t> ad_layers;
    std::vector<core::ParameterBounds> bounds;
    std::vector<std::string> names;
    core::Counts initial;
    std::vector<std::size_t> measurement_channels;
    std::vector<Score> scores;
    bool fixed_denominator = false;
    std::vector<core::Experiment> inputs;
    std::vector<simulation::Result> outputs;
    std::vector<std::size_t> indices;
    Result best;
    std::atomic_bool stop = false;
    std::size_t evaluations = 0, invalid = 0;
    Impl(Problem input_problem, std::shared_ptr<Predictor> model)
        : problem(std::move(input_problem)), predictor(std::move(model)),
          reference{problem.profile ? problem.profile->reference_sample() : problem.initial_sample,
                    problem.data.setup} {
        if (!predictor)
            throw std::invalid_argument("fitting predictor is required");
        macro_reference = reference;
        if (problem.missing_observations_as_zero)
            for (auto &d : problem.data.spectra) {
                std::size_t size = d.counts.size();
                for (const auto &selection : problem.selections)
                    if (selection.measurement == d.measurement)
                        for (const auto &r : selection.regions)
                            size = std::max(size, r.stop);
                if (size > 1000000)
                    throw std::invalid_argument("selected channel count exceeds limit");
                d.counts.resize(size, 0);
                d.valid.clear();
                if (!d.variances.empty())
                    d.variances.resize(size, 0);
            }
        problem.data.validate();
        reference.validate();
        if (static_cast<int>(problem.loss) < 0 ||
            static_cast<int>(problem.loss) > static_cast<int>(Loss::Gaussian))
            throw std::invalid_argument("unknown fitting loss");
        for (const auto &m : reference.setup.measurements)
            measurement_channels.push_back(
                std::find_if(problem.data.spectra.begin(), problem.data.spectra.end(),
                             [&](const auto &s) { return s.measurement == m.label; })
                    ->counts.size());
        if (!std::isfinite(problem.data_floor) || problem.data_floor <= 0)
            throw std::invalid_argument("data_floor must be positive");
        std::set<std::tuple<std::size_t, std::size_t, std::size_t>> used;
        std::set<std::string> labels;
        auto claim = [&](Binding b) {
            if (problem.profile && (b.kind == 1 || b.kind == 12))
                throw std::invalid_argument(
                    "profile weights must be varied through coefficient parameters");
            if (!used.emplace(b.kind, b.first, b.second).second)
                throw std::invalid_argument("duplicate fitting target");
        };
        auto permission = [&](const Target &t) {
            if (auto v = std::get_if<core::ParameterTarget>(&t)) {
                if (!predictor->can_vary(*v))
                    throw std::invalid_argument("predictor cannot vary " + core::parameter_key(*v));
            } else if (std::holds_alternative<LayerPropertyTarget>(t)) {
                if (!predictor->can_vary_layer_properties())
                    throw std::invalid_argument("predictor cannot vary layer properties");
            } else {
                const auto &a = std::get<ElementArealDensity>(t);
                if (!predictor->can_vary(core::LayerThickness{a.layer}))
                    throw std::invalid_argument("predictor cannot vary areal density");
                for (const auto &s : reference.sample.layers.at(a.layer).species)
                    if (reference.sample.layers.at(a.layer).species.size() > 1 &&
                        !predictor->can_vary(core::SpeciesConcentration{a.layer, s.element}))
                        throw std::invalid_argument("predictor cannot vary composition");
            }
        };
        for (const auto &p0 : problem.parameters) {
            p0.bounds.validate();
            if (p0.name.empty() || !labels.insert(p0.name).second)
                throw std::invalid_argument("fitting parameter names must be unique and nonempty");
            Scalar s{{}, {}, absent, 0};
            auto add = [&](const Target &t) {
                auto b = compile_binding(reference, t);
                claim(b);
                double v = b.ref(reference);
                if (b.kind == 12) {
                    ad_layers.insert(b.first);
                    v *= reference.sample.layers[b.first].thickness;
                }
                if (!s.bindings.empty() &&
                    (b.kind != s.bindings.front().kind ||
                     std::abs(v - s.fixed) > 1e-10 * std::max(1., std::abs(v))))
                    throw std::invalid_argument(
                        "tied targets must have equal initial values and kinds");
                if (!p0.bounds.contains(v))
                    throw std::invalid_argument("initial value outside bounds: " + p0.name);
                s.fixed = v;
                s.bindings.push_back(b);
                s.ratios.push_back(1);
                if (p0.bounds.minimum < p0.bounds.maximum)
                    permission(t);
            };
            add(p0.target);
            for (const auto &t : p0.tied_targets)
                add(t);
            for (const auto &link : p0.linked_targets) {
                validate_ratio(link.ratio);
                auto b = compile_binding(reference, Target{link.target});
                claim(b);
                if (std::abs(b.ref(reference) - s.fixed * link.ratio) >
                    1e-10 * std::max(1., std::abs(b.ref(reference))))
                    throw std::invalid_argument("linked target initial value differs from ratio");
                if (p0.bounds.minimum < p0.bounds.maximum)
                    permission(Target{link.target});
                s.bindings.push_back(b);
                s.ratios.push_back(link.ratio);
            }
            if (p0.bounds.minimum < p0.bounds.maximum) {
                s.coordinate = bounds.size();
                bounds.push_back(p0.bounds);
                names.push_back(p0.name);
                initial.push_back(s.fixed);
            }
            best.parameter_names.push_back(p0.name);
            scalars.push_back(std::move(s));
        }
        for (const auto &c : problem.compositions) {
            if (c.elements.size() < 2 ||
                (!c.bounds.empty() && c.bounds.size() != c.elements.size()))
                throw std::invalid_argument(
                    "composition needs at least two elements and matching bounds");
            Group g;
            g.total = 0;
            for (std::size_t j = 0; j < c.elements.size(); ++j) {
                auto t = core::ParameterTarget{core::SpeciesConcentration{c.layer, c.elements[j]}};
                auto b = compile_binding(reference, Target{t});
                claim(b);
                auto r = c.bounds.empty() ? core::ParameterBounds{0, 1} : c.bounds[j];
                r.validate();
                if (r.minimum < 0 || r.maximum > 1 || !r.contains(b.ref(reference)))
                    throw std::invalid_argument("invalid initial composition bounds");
                if (r.minimum < r.maximum)
                    permission(Target{t});
                g.bindings.push_back(b);
                g.bounds.push_back(r);
                g.total += b.ref(reference);
                auto n =
                    (c.name_prefix.empty() ? "layer" + std::to_string(c.layer) : c.name_prefix) +
                    "." + c.elements[j];
                if (!labels.insert(n).second)
                    throw std::invalid_argument("duplicate physical parameter name");
                best.parameter_names.push_back(n);
            }
            double remaining = g.total;
            for (std::size_t j = 0; j < g.bindings.size(); ++j) {
                double low = 0, high = 0;
                for (std::size_t k = j + 1; k < g.bindings.size(); ++k) {
                    low += g.bounds[k].minimum;
                    high += g.bounds[k].maximum;
                }
                auto lo = std::max(g.bounds[j].minimum, remaining - high),
                     hi = std::min(g.bounds[j].maximum, remaining - low);
                std::size_t index = absent;
                if (j + 1 < g.bindings.size() && g.bounds[j].minimum < g.bounds[j].maximum) {
                    index = bounds.size();
                    bounds.push_back({0, 1});
                    names.push_back((c.name_prefix.empty() ? "layer" + std::to_string(c.layer)
                                                           : c.name_prefix) +
                                    ".stick." + c.elements[j]);
                    initial.push_back(hi > lo ? (g.bindings[j].ref(reference) - lo) / (hi - lo)
                                              : 0);
                }
                g.coordinates.push_back(index);
                remaining -= g.bindings[j].ref(reference);
            }
            groups.push_back(std::move(g));
        }
        for (auto layer : ad_layers)
            for (const auto &[kind, first, second] : used)
                if (first == layer && (kind == 0 || kind == 1))
                    throw std::invalid_argument(
                        "areal density conflicts with thickness/composition targets");
        if (!problem.profile && !problem.profile_parameters.empty())
            throw std::invalid_argument("profile parameters require a profile");
        if (problem.profile) {
            if (!problem.compositions.empty() || !ad_layers.empty())
                throw std::invalid_argument(
                    "profile weights cannot be combined with fraction/AD groups");
            for (const auto &p : problem.parameters) {
                auto b = compile_binding(reference, p.target);
                if (b.kind == 1)
                    throw std::invalid_argument(
                        "vary profile coefficients instead of macro fractions");
            }
            parameterization =
                std::make_unique<Parameterization>(*problem.profile, problem.profile_parameters);
            std::size_t sublayer = 0;
            for (const auto &l : problem.profile->layers) {
                for (std::size_t j = 0; j < l.sublayers; ++j, ++sublayer) {
                    if (!predictor->can_vary(core::LayerThickness{sublayer}))
                        throw std::invalid_argument("predictor cannot vary profile thickness");
                    for (const auto &sp : l.reference.species)
                        if (!predictor->can_vary(core::SpeciesConcentration{sublayer, sp.element}))
                            throw std::invalid_argument(
                                "predictor cannot vary profile composition");
                }
            }
            for (const auto &p : problem.profile_parameters) {
                if (p.name.empty() || !labels.insert(p.name).second)
                    throw std::invalid_argument("profile parameter names must be unique");
                auto v = problem.profile->layers[p.layer]
                             .functions[p.species]
                             .coefficients[p.coefficient];
                profile_coordinates.push_back(p.bounds.minimum < p.bounds.maximum ? bounds.size()
                                                                                  : absent);
                profile_values.push_back(v);
                best.parameter_names.push_back(p.name);
                if (p.bounds.minimum < p.bounds.maximum) {
                    bounds.push_back(p.bounds);
                    initial.push_back(v);
                    names.push_back(p.name);
                }
            }
        }
        if ((problem.region_normalization != RegionNormalization::Sum &&
             problem.region_normalization != RegionNormalization::WeightSum) ||
            (problem.missing_prediction != MissingPrediction::Reject &&
             problem.missing_prediction != MissingPrediction::Zero))
            throw std::invalid_argument("unknown objective policy");
        std::set<std::string> selected;
        for (const auto &s : problem.selections) {
            core::find_measurement(reference.setup, s.measurement);
            if (!selected.insert(s.measurement).second)
                throw std::invalid_argument("duplicate measurement selection");
        }
        fixed_denominator = problem.loss == Loss::DataChiSquare ||
                            problem.loss == Loss::VarianceChiSquare ||
                            problem.loss == Loss::Neyman || problem.loss == Loss::Mse;
        std::size_t total = 0;
        for (const auto &d : problem.data.spectra) {
            Score score;
            auto it = std::find_if(problem.selections.begin(), problem.selections.end(),
                                   [&](const auto &s) { return s.measurement == d.measurement; });
            double w = it == problem.selections.end() ? 1 : it->weight;
            if (!std::isfinite(w) || w < 0)
                throw std::invalid_argument("invalid measurement weight");
            core::Counts weights(d.counts.size(),
                                 it == problem.selections.end() || it->regions.empty() ? w : 0);
            if (it != problem.selections.end() && !it->regions.empty()) {
                std::vector<bool> occupied(d.counts.size());
                for (const auto &r : it->regions) {
                    if (r.start >= r.stop || r.stop > d.counts.size() || !std::isfinite(r.weight) ||
                        r.weight < 0)
                        throw std::invalid_argument("invalid channel region");
                    for (auto j = r.start; j < r.stop; ++j) {
                        if (occupied[j] && !problem.allow_overlapping_regions)
                            throw std::invalid_argument("overlapping channel regions");
                        occupied[j] = true;
                        weights[j] += w * r.weight;
                    }
                }
            }
            if (problem.region_normalization == RegionNormalization::WeightSum &&
                it != problem.selections.end() && !it->regions.empty()) {
                double sum = 0;
                for (const auto &r : it->regions)
                    sum += r.weight;
                if (sum > 0)
                    for (auto &v : weights)
                        v /= sum;
            }
            for (std::size_t j = 0; j < weights.size(); ++j)
                if (weights[j] > 0 && (d.valid.empty() || d.valid[j])) {
                    if (problem.loss == Loss::VarianceChiSquare) {
                        if (d.variances.empty() || d.variances[j] <= 0)
                            throw std::invalid_argument(
                                "selected channels need positive variances");
                    }
                    score.channels.push_back(j);
                    double denominator = 1;
                    if (problem.loss == Loss::DataChiSquare)
                        denominator = d.counts[j] + problem.data_floor;
                    if (problem.loss == Loss::Neyman)
                        denominator = std::max(d.counts[j], 1.);
                    if (problem.loss == Loss::VarianceChiSquare)
                        denominator = d.variances[j];
                    score.coefficients.push_back(weights[j] / denominator);
                    ++total;
                }
            scores.push_back(std::move(score));
        }
        if (!total)
            throw std::invalid_argument("no observed channels selected");
        best.coordinates.resize(bounds.size());
        best.parameter_values.resize(best.parameter_names.size());
        for (std::size_t i = 0; i < scores.size(); ++i)
            best.contributions.push_back(
                {problem.data.spectra[i].measurement, 0, scores[i].channels.size()});
    }
    bool decode(std::span<const double> x, core::Experiment &e) {
        auto &work = parameterization ? macro_reference : e;
        if (x.size() != bounds.size())
            throw std::invalid_argument("candidate dimension differs");
        for (std::size_t j = 0; j < x.size(); ++j)
            if (!bounds[j].contains(x[j]))
                return false;
        for (auto l : ad_layers) {
            auto &a = work.sample.layers[l];
            const auto &r = reference.sample.layers[l];
            for (std::size_t k = 0; k < a.species.size(); ++k)
                a.species[k].concentration = r.thickness * r.species[k].concentration;
        }
        for (const auto &s : scalars) {
            auto v = s.coordinate == absent ? s.fixed : x[s.coordinate];
            for (std::size_t j = 0; j < s.bindings.size(); ++j)
                s.bindings[j].ref(work) = v * s.ratios[j];
        }
        for (const auto &g : groups) {
            double remaining = g.total;
            for (std::size_t j = 0; j < g.bindings.size(); ++j) {
                double low = 0, high = 0;
                for (std::size_t k = j + 1; k < g.bindings.size(); ++k) {
                    low += g.bounds[k].minimum;
                    high += g.bounds[k].maximum;
                }
                auto lo = std::max(g.bounds[j].minimum, remaining - high),
                     hi = std::min(g.bounds[j].maximum, remaining - low);
                if (hi < lo - 1e-10)
                    return false;
                double v = j + 1 == g.bindings.size() ? remaining
                           : g.coordinates[j] == absent
                               ? lo
                               : lo + std::max(0., hi - lo) * x[g.coordinates[j]];
                g.bindings[j].ref(work) = v;
                remaining -= v;
            }
        }
        for (auto l : ad_layers) {
            auto &a = work.sample.layers[l];
            double total = 0;
            for (const auto &s : a.species)
                total += s.concentration;
            a.thickness = total;
            for (std::size_t k = 0; k < a.species.size(); ++k)
                a.species[k].concentration =
                    total > 0 ? a.species[k].concentration / total
                              : reference.sample.layers[l].species[k].concentration;
        }
        if (parameterization) {
            for (std::size_t j = 0; j < profile_values.size(); ++j)
                profile_values[j] = profile_coordinates[j] == absent
                                        ? problem.profile_parameters[j].bounds.minimum
                                        : x[profile_coordinates[j]];
            try {
                parameterization->apply(profile_values, work.sample, e.sample);
                e.setup = work.setup;
            } catch (const std::invalid_argument &) {
                return false;
            }
        }
        if (!physical(e))
            return false;
        for (std::size_t i = 0; i < e.setup.measurements.size(); ++i) {
            const auto &d = e.setup.measurements[i].detector;
            const double end = double(measurement_channels[i]) - .5;
            const double left =
                d.calibrationOffset - .5 * (d.calibrationLinear - .5 * d.calibrationQuadratic);
            const double right =
                d.calibrationOffset + end * (d.calibrationLinear + end * d.calibrationQuadratic);
            if (!std::isfinite(left) || !std::isfinite(right) || d.calibrationLinear <= 0 ||
                d.calibrationLinear +
                        2 * (double(measurement_channels[i]) - 1) * d.calibrationQuadratic <=
                    0)
                return false;
        }
        return true;
    }
    double weighted_loss(const core::MeasuredSpectrum &d, std::size_t channel, double predicted,
                         double weight) const {
        if (fixed_denominator) {
            double delta = d.counts[channel] - predicted;
            return delta * delta * weight;
        }
        return loss_value(problem.loss, d.counts[channel], predicted, 0, problem.data_floor) *
               weight;
    }
    double score(const simulation::Result &r, bool compile = false) {
        if (r.failure)
            return INFINITY;
        double total = 0;
        for (std::size_t k = 0; k < scores.size(); ++k) {
            auto &s = scores[k];
            const auto &d = problem.data.spectra[k];
            if (compile) {
                for (std::size_t j = 0; j < r.spectra.size(); ++j)
                    if (r.spectra[j].measurement == d.measurement) {
                        if (s.spectrum != absent)
                            throw std::runtime_error("duplicate predicted measurement");
                        s.spectrum = j;
                    }
            }
            if (s.spectrum >= r.spectra.size() ||
                r.spectra[s.spectrum].measurement != d.measurement)
                throw std::runtime_error("predictor measurement layout differs");
            double measurement_cost = 0;
            const auto &p = r.spectra[s.spectrum].counts;
            if (problem.missing_prediction == MissingPrediction::Reject && !s.channels.empty() &&
                s.channels.back() >= p.size())
                throw std::runtime_error("prediction does not cover selected channels");
            for (std::size_t j = 0; j < s.channels.size(); ++j) {
                auto channel = s.channels[j];
                const double predicted = channel < p.size() ? p[channel] : 0;
                if (!std::isfinite(predicted) || predicted < 0)
                    throw std::runtime_error("nonphysical predicted counts");
                measurement_cost += weighted_loss(d, channel, predicted, s.coefficients[j]);
            }
            if (measurement_cost < 0)
                return INFINITY;
            total += measurement_cost;
        }
        return total;
    }
    void save(const core::Experiment &e, const simulation::Result &r, std::span<const double> x,
              double cost) {
        if (cost >= best.cost)
            return;
        best.cost = cost;
        best.experiment = e;
        best.prediction = r;
        best.coordinates.assign(x.begin(), x.end());
        std::size_t physical_index = 0;
        for (std::size_t j = 0; j < scalars.size(); ++j) {
            const auto &p = scalars[j];
            best.parameter_values[physical_index++] =
                p.coordinate == absent ? p.fixed : x[p.coordinate];
        }
        for (std::size_t k = 0; k < groups.size(); ++k)
            for (std::size_t j = 0; j < groups[k].bindings.size(); ++j) {
                best.parameter_values[physical_index++] =
                    groups[k].bindings[j].ref(best.experiment);
            }
        for (std::size_t j = 0; j < profile_coordinates.size(); ++j)
            best.parameter_values[physical_index++] =
                profile_coordinates[j] == absent ? problem.profile_parameters[j].bounds.minimum
                                                 : x[profile_coordinates[j]];
        for (std::size_t k = 0; k < scores.size(); ++k) {
            const auto &s = scores[k];
            const auto &d = problem.data.spectra[k];
            double v = 0;
            for (std::size_t j = 0; j < s.channels.size(); ++j) {
                auto channel = s.channels[j];
                const auto &counts = r.spectra[s.spectrum].counts;
                v += weighted_loss(d, channel, channel < counts.size() ? counts[channel] : 0,
                                   s.coefficients[j]);
            }
            best.contributions[k].cost = v;
        }
    }
};
Objective::Objective(Problem p, std::shared_ptr<Predictor> model)
    : impl_(std::make_unique<Impl>(std::move(p), std::move(model))) {
    auto &a = *impl_;
    a.inputs.push_back(a.reference);
    if (!a.decode(a.initial, a.inputs[0]))
        throw std::invalid_argument("initial fitting experiment is infeasible");
    a.outputs.resize(1);
    a.predictor->predict(a.inputs, a.outputs);
    double cost = a.score(a.outputs[0], true);
    if (!std::isfinite(cost))
        throw std::runtime_error(
            "initial fitting prediction failed" +
            (a.outputs[0].failure ? ": " + a.outputs[0].failure->message : std::string{}));
    for (const auto &s : a.outputs[0].spectra) {
        const auto m = std::find_if(
            a.reference.setup.measurements.begin(), a.reference.setup.measurements.end(),
            [&](const auto &value) { return value.label == s.measurement; });
        if (m != a.reference.setup.measurements.end()) {
            const auto index = std::size_t(m - a.reference.setup.measurements.begin());
            a.measurement_channels[index] =
                std::max(a.measurement_channels[index], s.counts.size());
        }
    }
    a.save(a.inputs[0], a.outputs[0], a.initial, cost);
    a.evaluations = 1;
}
Objective::~Objective() = default;
std::size_t Objective::dimension() const { return impl_->bounds.size(); }
std::span<const core::ParameterBounds> Objective::bounds() const { return impl_->bounds; }
std::span<const std::string> Objective::coordinate_names() const { return impl_->names; }
std::span<const double> Objective::initial_coordinates() const { return impl_->initial; }
void Objective::evaluate_batch(std::span<const double> rows, std::size_t population,
                               std::span<double> costs) {
    auto &a = *impl_;
    if (rows.size() != population * dimension() || costs.size() != population)
        throw std::invalid_argument("candidate/cost shapes differ");
    std::fill(costs.begin(), costs.end(), INFINITY);
    if (stopped())
        return;
    while (a.inputs.size() < population)
        a.inputs.push_back(a.reference);
    a.outputs.resize(std::max(population, a.outputs.size()));
    a.indices.clear();
    a.indices.reserve(population);
    for (std::size_t j = 0; j < population; ++j) {
        if (a.decode(rows.subspan(j * dimension(), dimension()), a.inputs[a.indices.size()]))
            a.indices.push_back(j);
        else
            ++a.invalid;
    }
    if (!a.indices.empty()) {
        a.predictor->predict(std::span<const core::Experiment>(a.inputs.data(), a.indices.size()),
                             std::span<simulation::Result>(a.outputs.data(), a.indices.size()));
        for (std::size_t j = 0; j < a.indices.size(); ++j) {
            auto index = a.indices[j];
            auto value = a.score(a.outputs[j]);
            costs[index] = value;
            if (!std::isfinite(value))
                ++a.invalid;
            else
                a.save(a.inputs[j], a.outputs[j], rows.subspan(index * dimension(), dimension()),
                       value);
        }
    }
    a.evaluations += population;
}
core::Experiment Objective::decode(std::span<const double> row) const {
    auto result = impl_->reference;
    if (!impl_->decode(row, result))
        throw std::invalid_argument("infeasible fitting candidate");
    return result;
}
Result Objective::best_result() const {
    auto result = impl_->best;
    result.evaluations = impl_->evaluations;
    result.invalid = impl_->invalid;
    result.status = stopped() ? "cancelled" : "evaluated";
    return result;
}
std::span<const double> Objective::best_coordinates() const { return impl_->best.coordinates; }
Progress Objective::progress() const {
    return {impl_->evaluations, impl_->invalid, impl_->best.cost};
}
void Objective::request_stop() noexcept {
    impl_->stop = true;
    impl_->predictor->request_stop();
}
bool Objective::stopped() const noexcept { return impl_->stop || impl_->predictor->stopped(); }
} // namespace ibeamlab::fitting
