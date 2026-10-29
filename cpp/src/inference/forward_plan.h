#pragma once
#include <algorithm>
#include <functional>
#include <ibeamlab/model/metadata.h>
#include <limits>
#include <optional>
#include <unordered_map>

namespace ibeamlab::inference::detail {
// Resolve parameter locations and permitted changes once. Only scalar values and
// structural identities are checked per call; no experiment copies or encoding.
class ParameterPlan {
    static constexpr auto absent = std::numeric_limits<std::size_t>::max();
    using Measurements = std::span<const core::Measurement *const>;
    struct Scalar {
        std::function<double(const core::Experiment &, Measurements)> read;
        double reference, minimum, maximum;
        std::size_t column;
        bool variable;
    };
    std::vector<Scalar> scalars_;
    std::vector<const core::Measurement *> actual_;
    core::Experiment reference_;

  public:
    ParameterPlan(const model::ForwardModelMetadata &metadata)
        : reference_{std::get<model::ParameterInputSchema>(metadata.input).reference_sample,
                     metadata.reference_setup} {
        const auto &reference = reference_;
        actual_.resize(reference.setup.measurements.size());
        for (std::size_t i = 0; i < actual_.size(); ++i)
            actual_[i] = &reference.setup.measurements[i];
        std::vector<std::string> keys;
        for (const auto &parameter : std::get<model::ParameterInputSchema>(metadata.input).parameters)
            keys.push_back(core::parameter_key(parameter.target));
        auto add = [&](auto read, double minimum, double maximum,
                       std::optional<core::ParameterTarget> target = {}, bool variable = false) {
            std::size_t column = absent;
            if (target) {
                auto key = core::parameter_key(*target);
                for (std::size_t i = 0; i < keys.size(); ++i)
                    if (keys[i] == key)
                        column = i;
            }
            double value = read(reference, actual_);
            scalars_.push_back({std::move(read), value, minimum, maximum, column, variable});
        };
        constexpr double infinity = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < reference.sample.layers.size(); ++i) {
            auto layer = [i](double core::Layer::*member) {
                return [i, member](const core::Experiment &x, Measurements) {
                    return x.sample.layers[i].*member;
                };
            };
            add(layer(&core::Layer::thickness), 0, infinity, core::LayerThickness{i});
            add(layer(&core::Layer::roughness), 0, infinity);
            add(layer(&core::Layer::porosityFraction), 0, 1);
            add(layer(&core::Layer::poreDiameter), 0, infinity);
            const auto &species = reference.sample.layers[i].species;
            for (std::size_t j = 0; j < species.size(); ++j)
                add([i, j](const core::Experiment &x,
                           Measurements) { return x.sample.layers[i].species[j].concentration; },
                    0, infinity, core::SpeciesConcentration{i, species[j].element});
        }
        for (std::size_t i = 0; i < actual_.size(); ++i) {
            const auto &label = reference.setup.measurements[i].label;
            bool rebin = false, exposure = false, times = false;
            for (const auto &pipeline : metadata.postprocessing)
                if (pipeline.measurement == label)
                    for (const auto &step : pipeline.steps) {
                        rebin |= std::holds_alternative<model::Rebin>(step);
                        exposure |= std::holds_alternative<model::ExposureScale>(step);
                        times |= std::holds_alternative<model::Pileup>(step) ||
                                 std::holds_alternative<model::RemovePileup>(step);
                    }
            auto measurement = [i](auto member) {
                return [i, member](const core::Experiment &, Measurements actual) {
                    return member(*actual[i]);
                };
            };
            add(measurement([](const auto &m) { return m.beam.energy; }),
                std::numeric_limits<double>::denorm_min(), infinity, core::BeamEnergy{label});
            add(measurement([](const auto &m) { return m.beam.spread; }), 0, infinity,
                core::BeamSpread{label});
            add(measurement([](const auto &m) { return m.detector.resolution; }), 0, infinity,
                core::DetectorResolution{label});
            add(measurement([](const auto &m) { return m.detector.calibrationLinear; }), -infinity,
                infinity, core::CalibrationLinear{label}, rebin);
            add(measurement([](const auto &m) { return m.detector.calibrationOffset; }), -infinity,
                infinity, core::CalibrationOffset{label}, rebin);
            add(measurement([](const auto &m) { return m.detector.calibrationQuadratic; }),
                -infinity, infinity, core::CalibrationQuadratic{label}, rebin);
            add(measurement([](const auto &m) { return m.acquisition.particlesSr; }), 0, infinity,
                core::ParticlesSr{label}, exposure);
            add(measurement([](const auto &m) { return m.acquisition.realTime; }), 0, infinity, {},
                times);
            add(measurement([](const auto &m) { return m.acquisition.liveTime; }), 0, infinity, {},
                times);
        }
    }

    void prepare(const core::Experiment &x, std::span<float> input) {
        const auto &reference = reference_;
        auto mismatch = [] {
            throw std::invalid_argument(
                "experiment differs in fields not covered by model parameters or processing");
        };
        if (x.sample.layers.size() != reference.sample.layers.size() ||
            x.setup.measurements.size() != actual_.size())
            mismatch();
        // Label order may change between calls; resolve into preallocated storage.
        for (std::size_t i = 0; i < actual_.size(); ++i) {
            actual_[i] = nullptr;
            const auto &expected = reference.setup.measurements[i];
            for (const auto &measurement : x.setup.measurements)
                if (measurement.label == expected.label) {
                    if (actual_[i])
                        mismatch();
                    actual_[i] = &measurement;
                }
            if (!actual_[i] || actual_[i]->beam.particle != expected.beam.particle)
                mismatch();
        }
        for (std::size_t i = 0; i < x.sample.layers.size(); ++i) {
            const auto &species = x.sample.layers[i].species;
            const auto &expected = reference.sample.layers[i].species;
            if (species.size() != expected.size())
                mismatch();
            double sum = 0;
            for (std::size_t j = 0; j < species.size(); ++j) {
                const auto &s = species[j];
                const auto &e = expected[j];
                if (s.element != e.element || s.isotopes.size() != e.isotopes.size())
                    mismatch();
                for (std::size_t k = 0; k < s.isotopes.size(); ++k) {
                    const auto &a = s.isotopes[k];
                    const auto &b = e.isotopes[k];
                    if (a.massNumber != b.massNumber || a.exactMass != b.exactMass ||
                        a.fraction != b.fraction)
                        mismatch();
                }
                sum += s.concentration;
            }
            if (!std::isfinite(sum) || std::abs(sum - 1) > 1e-6)
                throw std::invalid_argument("species concentrations must sum to one");
        }
        for (const auto &scalar : scalars_) {
            double value = scalar.read(x, actual_);
            if (!std::isfinite(value) || value < scalar.minimum || value > scalar.maximum)
                throw std::invalid_argument("invalid physical value in forward input");
            if (scalar.column != absent)
                input[scalar.column] = static_cast<float>(value);
            else if (!scalar.variable && value != scalar.reference)
                mismatch();
        }
    }

    const core::Measurement &measurement(std::size_t reference_index) const {
        return *actual_[reference_index];
    }
};

// Sequence packing and setup checks retain all lookup/scratch storage across calls.
class LayerPlan {
    const model::ForwardModelMetadata &metadata_;
    std::unordered_map<std::string, std::size_t> columns_;
    std::vector<unsigned char> seen_;
    struct MeasurementPlan {
        bool rebin = false, exposure = false, times = false;
        const core::Measurement *actual = nullptr;
    };
    std::vector<MeasurementPlan> measurements_;

  public:
    explicit LayerPlan(const model::ForwardModelMetadata &metadata) : metadata_(metadata) {
        const auto &schema = std::get<model::LayerInputSchema>(metadata.input);
        seen_.resize(schema.elements.size());
        for (std::size_t i = 0; i < schema.elements.size(); ++i) columns_.emplace(schema.elements[i], i);
        measurements_.resize(metadata.reference_setup.measurements.size());
        for (std::size_t i = 0; i < measurements_.size(); ++i)
            for (const auto &pipeline : metadata.postprocessing)
                if (pipeline.measurement == metadata.reference_setup.measurements[i].label)
                    for (const auto &step : pipeline.steps) {
                        measurements_[i].rebin |= std::holds_alternative<model::Rebin>(step);
                        measurements_[i].exposure |= std::holds_alternative<model::ExposureScale>(step);
                        measurements_[i].times |= std::holds_alternative<model::Pileup>(step) ||
                                                 std::holds_alternative<model::RemovePileup>(step);
                    }
    }
    void prepare_setup(const core::ExperimentalSetup &setup) {
        if (setup.measurements.size() != measurements_.size())
            throw std::invalid_argument("forward measurement count mismatch");
        auto nonnegative = [](double v) { return std::isfinite(v) && v >= 0; };
        for (std::size_t i = 0; i < measurements_.size(); ++i) {
            auto &plan = measurements_[i];
            const auto &reference = metadata_.reference_setup.measurements[i];
            plan.actual = nullptr;
            for (const auto &m : setup.measurements)
                if (m.label == reference.label) {
                    if (plan.actual) throw std::invalid_argument("duplicate measurement label");
                    plan.actual = &m;
                }
            if (!plan.actual) throw std::invalid_argument("missing measurement");
            const auto &m = *plan.actual;
            if (m.beam.particle != reference.beam.particle || m.beam.energy != reference.beam.energy ||
                m.beam.spread != reference.beam.spread || m.detector.resolution != reference.detector.resolution)
                throw std::invalid_argument("unsupported beam or resolution change");
            if (!std::isfinite(m.detector.calibrationLinear) || !std::isfinite(m.detector.calibrationOffset) ||
                !std::isfinite(m.detector.calibrationQuadratic) || !nonnegative(m.acquisition.particlesSr) ||
                !nonnegative(m.acquisition.realTime) || !nonnegative(m.acquisition.liveTime))
                throw std::invalid_argument("invalid measurement values");
            if (!plan.rebin && (m.detector.calibrationLinear != reference.detector.calibrationLinear ||
                m.detector.calibrationOffset != reference.detector.calibrationOffset ||
                m.detector.calibrationQuadratic != reference.detector.calibrationQuadratic))
                throw std::invalid_argument("calibration changed without rebin");
            if (!plan.exposure && m.acquisition.particlesSr != reference.acquisition.particlesSr)
                throw std::invalid_argument("exposure changed without scaling");
            if (!plan.times && (m.acquisition.realTime != reference.acquisition.realTime ||
                               m.acquisition.liveTime != reference.acquisition.liveTime))
                throw std::invalid_argument("acquisition times changed without processing");
        }
    }
    void pack(const core::Sample &sample, std::span<float> input) {
        const auto width = seen_.size();
        if (sample.layers.empty() || input.size() / width != sample.layers.size() || input.size() % width)
            throw std::invalid_argument("invalid layer input shape");
        std::fill(input.begin(), input.end(), 0.f);
        std::size_t offset = 0;
        for (const auto &layer : sample.layers) {
            if (!std::isfinite(layer.thickness) || layer.thickness < 0 || layer.roughness != 0 ||
                layer.porosityFraction != 0 || layer.poreDiameter != 0 || layer.species.empty())
                throw std::invalid_argument("layer AD inputs require nonnegative thickness and no roughness/porosity");
            std::fill(seen_.begin(), seen_.end(), static_cast<unsigned char>(0));
            double sum = 0;
            for (const auto &s : layer.species) {
                const auto found = columns_.find(s.element);
                if (found == columns_.end()) throw std::invalid_argument("unsupported element: " + s.element);
                const auto column = found->second;
                if (seen_[column]++ || !std::isfinite(s.concentration) || s.concentration < 0 || !s.isotopes.empty())
                    throw std::invalid_argument("invalid or unsupported species in AD input");
                sum += s.concentration;
                const float value = static_cast<float>(layer.thickness * s.concentration);
                if (!std::isfinite(value)) throw std::invalid_argument("areal density exceeds float32 range");
                input[offset + column] = value;
            }
            if (!std::isfinite(sum) || std::abs(sum - 1) > 1e-6)
                throw std::invalid_argument("species concentrations must sum to one");
            offset += width;
        }
    }
    const core::Measurement &measurement(std::size_t index) const { return *measurements_[index].actual; }
};
} // namespace ibeamlab::inference::detail
