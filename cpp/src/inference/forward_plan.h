#pragma once
#include <functional>
#include <ibeamlab/model/metadata.h>
#include <limits>
#include <optional>

namespace ibeamlab::inference::detail {
// Resolve parameter locations and permitted changes once. Only scalar values and
// structural identities are checked per call; no experiment copies or encoding.
class ForwardPlan {
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

  public:
    ForwardPlan(const model::ForwardModelMetadata &metadata) {
        const auto &reference = metadata.reference;
        actual_.resize(reference.setup.measurements.size());
        for (std::size_t i = 0; i < actual_.size(); ++i)
            actual_[i] = &reference.setup.measurements[i];
        std::vector<std::string> keys;
        for (const auto &parameter : metadata.input_parameters)
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

    void prepare(const core::Experiment &x, const core::Experiment &reference,
                 std::span<float> input) {
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
} // namespace ibeamlab::inference::detail
