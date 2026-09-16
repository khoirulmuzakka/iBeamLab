#include <ibeamlab/simnra_simulator.h>

#ifdef IBEAMLAB_HAS_SIMNRA
#include "simnra.h"
#include <windows.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ibeamlab::simulator {

#ifdef IBEAMLAB_HAS_SIMNRA
namespace {
std::wstring wide(const std::filesystem::path &path) { return path.wstring(); }
std::wstring wide(const std::string &text) {
    if (text.empty())
        return {};
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                                         static_cast<int>(text.size()), nullptr, 0);
    if (size <= 0)
        throw std::invalid_argument("invalid UTF-8 element name");
    std::wstring output(size, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
                        output.data(), size);
    return output;
}
const sample::Detector &detector(const SimulationInput &input, const std::string &label) {
    const auto it = std::find_if(input.setup.detectors.begin(), input.setup.detectors.end(),
                                 [&](const auto &value) { return value.label == label; });
    if (it == input.setup.detectors.end())
        throw std::invalid_argument("missing detector setup: " + label);
    return *it;
}
void configureTarget(SIMNRA &sim, const sample::SampleModel &sample) {
    sim.setNumberOfLayers(0);
    for (std::size_t layerIndex = 0; layerIndex < sample.layers.size(); ++layerIndex) {
        const auto &layer = sample.layers[layerIndex];
        std::vector<std::wstring> names;
        std::vector<double> concentrations;
        for (const auto &species : layer.species) {
            names.push_back(wide(species.element));
            concentrations.push_back(species.concentration);
        }
        sim.addLayerProperties(std::move(names), std::move(concentrations), layer.thickness);
        const int li = static_cast<int>(layerIndex + 1);
        sim.setHasLayerRoughness(li, layer.roughness > 0);
        if (layer.roughness > 0)
            sim.setLayerRoughness(li, layer.roughness);
        sim.setHasLayerPorosity(li, layer.porosityFraction > 0);
        if (layer.porosityFraction > 0) {
            sim.setPorosityFraction(li, layer.porosityFraction);
            sim.setPoreDiameter(li, layer.poreDiameter);
        }
        for (std::size_t elementIndex = 0; elementIndex < layer.species.size(); ++elementIndex) {
            const auto &isotopes = layer.species[elementIndex].isotopes;
            if (isotopes.empty())
                continue;
            const int ei = static_cast<int>(elementIndex + 1);
            while (sim.getNumberOfIsotopes(li, ei) > 0)
                sim.deleteIsotope(li, ei, 1);
            for (std::size_t isotopeIndex = 0; isotopeIndex < isotopes.size(); ++isotopeIndex) {
                sim.addIsotope(li, ei);
                const int ii = static_cast<int>(isotopeIndex + 1);
                const auto &isotope = isotopes[isotopeIndex];
                sim.setIsotopeMass(li, ei, ii,
                                   isotope.exactMass > 0 ? isotope.exactMass
                                                         : static_cast<double>(isotope.massNumber));
                sim.setIsotopeConcentration(li, ei, ii, isotope.fraction);
            }
        }
    }
}
std::string targetTopology(const sample::SampleModel &sample) {
    std::string result;
    for (const auto &layer : sample.layers) {
        result += "L";
        for (const auto &species : layer.species)
            result += species.element + ":" + std::to_string(species.isotopes.size()) + ";";
    }
    return result;
}
void updateTarget(SIMNRA &sim, const sample::SampleModel &sample) {
    for (std::size_t layerIndex = 0; layerIndex < sample.layers.size(); ++layerIndex) {
        const auto &layer = sample.layers[layerIndex];
        const int layerId = static_cast<int>(layerIndex + 1);
        sim.setLayerThickness(layerId, layer.thickness);
        std::vector<double> concentrations;
        for (const auto &species : layer.species)
            concentrations.push_back(species.concentration);
        sim.setElementConcentrationArray(layerId, concentrations);
        sim.setHasLayerRoughness(layerId, layer.roughness > 0);
        if (layer.roughness > 0)
            sim.setLayerRoughness(layerId, layer.roughness);
        sim.setHasLayerPorosity(layerId, layer.porosityFraction > 0);
        if (layer.porosityFraction > 0) {
            sim.setPorosityFraction(layerId, layer.porosityFraction);
            sim.setPoreDiameter(layerId, layer.poreDiameter);
        }
        for (std::size_t elementIndex = 0; elementIndex < layer.species.size(); ++elementIndex)
            for (std::size_t isotopeIndex = 0;
                 isotopeIndex < layer.species[elementIndex].isotopes.size(); ++isotopeIndex) {
                const auto &isotope = layer.species[elementIndex].isotopes[isotopeIndex];
                const int elementId = static_cast<int>(elementIndex + 1);
                const int isotopeId = static_cast<int>(isotopeIndex + 1);
                sim.setIsotopeMass(layerId, elementId, isotopeId,
                    isotope.exactMass > 0 ? isotope.exactMass
                                          : static_cast<double>(isotope.massNumber));
                sim.setIsotopeConcentration(layerId, elementId, isotopeId, isotope.fraction);
            }
    }
}
void configureSetup(SIMNRA &sim, const sample::Detector &setup) {
    sim.setParticlesSr(setup.particlesSr);
    sim.setCalibrationOffset(setup.calibrationOffset);
    sim.setCalibrationLinear(setup.calibrationLinear);
    sim.setCalibrationQuadratic(setup.calibrationQuadratic);
    sim.setDetectorResolution(setup.resolution);
    sim.setBeamEnergy(setup.beam.energy);
    sim.setBeamSpread(setup.beam.spread);
    if (setup.realTime > 0)
        sim.setRealTime(setup.realTime);
    if (setup.liveTime > 0)
        sim.setLiveTime(setup.liveTime);
}
} // namespace
#endif

struct SimnraSimulator::Impl {
    explicit Impl(SimnraSimulatorConfig value) : config(std::move(value)) {
        if (config.methods.empty() || config.workers == 0)
            throw std::invalid_argument("SIMNRA methods and workers cannot be empty");
#ifdef IBEAMLAB_HAS_SIMNRA
        for (std::size_t i = 0; i < config.workers; ++i)
            workers.push_back(std::make_unique<Worker>(*this));
#else
        throw std::runtime_error("iBeamLab was built without SIMNRA support");
#endif
    }
    ~Impl() { close(); }
    SimnraSimulatorConfig config;
    std::atomic_bool stop{false};
#ifdef IBEAMLAB_HAS_SIMNRA
    struct Worker {
        explicit Worker(Impl &owner) : owner(owner), thread([this] { run(); }) {}
        ~Worker() { shutdown(); }
        std::future<SimulationResult> submit(std::size_t index, const SimulationInput &input,
                                             const SimulationOptions &options) {
            auto task = std::make_shared<std::packaged_task<SimulationResult()>>(
                [this, index, &input, options] { return calculate(index, input, options); });
            auto future = task->get_future();
            {
                std::lock_guard lock(mutex);
                if (closing)
                    throw std::logic_error("SIMNRA worker is closed");
                queue.emplace_back([task] { (*task)(); });
            }
            condition.notify_one();
            return future;
        }
        void shutdown() {
            {
                std::lock_guard lock(mutex);
                closing = true;
            }
            condition.notify_one();
            if (thread.joinable())
                thread.join();
        }
        SimulationResult calculate(std::size_t index, const SimulationInput &input,
                                   const SimulationOptions &options) {
            SimulationResult result;
            if (owner.stop.load())
                return result;
            std::string currentMethod;
            try {
                input.sample.validate();
                input.setup.validate();
                if (instances.empty()) {
                    for (const auto &method : owner.config.methods) {
                        currentMethod = method.label;
                        auto sim = std::make_unique<SIMNRA>(owner.config.multithreadedApartment,
                                                            owner.config.threadPriority);
                        const auto path = wide(method.referenceFile);
                        sim->open(path.c_str(), -1);
                        instances.push_back(std::move(sim));
                    }
                }
                const auto topology = targetTopology(input.sample);
                for (std::size_t m = 0; m < owner.config.methods.size(); ++m) {
                    if (owner.stop.load())
                        break;
                    currentMethod = owner.config.methods[m].label;
                    auto &sim = *instances[m];
                    if (topology == cachedTopology)
                        updateTarget(sim, input.sample);
                    else
                        configureTarget(sim, input.sample);
                    configureSetup(sim, detector(input, currentMethod));
                    sim.setCalc_ElementSpectra(options.captureElementalSpectra);
                    const bool ok = owner.config.fastCalculation ? sim.calculateSpectrumFast()
                                                                 : sim.calculateSpectrum();
                    if (!ok)
                        throw std::runtime_error("SIMNRA calculation failed");
                    auto values = sim.getSpectrum(2);
                    Spectrum spectrum;
                    spectrum.label = currentMethod;
                    spectrum.counts.reserve(values.size());
                    for (double value : values)
                        spectrum.counts.push_back(static_cast<float>(value));
                    result.spectra.push_back(std::move(spectrum));
                    if (options.captureElementalSpectra) {
                        for (std::size_t li = 0; li < input.sample.layers.size(); ++li)
                            for (std::size_t ei = 0; ei < input.sample.layers[li].species.size();
                                 ++ei) {
                                const auto &species = input.sample.layers[li].species[ei];
                                if (result.elementalSpectra[currentMethod].count(species.element))
                                    continue;
                                const int z = sim.getElementZ(static_cast<int>(li + 1),
                                                              static_cast<int>(ei + 1));
                                const int spectrumId = sim.spectrumIDOfElement(z);
                                auto element = sim.getSpectrum(spectrumId);
                                Counts counts;
                                counts.reserve(element.size());
                                for (double value : element)
                                    counts.push_back(static_cast<float>(value));
                                result.elementalSpectra[currentMethod].emplace(species.element,
                                                                               std::move(counts));
                            }
                    }
                }
                cachedTopology = topology;
            } catch (const std::exception &error) {
                result.failure =
                    SimulationFailure{index, currentMethod, "SimnraError", error.what()};
                instances.clear();
                cachedTopology.clear();
            }
            return result;
        }
        void run() {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock lock(mutex);
                    condition.wait(lock, [&] { return closing || !queue.empty(); });
                    if (closing && queue.empty())
                        break;
                    task = std::move(queue.front());
                    queue.pop_front();
                }
                task();
            }
            instances.clear();
        }
        Impl &owner;
        std::thread thread;
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<std::function<void()>> queue;
        bool closing = false;
        std::vector<std::unique_ptr<SIMNRA>> instances;
        std::string cachedTopology;
    };
    std::vector<std::unique_ptr<Worker>> workers;
#endif
    std::atomic_bool closed{false};
    void close() noexcept {
#ifdef IBEAMLAB_HAS_SIMNRA
        if (!closed.exchange(true))
            workers.clear();
#endif
    }
};

SimnraSimulator::SimnraSimulator(SimnraSimulatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
SimnraSimulator::~SimnraSimulator() = default;
std::vector<SimulationResult>
SimnraSimulator::simulateBatch(const std::vector<SimulationInput> &inputs,
                               const SimulationOptions &options) {
    resetStop();
    std::vector<SimulationResult> results(inputs.size());
#ifdef IBEAMLAB_HAS_SIMNRA
    if (impl_->closed.load())
        throw std::logic_error("SimnraSimulator is closed");
    std::vector<std::future<SimulationResult>> futures;
    futures.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        futures.push_back(impl_->workers[i % impl_->workers.size()]->submit(i, inputs[i], options));
    for (std::size_t i = 0; i < futures.size(); ++i) {
        results[i] = futures[i].get();
        if (results[i].failure && !options.continueAfterFailure)
            requestStop();
    }
#endif
    return results;
}
void SimnraSimulator::requestStop() noexcept {
    ISimulator::requestStop();
    if (impl_)
        impl_->stop.store(true);
}
void SimnraSimulator::resetStop() noexcept {
    ISimulator::resetStop();
    if (impl_)
        impl_->stop.store(false);
}
void SimnraSimulator::close() noexcept {
    if (impl_)
        impl_->close();
}

} // namespace ibeamlab::simulator
