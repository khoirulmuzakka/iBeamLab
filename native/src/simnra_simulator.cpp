#include <ibeamlab/simnra_simulator.h>

#ifdef IBEAMLAB_HAS_SIMNRA
#include "simnra.h"
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace ibeamlab::simulator {

#ifdef IBEAMLAB_HAS_SIMNRA
namespace {
std::atomic_uint64_t temporaryDirectoryCounter{};

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
sample::Detector &detector(SimulationInput &input, const std::string &label) {
    return const_cast<sample::Detector &>(detector(std::as_const(input), label));
}
void configureTarget(SIMNRA &sim, const sample::SampleModel &sample) {
    // NumberOfLayers is read-only in current SIMNRA COM versions. Rebuild the
    // target through the supported collection methods instead of assigning it.
    while (sim.getNumberOfLayers() > 0) {
        if (!sim.deleteLayer(1))
            throw std::runtime_error("SIMNRA failed to delete an existing target layer");
    }
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
bool sameTarget(const sample::SampleModel &left, const sample::SampleModel &right) {
    if (left.layers.size() != right.layers.size())
        return false;
    for (std::size_t layerIndex = 0; layerIndex < left.layers.size(); ++layerIndex) {
        const auto &a = left.layers[layerIndex];
        const auto &b = right.layers[layerIndex];
        if (a.thickness != b.thickness || a.roughness != b.roughness ||
            a.porosityFraction != b.porosityFraction || a.poreDiameter != b.poreDiameter ||
            a.species.size() != b.species.size())
            return false;
        for (std::size_t speciesIndex = 0; speciesIndex < a.species.size(); ++speciesIndex) {
            const auto &as = a.species[speciesIndex];
            const auto &bs = b.species[speciesIndex];
            if (as.element != bs.element || as.concentration != bs.concentration ||
                as.isotopes.size() != bs.isotopes.size())
                return false;
            for (std::size_t isotopeIndex = 0; isotopeIndex < as.isotopes.size(); ++isotopeIndex) {
                const auto &ai = as.isotopes[isotopeIndex];
                const auto &bi = bs.isotopes[isotopeIndex];
                if (ai.massNumber != bi.massNumber || ai.exactMass != bi.exactMass ||
                    ai.fraction != bi.fraction)
                    return false;
            }
        }
    }
    return true;
}
bool sameSetup(const sample::Detector &a, const sample::Detector &b) {
    return a.label == b.label && a.beam.energy == b.beam.energy &&
           a.beam.spread == b.beam.spread && a.calibrationLinear == b.calibrationLinear &&
           a.particlesSr == b.particlesSr && a.calibrationOffset == b.calibrationOffset &&
           a.calibrationQuadratic == b.calibrationQuadratic && a.resolution == b.resolution &&
           a.realTime == b.realTime && a.liveTime == b.liveTime;
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

sample::SampleModel readTarget(SIMNRA &sim) {
    sample::SampleModel result;
    const int layerCount = sim.getNumberOfLayers();
    result.layers.reserve(static_cast<std::size_t>(layerCount));
    for (int layerIndex = 1; layerIndex <= layerCount; ++layerIndex) {
        sample::Layer layer;
        layer.thickness = sim.getLayerThickness(layerIndex);
        if (sim.getHasLayerRoughness(layerIndex))
            layer.roughness = sim.getLayerRoughness(layerIndex);
        if (sim.getHasLayerPorosity(layerIndex)) {
            layer.porosityFraction = sim.getPorosityFraction(layerIndex);
            layer.poreDiameter = sim.getPoreDiameter(layerIndex);
        }
        const int elementCount = sim.getNumberOfElements(layerIndex);
        layer.species.reserve(static_cast<std::size_t>(elementCount));
        for (int elementIndex = 1; elementIndex <= elementCount; ++elementIndex) {
            sample::Species species;
            species.element = w2s(sim.getElementName(layerIndex, elementIndex));
            while (!species.element.empty() &&
                   std::isspace(static_cast<unsigned char>(species.element.back())))
                species.element.pop_back();
            species.concentration = sim.getElementConcentration(layerIndex, elementIndex);
            const int isotopeCount = sim.getNumberOfIsotopes(layerIndex, elementIndex);
            species.isotopes.reserve(static_cast<std::size_t>(isotopeCount));
            for (int isotopeIndex = 1; isotopeIndex <= isotopeCount; ++isotopeIndex) {
                sample::Isotope isotope;
                isotope.exactMass = sim.getIsotopeMass(layerIndex, elementIndex, isotopeIndex);
                isotope.massNumber = static_cast<std::int32_t>(std::lround(isotope.exactMass));
                isotope.fraction =
                    sim.getIsotopeConcentration(layerIndex, elementIndex, isotopeIndex);
                species.isotopes.push_back(isotope);
            }
            layer.species.push_back(std::move(species));
        }
        result.layers.push_back(std::move(layer));
    }
    return result;
}

sample::Detector readSetup(SIMNRA &sim, const sample::Detector &requested) {
    sample::Detector result;
    result.label = requested.label;
    result.beam.particle = requested.beam.particle;
    result.beam.energy = sim.getBeamEnergy();
    result.beam.spread = sim.getBeamSpread();
    result.calibrationLinear = sim.getCalibrationLinear();
    result.particlesSr = sim.getParticlesSr();
    result.calibrationOffset = sim.getCalibrationOffset();
    result.calibrationQuadratic = sim.getCalibrationQuadratic();
    result.resolution = sim.getDetectorResolution();
    result.realTime = sim.getRealTime();
    result.liveTime = sim.getLiveTime();
    result.info = requested.info;
    return result;
}
} // namespace
#endif

struct SimnraSimulator::Impl {
    explicit Impl(SimnraSimulatorConfig value) : config(std::move(value)) {
        if (config.methods.empty() || config.workers == 0)
            throw std::invalid_argument("SIMNRA methods and workers cannot be empty");
#ifdef IBEAMLAB_HAS_SIMNRA
        for (std::size_t i = 0; i < config.workers; ++i)
            workers.push_back(std::make_unique<Worker>(*this, i));
#else
        throw std::runtime_error("iBeamLab was built without SIMNRA support");
#endif
    }
    ~Impl() { close(); }
    SimnraSimulatorConfig config;
    std::atomic_bool stop{false};
#ifdef IBEAMLAB_HAS_SIMNRA
    struct Worker {
        explicit Worker(Impl &owner, std::size_t workerIndex)
            : owner(owner), workerIndex(workerIndex), thread([this] { run(); }) {}
        ~Worker() { shutdown(); }
        std::future<SimulationResult> submit(std::size_t sampleIndex, const SimulationInput &input,
                                             const SimulationOptions &options) {
            auto task = std::make_shared<std::packaged_task<SimulationResult()>>(
                [this, sampleIndex, &input, options] {
                    return calculate(sampleIndex, input, options);
                });
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
        std::future<SimulationInput> submitInspection(const SimulationInput &input,
                                                       std::string methodLabel) {
            auto task = std::make_shared<std::packaged_task<SimulationInput()>>(
                [this, &input, methodLabel = std::move(methodLabel)] {
                    return inspect(input, methodLabel);
                });
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
        void ensureInstances() {
            if (!instances.empty())
                return;
            if (referenceCopies.empty()) {
                const auto sequence = temporaryDirectoryCounter.fetch_add(1);
                workingDirectory = std::filesystem::temp_directory_path() /
                    ("ibeamlab-simnra-" + std::to_string(GetCurrentProcessId()) + "-" +
                     std::to_string(workerIndex) + "-" + std::to_string(sequence));
                std::filesystem::create_directories(workingDirectory);
                try {
                    for (std::size_t methodIndex = 0; methodIndex < owner.config.methods.size();
                         ++methodIndex) {
                        const auto &source = owner.config.methods[methodIndex].referenceFile;
                        if (!std::filesystem::is_regular_file(source))
                            throw std::runtime_error("SIMNRA reference file does not exist: " +
                                                     source.string());
                        const auto destination = workingDirectory /
                            (std::to_wstring(methodIndex) + L"_" + source.filename().wstring());
                        std::filesystem::copy_file(source, destination,
                                                   std::filesystem::copy_options::overwrite_existing);
                        referenceCopies.push_back(destination);
                    }
                } catch (...) {
                    std::error_code ignored;
                    std::filesystem::remove_all(workingDirectory, ignored);
                    workingDirectory.clear();
                    referenceCopies.clear();
                    throw;
                }
            }
            try {
                for (std::size_t methodIndex = 0; methodIndex < owner.config.methods.size();
                     ++methodIndex) {
                    auto sim = std::make_unique<SIMNRA>(owner.config.multithreadedApartment,
                                                        owner.config.threadPriority);
                    const auto path = wide(referenceCopies[methodIndex]);
                    sim->open(path.c_str(), -1);
                    instances.push_back(std::move(sim));
                }
            } catch (...) {
                discardInstances();
                throw;
            }
            cachedTopologies.resize(instances.size());
            cachedTargets.resize(instances.size());
            cachedSetups.resize(instances.size());
        }
        void discardInstances() {
            instances.clear();
            cachedTopologies.clear();
            cachedTargets.clear();
            cachedSetups.clear();
        }
        SIMNRA &applyConfiguration(std::size_t methodIndex, const SimulationInput &input) {
            auto &sim = *instances.at(methodIndex);
            const auto topology = targetTopology(input.sample);
            if (!cachedTargets[methodIndex] ||
                !sameTarget(*cachedTargets[methodIndex], input.sample)) {
                if (cachedTopologies[methodIndex] == topology)
                    updateTarget(sim, input.sample);
                else
                    configureTarget(sim, input.sample);
                cachedTopologies[methodIndex] = topology;
                cachedTargets[methodIndex] = input.sample;
            }

            const auto &requestedSetup =
                detector(input, owner.config.methods[methodIndex].label);
            if (!cachedSetups[methodIndex] ||
                !sameSetup(*cachedSetups[methodIndex], requestedSetup)) {
                configureSetup(sim, requestedSetup);
                cachedSetups[methodIndex] = requestedSetup;
            }
            return sim;
        }
        SimulationInput inspect(const SimulationInput &input, const std::string &methodLabel) {
            input.sample.validate();
            input.setup.validate();
            ensureInstances();
            const auto method = std::find_if(owner.config.methods.begin(), owner.config.methods.end(),
                [&](const auto &value) { return value.label == methodLabel; });
            if (method == owner.config.methods.end())
                throw std::invalid_argument("unknown SIMNRA method: " + methodLabel);
            const auto methodIndex = static_cast<std::size_t>(method - owner.config.methods.begin());
            auto &sim = applyConfiguration(methodIndex, input);
            const auto &requestedSetup = detector(input, methodLabel);

            SimulationInput result = input;
            result.sample = readTarget(sim);
            auto &resultSetup = detector(result, methodLabel);
            resultSetup = readSetup(sim, requestedSetup);
            return result;
        }
        SimulationResult calculate(std::size_t sampleIndex, const SimulationInput &input,
                                   const SimulationOptions &options) {
            SimulationResult result;
            if (owner.stop.load())
                return result;
            std::string currentMethod;
            try {
                input.sample.validate();
                input.setup.validate();
                ensureInstances();
                for (std::size_t m = 0; m < owner.config.methods.size(); ++m) {
                    if (owner.stop.load())
                        break;
                    currentMethod = owner.config.methods[m].label;
                    auto &sim = applyConfiguration(m, input);
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
            } catch (const std::exception &error) {
                result.failure =
                    SimulationFailure{sampleIndex, currentMethod, "SimnraError", error.what()};
                discardInstances();
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
            discardInstances();
            // SIMNRA is an out-of-process COM server and may release its file
            // handle shortly after the final IDispatch release.
            for (int attempt = 0; attempt < 100 && !workingDirectory.empty(); ++attempt) {
                std::error_code ignored;
                std::filesystem::remove_all(workingDirectory, ignored);
                if (!std::filesystem::exists(workingDirectory, ignored))
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            referenceCopies.clear();
            workingDirectory.clear();
        }
        Impl &owner;
        std::size_t workerIndex;
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<std::function<void()>> queue;
        bool closing = false;
        std::filesystem::path workingDirectory;
        std::vector<std::filesystem::path> referenceCopies;
        std::vector<std::unique_ptr<SIMNRA>> instances;
        std::vector<std::string> cachedTopologies;
        std::vector<std::optional<sample::SampleModel>> cachedTargets;
        std::vector<std::optional<sample::Detector>> cachedSetups;
        // Keep this last: C++ initializes members in declaration order, so the
        // queue and synchronization state must exist before run() can start.
        std::thread thread;
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
SimulationInput SimnraSimulator::inspectConfiguration(const SimulationInput &input,
                                                       const std::string &methodLabel) {
#ifdef IBEAMLAB_HAS_SIMNRA
    if (impl_->closed.load())
        throw std::logic_error("SimnraSimulator is closed");
    return impl_->workers.front()->submitInspection(input, methodLabel).get();
#else
    (void)input;
    (void)methodLabel;
    throw std::runtime_error("iBeamLab was built without SIMNRA support");
#endif
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
