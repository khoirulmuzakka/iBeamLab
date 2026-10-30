#include <ibeamlab/core/isotopes.h>
#include "io.h"
#include <cctype>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <future>
#include <ibeamlab/simulation/simnra.h>
#include <thread>
#ifdef IBEAMLAB_HAS_SIMNRA
#define NOMINMAX
#include <map>
// clang-format off
#include <windows.h>
#include <oleauto.h>
// clang-format on
namespace ibeamlab::simulation {
namespace {
void checked(HRESULT hr) {
    if (FAILED(hr))
        throw std::runtime_error("SIMNRA COM HRESULT " +
                                 std::to_string(static_cast<unsigned long>(hr)));
}
struct Value {
    VARIANT value;
    Value() { VariantInit(&value); }
    explicit Value(double d) : Value() {
        value.vt = VT_R8;
        value.dblVal = d;
    }
    explicit Value(int n) : Value() {
        value.vt = VT_I4;
        value.lVal = n;
    }
    explicit Value(bool b) : Value() {
        value.vt = VT_BOOL;
        value.boolVal = b ? VARIANT_TRUE : VARIANT_FALSE;
    }
    explicit Value(const std::wstring &s) : Value() {
        value.vt = VT_BSTR;
        value.bstrVal = SysAllocStringLen(s.data(), static_cast<UINT>(s.size()));
        if (!value.bstrVal)
            throw std::bad_alloc();
    }
    Value(const Value &v) : Value() {
        checked(VariantCopy(&value, const_cast<VARIANT *>(&v.value)));
    }
    Value(Value &&v) noexcept : value(v.value) { VariantInit(&v.value); }
    ~Value() { VariantClear(&value); }
    std::string text() const {
        if (value.vt != VT_BSTR || !value.bstrVal)
            throw std::runtime_error("expected COM text");
        int n = WideCharToMultiByte(CP_UTF8, 0, value.bstrVal,
                                    static_cast<int>(SysStringLen(value.bstrVal)), nullptr, 0,
                                    nullptr, nullptr);
        std::string result(static_cast<std::size_t>(n), ' ');
        WideCharToMultiByte(CP_UTF8, 0, value.bstrVal,
                            static_cast<int>(SysStringLen(value.bstrVal)), result.data(), n,
                            nullptr, nullptr);
        return result;
    }
    double number() const {
        Value out;
        checked(VariantChangeType(&out.value, const_cast<VARIANT *>(&value), 0, VT_R8));
        return out.value.dblVal;
    }
};
std::wstring wide(std::string_view s) {
    if (s.empty())
        return {};
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()),
                                nullptr, 0);
    if (!n)
        throw std::invalid_argument("invalid UTF-8");
    std::wstring w(n, L' ');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()),
                        w.data(), n);
    return w;
}
struct Dispatch {
    IDispatch *p = nullptr;
    std::map<std::wstring, DISPID> ids;
    explicit Dispatch(const wchar_t *name) {
        CLSID cls;
        checked(CLSIDFromProgID(name, &cls));
        checked(CoCreateInstance(cls, nullptr, CLSCTX_LOCAL_SERVER, IID_IDispatch,
                                 reinterpret_cast<void **>(&p)));
    }
    ~Dispatch() {
        if (p)
            p->Release();
    }
    Dispatch(const Dispatch &) = delete;
    Value call(const wchar_t *name, WORD flags, std::initializer_list<Value> args = {}) {
        auto found = ids.find(name);
        DISPID id;
        if (found == ids.end()) {
            LPOLESTR key = const_cast<wchar_t *>(name);
            checked(p->GetIDsOfNames(IID_NULL, &key, 1, LOCALE_USER_DEFAULT, &id));
            ids.emplace(name, id);
        } else
            id = found->second;
        std::vector<Value> owned(args.begin(), args.end());
        std::vector<VARIANT> reversed;
        reversed.reserve(owned.size());
        for (auto i = owned.rbegin(); i != owned.rend(); ++i)
            reversed.push_back(i->value);
        DISPID put = DISPID_PROPERTYPUT;
        DISPPARAMS parameters{reversed.data(), flags == DISPATCH_PROPERTYPUT ? &put : nullptr,
                              static_cast<UINT>(reversed.size()),
                              flags == DISPATCH_PROPERTYPUT ? 1u : 0u};
        Value result;
        EXCEPINFO exception{};
        UINT argument = 0;
        HRESULT hr = p->Invoke(id, IID_NULL, LOCALE_USER_DEFAULT, flags, &parameters, &result.value,
                               &exception, &argument);
        if (exception.pfnDeferredFillIn)
            exception.pfnDeferredFillIn(&exception);
        SysFreeString(exception.bstrSource);
        SysFreeString(exception.bstrDescription);
        SysFreeString(exception.bstrHelpFile);
        checked(hr);
        MSG message;
        while (PeekMessage(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessage(&message);
        }
        return result;
    }
    double get(const wchar_t *n, std::initializer_list<Value> a = {}) {
        return call(n, DISPATCH_PROPERTYGET, a).number();
    }
    void set(const wchar_t *n, std::initializer_list<Value> a) { call(n, DISPATCH_PROPERTYPUT, a); }
    void method(const wchar_t *n, std::initializer_list<Value> a = {}) {
        auto v = call(n, DISPATCH_METHOD, a);
        if (v.value.vt == VT_BOOL && !v.value.boolVal) {
            std::string message = "SIMNRA method returned false: ";
            for (auto c : std::wstring_view(n))
                message.push_back(static_cast<char>(c));
            if (std::wstring_view(n) == L"CalculateSpectrum" ||
                std::wstring_view(n) == L"CalculateSpectrumFast") {
                auto detail = call(L"LastMessage", DISPATCH_PROPERTYGET);
                if (detail.value.vt == VT_BSTR && detail.value.bstrVal) {
                    int size =
                        WideCharToMultiByte(CP_UTF8, 0, detail.value.bstrVal,
                                            static_cast<int>(SysStringLen(detail.value.bstrVal)),
                                            nullptr, 0, nullptr, nullptr);
                    std::string text(static_cast<std::size_t>(size), ' ');
                    WideCharToMultiByte(CP_UTF8, 0, detail.value.bstrVal,
                                        static_cast<int>(SysStringLen(detail.value.bstrVal)),
                                        text.data(), size, nullptr, nullptr);
                    message += "; " + text;
                }
            }
            throw std::runtime_error(message);
        }
    }
};
struct Apartment {
    Apartment(bool mta) {
        checked(CoInitializeEx(nullptr, mta ? COINIT_MULTITHREADED : COINIT_APARTMENTTHREADED));
    }
    ~Apartment() { CoUninitialize(); }
};
struct PrivateDirectory {
    std::filesystem::path path;
    ~PrivateDirectory() {
        std::error_code ec;
        if (!path.empty())
            std::filesystem::remove_all(path, ec);
    }
};
struct Session {
    PrivateDirectory directory;
    Dispatch app{L"SIMNRA.App"}, target{L"SIMNRA.Target"}, setup{L"SIMNRA.Setup"},
        spectrum{L"SIMNRA.Spectrum"}, calculation{L"SIMNRA.Calc"};
    std::optional<core::Sample> topology;
    std::optional<core::Measurement> configured_measurement;
    double reference_real_time = 0, reference_live_time = 0;
    explicit Session(const SimnraMethod &method) {
        static std::atomic_uint64_t sequence{0};
        auto root = std::filesystem::temp_directory_path();
        for (;;) {
            directory.path = root / ("ibeamlab-cpp-" + std::to_string(GetCurrentProcessId()) + "-" +
                                     std::to_string(sequence++));
            if (std::filesystem::create_directory(directory.path))
                break;
        }
        try {
            auto reference = directory.path / method.reference.filename();
            std::filesystem::copy_file(method.reference, reference);
            app.method(L"Open", {Value(reference.wstring()), Value(-1)});
            // This API returns total spectra only, as in the reference generator.
            calculation.set(L"ElementSpectra", {Value(false)});
            reference_real_time = setup.get(L"RealTime");
            reference_live_time = setup.get(L"LiveTime");
            // Reuse the loaded reference topology when it already matches.
            // Rebuilding it can reset per-element physics stored in the project.
            core::Sample loaded;
            int layers = static_cast<int>(target.get(L"NumberOfLayers"));
            if (layers < 1 || layers > 10000)
                throw std::runtime_error("invalid reference layer count");
            loaded.layers.resize(static_cast<std::size_t>(layers));
            for (int li = 1; li <= layers; ++li) {
                int elements = static_cast<int>(target.get(L"NumberOfElements", {Value(li)}));
                if (elements < 1 || elements > 256)
                    throw std::runtime_error("invalid reference element count");
                auto &species = loaded.layers[static_cast<std::size_t>(li - 1)].species;
                species.resize(static_cast<std::size_t>(elements));
                for (int ei = 1; ei <= elements; ++ei) {
                    auto name =
                        target.call(L"ElementName", DISPATCH_PROPERTYGET, {Value(li), Value(ei)})
                            .text();
                    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())))
                        name.pop_back();
                    auto &entry = species[static_cast<std::size_t>(ei - 1)];
                    entry.element = std::move(name);
                    const int count = static_cast<int>(target.get(L"NumberOfIsotopes", {Value(li), Value(ei)}));
                    if (count < 1 || count > 512) throw std::runtime_error("invalid reference isotope count");
                    for (int ii = 1; ii <= count; ++ii) {
                        auto mass = target.get(L"IsotopeMass", {Value(li), Value(ei), Value(ii)});
                        auto fraction = target.get(L"IsotopeConcentration", {Value(li), Value(ei), Value(ii)});
                        entry.isotopes.push_back({static_cast<std::int32_t>(std::lround(mass)), fraction});
                    }
                }
            }
            topology = std::move(loaded);
        } catch (...) {
            throw;
        }
    }

    void configure(const core::Experiment &x, const core::Measurement &m, bool reuse_topology) {
        bool rebuild =
            !reuse_topology || !topology || topology->layers.size() != x.sample.layers.size();
        if (!rebuild)
            for (std::size_t i = 0; i < x.sample.layers.size(); ++i) {
                auto &a = topology->layers[i].species;
                auto &b = x.sample.layers[i].species;
                if (a.size() != b.size()) {
                    rebuild = true;
                    break;
                }
                for (std::size_t j = 0; j < a.size(); ++j) {
                    if (a[j].element != b[j].element) { rebuild = true; break; }
                    const auto previous = core::resolved_isotopes(a[j]);
                    const auto requested = core::resolved_isotopes(b[j]);
                    if (previous.size() != requested.size() ||
                        !std::equal(previous.begin(), previous.end(), requested.begin(),
                                    [](auto &p, auto &r) { return p.massNumber == r.massNumber; })) {
                        rebuild = true;
                        break;
                    }
                }
            }
        // Keep every old element present until replacement layers are populated:
        // SIMNRA otherwise discards its selected cross sections for that element.
        const int old_layers = rebuild ? static_cast<int>(target.get(L"NumberOfLayers")) : 0;
        int li = 0;
        for (auto &l : x.sample.layers) {
            if (rebuild) {
                target.method(L"AddLayer");
                li = static_cast<int>(target.get(L"NumberOfLayers"));
                target.method(L"AddElements",
                              {Value(li), Value(static_cast<int>(l.species.size()))});
            } else {
                ++li;
            }
            const auto *previous = configured_measurement && !rebuild
                                       ? &topology->layers[static_cast<std::size_t>(li - 1)]
                                       : nullptr;
            if (!previous || previous->thickness != l.thickness)
                target.set(L"LayerThickness", {Value(li), Value(l.thickness)});
            int ei = 0;
            for (auto &s : l.species) {
                ++ei;
                if (rebuild)
                    target.set(L"ElementName", {Value(li), Value(ei), Value(wide(s.element))});
                {
                    const auto composition = core::resolved_isotopes(s);
                    // Fixed mixtures need no per-isotope COM calls on subsequent spectra.
                    if (configured_measurement && !rebuild && core::same_isotopes(composition,
                            core::resolved_isotopes(topology->layers[static_cast<std::size_t>(li - 1)]
                                .species[static_cast<std::size_t>(ei - 1)]), 0.0))
                        continue;
                    if (rebuild) {
                        int count = static_cast<int>(
                            target.get(L"NumberOfIsotopes", {Value(li), Value(ei)}));
                        while (count-- > 0)
                            target.method(L"DeleteIsotope", {Value(li), Value(ei), Value(1)});
                    }
                    int ii = 0;
                    for (const auto &i : composition) {
                        if (rebuild)
                            target.method(L"AddIsotope", {Value(li), Value(ei)});
                        ++ii;
                        if (rebuild || !configured_measurement) {
                            const auto records = core::isotope_data(s.element);
                            const auto mass = std::find_if(records.begin(), records.end(), [&](const auto &r) {
                                return r.massNumber == i.massNumber;
                            })->exactMass;
                            target.set(L"IsotopeMass", {Value(li), Value(ei), Value(ii), Value(mass)});
                        }
                        target.set(L"IsotopeConcentration",
                                   {Value(li), Value(ei), Value(ii), Value(i.fraction)});
                    }
                }
            }
            Value concentrations;
            concentrations.value.vt = VT_ARRAY | VT_R8;
            concentrations.value.parray =
                SafeArrayCreateVector(VT_R8, 1, static_cast<ULONG>(l.species.size()));
            if (!concentrations.value.parray)
                throw std::bad_alloc();
            double *fractions = nullptr;
            checked(SafeArrayAccessData(concentrations.value.parray,
                                        reinterpret_cast<void **>(&fractions)));
            for (std::size_t i = 0; i < l.species.size(); ++i)
                fractions[i] = l.species[i].concentration;
            checked(SafeArrayUnaccessData(concentrations.value.parray));
            target.set(L"ElementConcentrationArray", {Value(li), concentrations});
            Value rough;
            rough.value.vt = VT_BOOL;
            rough.value.boolVal = l.roughness > 0 ? VARIANT_TRUE : VARIANT_FALSE;
            if (!previous || (previous->roughness > 0) != (l.roughness > 0))
                target.set(L"HasLayerRoughness", {Value(li), rough});
            if (l.roughness > 0 && (!previous || previous->roughness != l.roughness))
                target.set(L"LayerRoughness", {Value(li), Value(l.roughness)});
            Value porous;
            porous.value.vt = VT_BOOL;
            porous.value.boolVal = l.porosityFraction > 0 ? VARIANT_TRUE : VARIANT_FALSE;
            if (!previous || (previous->porosityFraction > 0) != (l.porosityFraction > 0))
                target.set(L"HasLayerPorosity", {Value(li), porous});
            if (l.porosityFraction > 0) {
                if (!previous || previous->porosityFraction != l.porosityFraction)
                    target.set(L"PorosityFraction", {Value(li), Value(l.porosityFraction)});
                if (!previous || previous->poreDiameter != l.poreDiameter)
                    target.set(L"PoreDiameter", {Value(li), Value(l.poreDiameter)});
            }
        }
        if (rebuild) {
            // Only now remove the original prefix. New layers retain element
            // identities and their inherited cross-section selections.
            for (int i = 0; i < old_layers; ++i)
                target.method(L"DeleteLayer", {Value(1)});
        }
        // Fixed setup fields are common in large generation runs. Avoid a COM
        // round trip for each unchanged value on every spectrum.
        const auto &old = configured_measurement ? *configured_measurement : m;
        auto update = [&](const wchar_t *name, double value, double previous) {
            if (!configured_measurement || value != previous)
                setup.set(name, {Value(value)});
        };
        update(L"Energy", m.beam.energy, old.beam.energy);
        update(L"Beamspread", m.beam.spread, old.beam.spread);
        update(L"DetectorResolution", m.detector.resolution, old.detector.resolution);
        update(L"CalibrationLinear", m.detector.calibrationLinear, old.detector.calibrationLinear);
        update(L"CalibrationOffset", m.detector.calibrationOffset, old.detector.calibrationOffset);
        update(L"CalibrationQuadratic", m.detector.calibrationQuadratic,
               old.detector.calibrationQuadratic);
        update(L"ParticlesSr", m.acquisition.particlesSr, old.acquisition.particlesSr);
        update(L"RealTime",
               m.acquisition.realTime > 0 ? m.acquisition.realTime : reference_real_time,
               old.acquisition.realTime > 0 ? old.acquisition.realTime : reference_real_time);
        update(L"LiveTime",
               m.acquisition.liveTime > 0 ? m.acquisition.liveTime : reference_live_time,
               old.acquisition.liveTime > 0 ? old.acquisition.liveTime : reference_live_time);
        // Assignment reuses nested vector storage when topology is unchanged.
        topology = x.sample;
        configured_measurement = m;
    }
    void calculate(core::Counts &out, bool fast) {
        app.method(fast ? L"CalculateSpectrumFast" : L"CalculateSpectrum");
        auto data = spectrum.call(L"GetDataArray", DISPATCH_METHOD, {Value(2)});
        if ((data.value.vt != (VT_ARRAY | VT_R8) && data.value.vt != (VT_ARRAY | VT_VARIANT)) ||
            !data.value.parray || SafeArrayGetDim(data.value.parray) != 1)
            throw std::runtime_error("invalid SIMNRA spectrum array");
        LONG lo, hi;
        checked(SafeArrayGetLBound(data.value.parray, 1, &lo));
        checked(SafeArrayGetUBound(data.value.parray, 1, &hi));
        if (hi < lo || static_cast<long long>(hi) - lo > 1000000)
            throw std::runtime_error("invalid SIMNRA spectrum length");
        out.resize(static_cast<std::size_t>(hi - lo + 1));
        for (LONG i = lo; i <= hi; ++i) {
            double value;
            if (data.value.vt == (VT_ARRAY | VT_R8))
                checked(SafeArrayGetElement(data.value.parray, &i, &value));
            else {
                Value v;
                checked(SafeArrayGetElement(data.value.parray, &i, &v.value));
                value = v.number();
            }
            out[i - lo] = value;
        }
        core::validate_counts(out);
    }
};
} // namespace
struct SimnraSimulator::Impl {
    SimnraConfig config;
    SimnraSimulator *owner;
    std::mutex mutex;
    std::condition_variable cv;
    bool closed = false;
    using Sessions = std::vector<std::unique_ptr<Session>>;
    std::deque<std::function<void(Sessions &, std::exception_ptr)>> tasks;
    std::vector<std::thread> workers;
    std::string provenance;
    Impl(SimnraConfig c, SimnraSimulator *o) : config(std::move(c)), owner(o) {
        if (!config.workers || config.workers > 256 || config.methods.empty())
            throw std::invalid_argument("invalid SIMNRA worker configuration");
        std::vector<std::string> labels;
        provenance = std::string("simnra-com-v1;calculation=") +
                     (config.fast_calculation ? "fast" : "standard") +
                     ";workers=" + std::to_string(config.workers);
        for (auto &m : config.methods) {
            if (m.measurement.empty() || m.particle.empty() ||
                std::find(labels.begin(), labels.end(), m.measurement) != labels.end())
                throw std::invalid_argument("invalid SIMNRA measurement");
            labels.push_back(m.measurement);
            std::ifstream f(m.reference, std::ios::binary);
            if (!f)
                throw std::invalid_argument("missing SIMNRA reference");
            std::vector<std::byte> b;
            char ch;
            while (f.get(ch))
                b.push_back(std::byte(static_cast<unsigned char>(ch)));
            provenance += ";" + m.measurement + ":" + std::to_string(detail::checksum(b));
        }
        try {
            for (std::size_t i = 0; i < config.workers; ++i)
                workers.emplace_back([this] {
                    std::unique_ptr<Apartment> apartment;
                    std::exception_ptr apartment_error;
                    try {
                        apartment = std::make_unique<Apartment>(config.multithreaded_apartment);
                    } catch (...) {
                        apartment_error = std::current_exception();
                    }
                    Sessions sessions;
                    for (;;) {
                        std::function<void(Sessions &, std::exception_ptr)> task;
                        {
                            std::unique_lock lock(mutex);
                            cv.wait(lock, [&] { return closed || !tasks.empty(); });
                            if (tasks.empty() && closed)
                                break;
                            task = std::move(tasks.front());
                            tasks.pop_front();
                        }
                        task(sessions, apartment_error);
                    }
                });
        } catch (...) {
            close();
            throw;
        }
    }
    std::future<Result> submit(const core::Experiment &x) {
        auto promise = std::make_shared<std::promise<Result>>();
        auto future = promise->get_future();
        std::unique_lock lock(mutex);
        if (closed)
            throw std::logic_error("SIMNRA backend closed");
        tasks.emplace_back(
            [this, promise, &x](Sessions &sessions, std::exception_ptr apartment_error) {
                Result out;
                std::string label;
                try {
                    if (apartment_error)
                        std::rethrow_exception(apartment_error);
                    // COM apartments are re-entered on the same worker; session ownership stays
                    // thread-local.
                    x.validate();
                    if (x.setup.measurements.size() != config.methods.size())
                        throw std::invalid_argument("SIMNRA reference count mismatch");
                    if (sessions.empty())
                        sessions.resize(config.methods.size());
                    for (const auto &measurement : x.setup.measurements) {
                        if (owner->stopped())
                            throw std::runtime_error("simulation cancelled");
                        auto found = std::find_if(
                            config.methods.begin(), config.methods.end(),
                            [&](const auto &m) { return m.measurement == measurement.label; });
                        if (found == config.methods.end())
                            throw std::invalid_argument("missing SIMNRA reference");
                        auto i = static_cast<std::size_t>(found - config.methods.begin());
                        auto &m = *found;
                        label = m.measurement;
                        if (measurement.beam.particle != m.particle)
                            throw std::invalid_argument(
                                "particle differs from declared SIMNRA reference");
                        if (!sessions[i])
                            sessions[i] = std::make_unique<Session>(m);
                        sessions[i]->configure(x, measurement, config.reuse_target_topology);
                        out.spectra.push_back({label, {}});
                        sessions[i]->calculate(out.spectra.back().counts, config.fast_calculation);
                    }
                } catch (const std::exception &e) {
                    out.failure = Failure{label, e.what()};
                    sessions.clear();
                }
                promise->set_value(std::move(out));
            });
        lock.unlock();
        cv.notify_one();
        return future;
    }
    void close() noexcept {
        {
            std::lock_guard lock(mutex);
            closed = true;
        }
        cv.notify_all();
        for (auto &w : workers)
            if (w.joinable())
                w.join();
        workers.clear();
    }
    ~Impl() { close(); }
};
SimnraSimulator::SimnraSimulator(SimnraConfig c)
    : impl_(std::make_unique<Impl>(std::move(c), this)) {}
SimnraSimulator::~SimnraSimulator() = default;
void SimnraSimulator::simulate(const core::Experiment &x, Result &out) {
    out = impl_->submit(x).get();
}
std::vector<Result> SimnraSimulator::simulate_batch(std::span<const core::Experiment> inputs,
                                                    Progress progress) {
    std::vector<std::future<Result>> jobs;
    for (auto &x : inputs) {
        if (stopped())
            break;
        jobs.push_back(impl_->submit(x));
    }
    std::vector<Result> out;
    out.reserve(jobs.size());
    std::exception_ptr error;
    for (auto &job : jobs) {
        out.push_back(job.get());
        if (progress && !error)
            try {
                progress(out.size(), inputs.size());
            } catch (...) {
                error = std::current_exception();
                request_stop();
            }
    }
    if (error)
        std::rethrow_exception(error);
    return out;
}
std::string SimnraSimulator::provenance() const { return impl_->provenance; }
void SimnraSimulator::close() noexcept {
    request_stop();
    impl_->close();
}
} // namespace ibeamlab::simulation
#else
namespace ibeamlab::simulation {
struct SimnraSimulator::Impl {};
SimnraSimulator::SimnraSimulator(SimnraConfig) {
    throw std::runtime_error("SIMNRA backend not enabled in this build");
}
SimnraSimulator::~SimnraSimulator() = default;
void SimnraSimulator::simulate(const core::Experiment &, Result &) {
    throw std::runtime_error("SIMNRA unavailable");
}
std::vector<Result> SimnraSimulator::simulate_batch(std::span<const core::Experiment>, Progress) {
    throw std::runtime_error("SIMNRA unavailable");
}
std::string SimnraSimulator::provenance() const { return "simnra-disabled"; }
void SimnraSimulator::close() noexcept {}
} // namespace ibeamlab::simulation
#endif
