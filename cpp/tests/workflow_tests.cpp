#include <fstream>
#include <cstdlib>
#include <ibeamlab/generation/generator.h>
#include <ibeamlab/inference/models.h>
#include <ibeamlab/model/metadata.h>
#include <ibeamlab/processing/pileup.h>
#include <ibeamlab/processing/rebin.h>
#include <ibeamlab/processing/transforms.h>
#include <ibeamlab/simulation/simnra.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
using namespace ibeamlab;
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
void near(double a, double b, double tolerance = 1e-6) {
    require(std::abs(a - b) <= tolerance * std::max(1., std::abs(b)), "numerical mismatch");
}
template <class F> void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "expected rejection");
}
core::Experiment experiment() {
    core::Experiment x;
    core::Layer l;
    l.thickness = 100;
    l.species = {{"Si", .5, {}}, {"O", .5, {}}};
    x.sample.layers = {l};
    core::Measurement m;
    m.label = "RBS";
    m.beam = {"H", 2000, 0};
    m.acquisition = {1000, 10, 8};
    x.setup.measurements = {m};
    x.validate();
    return x;
}
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        for (;;) {
            path = std::filesystem::temp_directory_path() /
                   ("ibeamlab-test-" + std::to_string(std::random_device{}()));
            if (std::filesystem::create_directory(path))
                break;
        }
    }
    ~Temporary() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};
void processing_tests() {
    std::vector<float> tensor{50, 100};
    model::TensorPipeline scalers{model::MinMaxScale{{0}, {100}}, model::StandardScale{{.5}, {.5}},
                                  model::MultiplyScale{{2}}};
    processing::apply_preprocessing(tensor, scalers);
    near(tensor[0], 0);
    near(tensor[1], 2);
    model::TransformPipeline reversible{{model::Log1p{}, model::StandardScale{{1}, {2}}}};
    processing::TransformPlan encode(reversible, 2);
    reversible.direction = model::TransformDirection::Inverse;
    processing::TransformPlan decode(reversible, 2);
    reversible.steps.clear(); // Plans own their coefficients independently of metadata lifetime.
    core::Counts roundtrip{3, 17};
    encode.apply(roundtrip);
    decode.apply(roundtrip);
    near(roundtrip[0], 3);
    near(roundtrip[1], 17);
    processing::register_custom_transform(
        "test.add", 1, [](std::span<float> values, const model::CustomParameters &parameters) {
            for (auto &value : values)
                value += static_cast<float>(parameters.at("offset")[0]);
        });
    processing::apply_preprocessing(tensor,
                                    {model::CustomTransform{"test.add", 1, {{"offset", {3}}}}});
    near(tensor[0], 3);
    near(tensor[1], 5);
    rejects([&] {
        processing::require_supported_transforms({model::CustomTransform{"missing", 1, {}}});
    });
    rejects([&] { processing::apply_preprocessing(tensor, {model::StandardScale{{0}, {0}}}); });
    model::SpectraSchema schema{{{"RBS", 4}, {"NRA", 3}}};
    require(schema.width() == 7 && schema.offset("NRA") == 4, "packed spectrum mapping");
    rejects([&] { schema.offset("unknown"); });

    core::Counts x{2, 4}, edges{-.5, .5, 1.5}, wide{-.5, 1.5};
    auto y = processing::rebin(x, edges, wide);
    near(y[0], 6);
    auto identity = processing::rebin(x, edges, edges);
    require(identity == x, "identity rebin");
    core::Counts partial{0, 1};
    auto split = processing::rebin(x, edges, partial);
    near(split[0], 3);
    core::Counts outside{-1, 2};
    rejects([&] { processing::rebin(x, edges, outside, processing::Coverage::RequireFull); });
    rejects([&] { processing::rebin(x, edges, edges, x); });
    core::Counts bad{0, 0, 2};
    rejects([&] { processing::rebin(x, bad, edges); });
    auto d = experiment().setup.measurements[0].detector;
    core::Counts calibrated(3);
    core::channel_edges(d, calibrated);
    require(calibrated == edges, "edge convention");
    d.calibrationLinear = -1;
    rejects([&] { core::channel_edges(d, calibrated); });
    processing::PileupWorkspace workspace;
    core::Counts input{2, 4, 1}, piled(5), recovered(3);
    double tau = .01, t = 10, l = 8;
    processing::apply_pileup(input, piled, t, l, tau, workspace);
    double total = 7, a = tau / t * std::exp(-tau * total / t);
    for (std::size_t i = 0; i < piled.size(); ++i) {
        double convolution = 0;
        for (std::size_t j = 0; j < input.size(); ++j)
            if (i >= j && i - j < input.size())
                convolution += input[j] * input[i - j];
        near(piled[i],
             l / t * ((i < input.size() ? input[i] * (1 - 2 * a * total) : 0) + a * convolution));
    }
    processing::remove_pileup(piled, recovered, t, l, tau);
    for (std::size_t i = 0; i < input.size(); ++i)
        near(recovered[i], input[i]);
    rejects([&] { processing::apply_pileup(input, piled, t, 11, tau, workspace); });
    for (double occupancy : {.1, 1., 2., 7.}) {
        const double shaping = occupancy * t / total;
        processing::apply_pileup(input, piled, t, l, shaping, workspace);
        const double factor = shaping / t * std::exp(-occupancy);
        for (std::size_t i = 0; i < piled.size(); ++i) {
            double convolution = 0;
            for (std::size_t j = 0; j < input.size(); ++j)
                if (i >= j && i - j < input.size()) convolution += input[j] * input[i - j];
            const double original = i < input.size() ? input[i] : 0;
            near(piled[i], l / t * (original - 2 * factor * total * original + factor * convolution));
        }
        processing::remove_pileup(piled, recovered, t, l, shaping);
        for (std::size_t i = 0; i < input.size(); ++i) near(recovered[i], input[i]);
    }
    core::Counts zero(3), zero_piled(5);
    processing::apply_pileup(zero, zero_piled, t, l, 0, workspace);
    processing::remove_pileup(zero_piled, recovered, t, l, 0);
    require(recovered == zero, "zero inverse");
}
void generation_tests(const std::filesystem::path &root) {
    generation::GenerationPlan plan{experiment(),
                                    {{"thickness", core::LayerThickness{0}, "1e15 atoms/cm2"}}};
    simulation::DummySimulator sim(32);
    core::Counts rows{100, 200, 300};
    auto result = generation::generate(plan, rows, 3, sim, root / "dataset",
                                       generation::FailurePolicy::Stop, {}, "seed=42", 1);
    require(result.accepted == 3, "accepted records");
    datasets::Reader reader(root / "dataset");
    require(reader.metadata().provenance.find("seed=42") != std::string::npos,
            "provenance missing");
    datasets::Record record;
    std::size_t n = 0;
    while (reader.next(record)) {
        require(record.sample_index == n && record.parameters[0] == rows[n], "record order");
        simulation::Result expected;
        auto x = plan.reference;
        x.sample.layers[0].thickness = rows[n];
        sim.simulate(x, expected);
        require(expected.spectra[0].counts == record.result.spectra[0].counts,
                "deterministic spectra");
        ++n;
    }
    require(n == 3, "record count");
    reader.rewind();
    require(reader.next(record) && record.sample_index == 0, "rewind");
    require(reader.maximum_spectrum_lengths().at("RBS") == 32, "maximum spectrum length");
    require(reader.next(record) && record.sample_index == 1, "length scan preserves cursor");
    {
        datasets::Metadata variable;
        variable.reference = experiment();
        auto second = variable.reference.setup.measurements.front();
        second.label = "second";
        variable.reference.setup.measurements.push_back(second);
        variable.requested = 4;
        datasets::Writer writer(root / "variable-lengths", variable, 1);
        datasets::Record row;
        for (std::size_t i = 0; i < 4; ++i) {
            row.sample_index = i;
            row.sample_id = std::to_string(i);
            row.result.failure.reset();
            if (i == 1) {
                row.result.spectra.clear();
                row.result.failure = simulation::Failure{"RBS", "test failure"};
            } else {
                row.result.spectra = {{"RBS", core::Counts(i == 2 ? 11 : 2, 1)},
                                      {"second", core::Counts(i == 0 ? 9 : 3, 1)}};
            }
            writer.append(row);
        }
        writer.finish();
        const auto lengths = datasets::Reader(root / "variable-lengths").maximum_spectrum_lengths();
        require(lengths.at("RBS") == 11 && lengths.at("second") == 9,
                "independent detector maxima across shards and failures");
    }
    rejects([&] { generation::generate(plan, rows, 3, sim, root / "dataset"); });
    core::Counts invalid{100, -1, 200};
    auto recorded = generation::generate(plan, invalid, 3, sim, root / "record",
                                         generation::FailurePolicy::Record);
    require(recorded.accepted == 2 && recorded.failed == 1, "record policy");
    datasets::Reader failures(root / "record");
    require(failures.next(record) && failures.next(record) && record.result.failure.has_value(),
            "stored failure");
    auto discarded = generation::generate(plan, invalid, 3, sim, root / "discard",
                                          generation::FailurePolicy::Discard);
    require(discarded.accepted == 2 && discarded.discarded == 1, "discard policy");
    rejects([&] { generation::generate(plan, invalid, 3, sim, root / "stop"); });
    rejects([&] { datasets::Reader incomplete(root / "stop"); });
    auto cancelled =
        generation::generate(plan, rows, 3, sim, root / "cancel", generation::FailurePolicy::Stop,
                             [&](std::size_t done, std::size_t) {
                                 if (done == 1)
                                     sim.request_stop();
                             });
    require(cancelled.cancelled, "cancel flag");
    rejects([&] { datasets::Reader incomplete(root / "cancel"); });
    sim.reset_stop();
    // The generator must submit bounded batches rather than serial simulate calls.
    struct BatchProbe final : simulation::Simulator {
        std::vector<std::size_t> sizes;
        simulation::DummySimulator backend{32};
        void simulate(const core::Experiment &, simulation::Result &) override {
            throw std::logic_error("generator used serial simulation");
        }
        std::string provenance() const override { return "batch-probe"; }
        std::vector<simulation::Result> simulate_batch(std::span<const core::Experiment> inputs,
                                                       simulation::Progress progress) override {
            sizes.push_back(inputs.size());
            std::vector<simulation::Result> results(inputs.size());
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                backend.simulate(inputs[i], results[i]);
                if (progress)
                    progress(i + 1, inputs.size());
            }
            return results;
        }
    } batched;
    core::Counts batch_rows{100, 200, 300, 400, 500};
    auto batched_summary = generation::generate(plan, batch_rows, 5, batched, root / "batched",
                                                generation::FailurePolicy::Stop, {}, "", 3, 2);
    require(batched_summary.accepted == 5 && batched.sizes == std::vector<std::size_t>{2, 2, 1},
            "bounded backend batches");
    datasets::Reader batch_reader(root / "batched");
    std::size_t batch_index = 0;
    while (batch_reader.next(record)) {
        require(record.sample_index == batch_index &&
                    record.parameters[0] == batch_rows[batch_index],
                "batch source row identity");
        ++batch_index;
    }
    require(batch_index == 5, "batch result count");
    rejects([&] {
        generation::generate(plan, batch_rows, 5, batched, root / "zero-batch",
                             generation::FailurePolicy::Stop, {}, "", 3, 0);
    });
    {
        std::fstream f(root / "dataset" / "shard-0.ibd",
                       std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(25);
        char c = 0;
        f.read(&c, 1);
        c ^= 1;
        f.seekp(25);
        f.write(&c, 1);
    }
    rejects([&] {
        reader.rewind();
        reader.next(record);
    });
    core::Counts a(10), b(10);
    generation::Uniform range[]{{{1, 2}}};
    generation::sample_uniform(range, 10, 123, a);
    generation::sample_uniform(range, 10, 123, b);
    require(a == b, "sampling seed");
    for (double v : a)
        require(v >= 1 && v <= 2, "sampling bounds");
}
model::ParameterInputSchema &parameter_input(model::ForwardModelMetadata &f) {
    return std::get<model::ParameterInputSchema>(f.input_preparation);
}
core::Experiment reference(const model::ForwardModelMetadata &f) {
    return {std::get<model::ParameterInputSchema>(f.input_preparation).reference_sample, f.reference_setup};
}
model::ModelMetadata metadata(model::Direction direction) {
    if (direction == model::Direction::Forward) {
        model::ForwardModelMetadata forward;
        forward.reference_setup = experiment().setup;
        parameter_input(forward).reference_sample = experiment().sample;
        parameter_input(forward).parameters = {{"thickness", core::LayerThickness{0}, "1e15 atoms/cm2"}};
        forward.output_spectra.spectra = {{"RBS", 4}};
        forward.output_preparation = {{"RBS", {model::Rebin{}}}};
        return model::ModelMetadata(std::move(forward));
    }
    model::InverseModelMetadata inverse;
    inverse.reference_setup = experiment().setup;
    inverse.input_spectra.spectra = {{"RBS", 4}};
    inverse.input_preparation = {{"RBS", {model::Rebin{}}}};
    inverse.output_preparation.elements = {"Si", "O"};
    inverse.output_preparation.layers = 1;
    inverse.output_preparation.uncertainty = true;
    return model::ModelMetadata(std::move(inverse));
}
#ifdef IBEAMLAB_TEST_ONNX
void forward_contract_tests() {
    auto m = metadata(model::Direction::Forward);
    auto &f = std::get<model::ForwardModelMetadata>(m.model);
    f.output_preparation.clear();
    parameter_input(f).reference_sample.layers[0].species[0].isotopes = {{28, 28, 1}};
    auto package =
        model::Package::from_onnx(std::filesystem::path(IBEAMLAB_FIXTURES) / "forward.onnx", m);
    inference::ForwardModel forward(package);
    inference::InferenceOptions cpu_options;
    cpu_options.intra_op_threads = 2;
    cpu_options.enable_graph_optimizations = false;
    inference::ForwardModel tuned(package, cpu_options);
    require(tuned.options().intra_op_threads == 2 && !tuned.options().enable_graph_optimizations,
            "C++ inference options snapshot");
    const auto available = inference::available_execution_providers();
    inference::InferenceOptions cuda_options;
    cuda_options.execution_provider = inference::ExecutionProvider::Cuda;
    if (std::getenv("IBEAMLAB_TEST_CUDA")) {
        require(std::find(available.begin(), available.end(), "cuda") != available.end(), "CUDA runtime required");
        inference::ForwardModel gpu(package, cuda_options);
        near(gpu.predict(reference(f)).spectra[0].counts.back(), 400);
        inference::InverseModel inverse_gpu(model::Package::from_onnx(
            std::filesystem::path(IBEAMLAB_FIXTURES) / "inverse.onnx", metadata(model::Direction::Inverse)), cuda_options);
        inference::Prediction prediction;
        const std::vector<core::Spectrum> spectra{{"RBS", {1, 2, 3, 4}}};
        inverse_gpu.predict_reference(spectra, prediction);
        near(prediction.values[0], 10);
    } else if (std::find(available.begin(), available.end(), "cuda") == available.end()) {
        rejects([&] { inference::ForwardModel unavailable(package, cuda_options); });
    }
    auto reference = ::reference(f);
    simulation::Result output;
    forward.predict(reference, output);
    const auto *storage = output.spectra[0].counts.data();
    auto changed = reference;
    changed.sample.layers[0].thickness = 125;
    changed.setup.measurements[0].detector.info = "display-only information";
    forward.predict(changed, output);
    near(output.spectra[0].counts[3], 500);
    require(output.spectra[0].counts.data() == storage, "forward result buffer reuse");

    // Every immutable part of the experiment must still be checked without
    // serializing it. Each case starts from the same valid reference.
    auto invalid = [&](auto mutate) {
        auto x = reference;
        mutate(x);
        rejects([&] { forward.predict(x, output); });
        forward.predict(reference, output);
        near(output.spectra[0].counts[0], 100);
    };
    invalid([](auto &x) { x.sample.layers.clear(); });
    invalid([](auto &x) { x.sample.layers.push_back(x.sample.layers[0]); });
    invalid([](auto &x) { x.sample.layers[0].roughness = 1; });
    invalid([](auto &x) { x.sample.layers[0].porosityFraction = .1; });
    invalid([](auto &x) { x.sample.layers[0].poreDiameter = 1; });
    invalid([](auto &x) { x.sample.layers[0].species.pop_back(); });
    invalid([](auto &x) {
        std::reverse(x.sample.layers[0].species.begin(), x.sample.layers[0].species.end());
    });
    invalid([](auto &x) { x.sample.layers[0].species[0].element = "C"; });
    invalid([](auto &x) { x.sample.layers[0].species[0].concentration = .4; });
    invalid([](auto &x) {
        x.sample.layers[0].species[0].concentration = .4;
        x.sample.layers[0].species[1].concentration = .6;
    });
    invalid([](auto &x) { x.sample.layers[0].species[0].isotopes.clear(); });
    invalid([](auto &x) { x.sample.layers[0].species[0].isotopes[0].massNumber = 29; });
    invalid([](auto &x) { x.sample.layers[0].species[0].isotopes[0].exactMass = 29; });
    invalid([](auto &x) { x.sample.layers[0].species[0].isotopes[0].fraction = .5; });
    invalid([](auto &x) { x.setup.measurements.clear(); });
    invalid([](auto &x) { x.setup.measurements[0].label = "unknown"; });
    invalid([](auto &x) { x.setup.measurements[0].beam.particle = "He"; });
    invalid([](auto &x) { x.setup.measurements[0].beam.energy = 3000; });
    invalid([](auto &x) { x.setup.measurements[0].beam.spread = 1; });
    invalid([](auto &x) { x.setup.measurements[0].detector.resolution = 1; });
    invalid([](auto &x) { x.setup.measurements[0].detector.calibrationLinear = 2; });
    invalid([](auto &x) { x.setup.measurements[0].detector.calibrationOffset = 1; });
    invalid([](auto &x) { x.setup.measurements[0].detector.calibrationQuadratic = .1; });
    invalid([](auto &x) { x.setup.measurements[0].acquisition.particlesSr = 2000; });
    invalid([](auto &x) { x.setup.measurements[0].acquisition.realTime = 11; });
    invalid([](auto &x) { x.setup.measurements[0].acquisition.liveTime = 9; });
    invalid([](auto &x) { x.sample.layers[0].thickness = -1; });
    invalid(
        [](auto &x) { x.sample.layers[0].thickness = std::numeric_limits<double>::quiet_NaN(); });

    // Resolve all supported target types, including coupled compositions and
    // deliberately non-structural input column order.
    parameter_input(f).parameters = {{"energy", core::BeamEnergy{"RBS"}, "keV"},
                          {"oxygen", core::SpeciesConcentration{0, "O"}, ""},
                          {"linear", core::CalibrationLinear{"RBS"}, ""},
                          {"thickness", core::LayerThickness{0}, ""},
                          {"resolution", core::DetectorResolution{"RBS"}, ""},
                          {"offset", core::CalibrationOffset{"RBS"}, ""},
                          {"silicon", core::SpeciesConcentration{0, "Si"}, ""},
                          {"spread", core::BeamSpread{"RBS"}, ""},
                          {"exposure", core::ParticlesSr{"RBS"}, ""},
                          {"quadratic", core::CalibrationQuadratic{"RBS"}, ""}};
    // Distinct coefficients expose incorrect column placement in the graph.
    f.input_transform.steps = {model::MultiplyScale{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}};
    inference::ForwardModel varied(model::Package::from_onnx(
        std::filesystem::path(IBEAMLAB_FIXTURES) / "forward_parameters.onnx", m));
    auto x = reference;
    const core::Counts values{2100, .25, 2, 123, 4, 5, .75, 6, 1200, .01};
    double expected = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        core::write_parameter(x, parameter_input(f).parameters[i].target, values[i]);
        expected += values[i] * (i + 1);
    }
    near(varied.predict(x).spectra[0].counts[0], expected);
    auto bad = x;
    bad.sample.layers[0].species[0].concentration = -.25;
    bad.sample.layers[0].species[1].concentration = 1.25;
    rejects([&] { varied.predict(bad); });
    bad = x;
    bad.setup.measurements[0].beam.energy = 0;
    rejects([&] { varied.predict(bad); });
    bad = x;
    bad.setup.measurements[0].detector.calibrationOffset = std::numeric_limits<double>::infinity();
    rejects([&] { varied.predict(bad); });
    auto negative = package;
    std::get<model::ForwardModelMetadata>(negative.metadata.model).input_transform.steps = {
        model::MultiplyScale{{-1}}};
    inference::ForwardModel negative_output(negative);
    rejects([&] { negative_output.predict(reference); });
}
#endif
void package_tests(const std::filesystem::path &root) {
    inference::InferenceOptions tuning;
    tuning.validate();
    tuning.device_id = -1;
    rejects([&] { tuning.validate(); });
    tuning = {};
    tuning.intra_op_threads = 0;
    rejects([&] { tuning.validate(); });
    tuning = {};
    tuning.inter_op_threads = 0;
    rejects([&] { tuning.validate(); });
    tuning = {};
    tuning.execution_provider = static_cast<inference::ExecutionProvider>(100);
    rejects([&] { tuning.validate(); });
    model::Package package;
    package.metadata = metadata(model::Direction::Forward);
    package.graph = {std::byte{1}, std::byte{2}};
    package.write(root / "model.ibm");
    auto opened = model::Package::open(root / "model.ibm");
    require(opened.metadata.direction() == model::Direction::Forward, "forward variant direction");
    require(parameter_input(std::get<model::ForwardModelMetadata>(opened.metadata.model)).parameters[0].name ==
                "thickness",
            "forward input schema round trip");
    {
        std::ifstream file(root / "model.ibm", std::ios::binary);
        char magic[8];
        file.read(magic, 8);
        require(std::string_view(magic, 8) == "IBMODEL1", "package must use format 1");
    }
    auto inverse_storage = package;
    inverse_storage.metadata = metadata(model::Direction::Inverse);
    std::get<model::InverseModelMetadata>(inverse_storage.metadata.model).input_transform = {
        {"RBS", {{model::Log1p{}, model::StandardScale{{1, 2, 3, 4}, {2}}}}}};
    inverse_storage.write(root / "inverse-storage.ibm");
    auto loaded_inverse = model::Package::open(root / "inverse-storage.ibm");
    require(loaded_inverse.metadata.direction() == model::Direction::Inverse,
            "inverse variant direction");
    const auto &inverse_spec = std::get<model::InverseModelMetadata>(loaded_inverse.metadata.model);
    require(inverse_spec.output_preparation.elements == std::vector<std::string>{"Si", "O"} &&
                inverse_spec.output_preparation.layers == 1 && inverse_spec.output_preparation.uncertainty,
            "inverse EDP schema round trip");
    require(std::holds_alternative<model::Log1p>(inverse_spec.input_transform[0].transform.steps[0]) &&
                std::get<model::StandardScale>(inverse_spec.input_transform[0].transform.steps[1]).mean ==
                    core::Counts{1, 2, 3, 4},
            "ordered preprocessing round trip");
    auto switching = metadata(model::Direction::Forward);
    switching.model = inverse_spec;
    require(switching.direction() == model::Direction::Inverse && switching.input_width() == 4 &&
                switching.output_width() == 6,
            "direction follows active variant");
    switching.validate();
    std::get<model::InverseModelMetadata>(switching.model).output_preparation.elements = {"Si", "Si"};
    rejects([&] { switching.validate(); });
    rejects([&] { inference::ForwardModel wrong_direction(inverse_storage); });

    require(opened.graph == package.graph && opened.metadata.input_width() == 1,
            "package round trip");
    auto layer_package = package;
    auto &layer_forward = std::get<model::ForwardModelMetadata>(layer_package.metadata.model);
    layer_forward.input_preparation = model::LayerInputSchema{{"Si", "O"}};
    layer_forward.input_transform.steps = {model::MultiplyScale{{.01, .02}}};
    layer_package.write(root / "layers-storage.ibm");
    auto layer_loaded = model::Package::open(root / "layers-storage.ibm");
    require(layer_loaded.metadata.layer_input() && layer_loaded.metadata.input_rank() == 3 &&
                layer_loaded.metadata.input_width() == 2, "layer input rank and width round trip");
    const auto &layer_schema = std::get<model::LayerInputSchema>(
        std::get<model::ForwardModelMetadata>(layer_loaded.metadata.model).input_preparation);
    require(layer_schema.elements == std::vector<std::string>{"Si", "O"} &&
                std::get<model::MultiplyScale>(std::get<model::ForwardModelMetadata>(layer_loaded.metadata.model).input_transform.steps[0]).factors == core::Counts{.01, .02},
            "layer element order and per-element preprocessing round trip");
    auto &invalid_layers = std::get<model::LayerInputSchema>(layer_forward.input_preparation);
    invalid_layers.elements = {"Si", "Si"};
    rejects([&] { layer_package.validate(); });
    invalid_layers.elements = {"Si", "O"};
    layer_forward.input_transform.steps = {model::MultiplyScale{{1, 2, 3}}};
    rejects([&] { layer_package.validate(); });
    rejects([&] { package.write(root / "model.ibm"); });
    std::get<model::ForwardModelMetadata>(opened.metadata.model).input_transform.steps = {
        model::StandardScale{{0}, {0}}};
    rejects([&] { opened.validate(); });
    {
        std::fstream f(root / "model.ibm", std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(-1, std::ios::end);
        char c = 5;
        f.write(&c, 1);
    }
    rejects([&] { model::Package::open(root / "model.ibm"); });
#ifdef IBEAMLAB_TEST_ONNX
    forward_contract_tests();
    auto forward_package =
        model::Package::from_onnx(std::filesystem::path(IBEAMLAB_FIXTURES) / "forward.onnx",
                                  metadata(model::Direction::Forward));
    inference::ForwardModel forward(forward_package);
    auto x = experiment();
    auto out = forward.predict(x);
    require(out.spectra[0].counts.size() == 4, "forward width");
    for (int i = 0; i < 4; ++i)
        near(out.spectra[0].counts[i], 100 * (i + 1));
    std::vector<core::Experiment> forward_batch{x, x};
    auto batch_results = forward.predict_batch(forward_batch);
    require(batch_results.size() == 2 &&
                batch_results[1].spectra[0].counts == out.spectra[0].counts,
            "forward batch parity");
    auto changed = x;
    changed.setup.measurements[0].beam.energy += 1;
    rejects([&] { forward.predict(changed); });
    std::get<model::ForwardModelMetadata>(forward_package.metadata.model)
        .output_preparation[0]
        .steps.insert(std::get<model::ForwardModelMetadata>(forward_package.metadata.model)
                          .output_preparation[0]
                          .steps.begin(),
                      model::ExposureScale{});
    inference::ForwardModel corrected(forward_package);
    changed = x;
    changed.setup.measurements[0].acquisition.particlesSr *= 2;
    near(corrected.predict(changed).spectra[0].counts[0], 200);
    auto inverse_package =
        model::Package::from_onnx(std::filesystem::path(IBEAMLAB_FIXTURES) / "inverse.onnx",
                                  metadata(model::Direction::Inverse));
    inverse_package.write(root / "inverse.ibm");
    inference::InverseModel inverse(model::Package::open(root / "inverse.ibm"));
    inference::InverseInput input{x.setup, {{"RBS", {1, 2, 3, 4}}}, false};
    auto prediction = inverse.predict(input);
    near(prediction.values[0], 10);
    near(prediction.values[1], 20);
    near(prediction.presence_probability[0], .8);
    near(prediction.posterior_std[1], .3);
    require(prediction.uncertainty_predicted, "uncertainty metadata");
    std::vector<inference::InverseInput> inverse_batch{input, input};
    auto batch_predictions = inverse.predict_batch(inverse_batch);
    require(batch_predictions.size() == 2 && batch_predictions[1].values == prediction.values,
            "inverse batch parity");
    inference::Prediction prepared;
    inverse.predict_reference(input.spectra, prepared);
    require(prediction.values == prepared.values, "prepared parity");
    auto exposure_input = input;
    exposure_input.setup.measurements[0].acquisition.particlesSr *= 2;
    for (auto &v : exposure_input.spectra[0].counts)
        v *= 2;
    std::get<model::InverseModelMetadata>(inverse_package.metadata.model)
        .input_preparation[0]
        .steps.push_back(model::ExposureScale{});
    inference::InverseModel exposure_inverse(inverse_package);
    near(exposure_inverse.predict(exposure_input).values[0], prediction.values[0]);
    auto calibrated_input = input;
    calibrated_input.setup.measurements[0].detector.calibrationLinear = .5;
    calibrated_input.setup.measurements[0].detector.calibrationOffset = -.25;
    calibrated_input.spectra[0].counts = {.5, .5, 1, 1, 1.5, 1.5, 2, 2};
    near(inverse.predict(calibrated_input).values[0], prediction.values[0]);
    std::get<model::InverseModelMetadata>(inverse_package.metadata.model)
        .input_preparation[0]
        .steps.insert(std::get<model::InverseModelMetadata>(inverse_package.metadata.model)
                          .input_preparation[0]
                          .steps.begin(),
                      model::RemovePileup{.01});
    inverse_package.write(root / "inverse-pileup.ibm");
    inference::InverseModel pileup_inverse(model::Package::open(root / "inverse-pileup.ibm"));
    auto piled_input = input;
    piled_input.spectra[0].counts.resize(7);
    processing::PileupWorkspace workspace;
    processing::apply_pileup(input.spectra[0].counts, piled_input.spectra[0].counts, 10, 8, .01,
                             workspace);
    near(pileup_inverse.predict(piled_input).values[0], prediction.values[0]);
    piled_input = input;
    piled_input.pileup_already_removed = true;
    near(pileup_inverse.predict(piled_input).values[0], prediction.values[0]);
    std::get<model::ForwardModelMetadata>(forward_package.metadata.model)
        .output_preparation[0]
        .steps.push_back(model::Pileup{.00001});
    forward_package.write(root / "forward-pileup.ibm");
    inference::ForwardModel pileup_forward(model::Package::open(root / "forward-pileup.ibm"));
    auto forward_piled = pileup_forward.predict(x);
    inference::InferenceOptions bare_options;
    bare_options.apply_pileup = false;
    inference::ForwardModel bare_forward(model::Package::open(root / "forward-pileup.ibm"), bare_options);
    require(bare_forward.predict(x).spectra[0].counts == out.spectra[0].counts,
            "disable declared pileup without changing count scaling");
    require(bare_forward.spectrum_channels()[0] == 4, "disabled pileup retains bare width");
    core::Counts expected_piled(7);
    processing::apply_pileup(out.spectra[0].counts, expected_piled, 10, 8, .00001, workspace);
    for (std::size_t i = 0; i < expected_piled.size(); ++i)
        near(forward_piled.spectra[0].counts[i], expected_piled[i]);
    input.setup.measurements[0].beam.energy += 1;
    rejects([&] { inverse.predict(input); });
    auto scaled_inverse_package = inverse_package;
    std::get<model::InverseModelMetadata>(scaled_inverse_package.metadata.model).output_transform.steps = {
        model::MultiplyScale{{1000}}};
    scaled_inverse_package.write(root / "edp-scale.ibm");
    inference::InverseModel scaled_inverse(model::Package::open(root / "edp-scale.ibm"));
    auto scaled_result = scaled_inverse.predict(piled_input);
    near(scaled_result.values[0], 10000);
    near(scaled_result.presence_probability[0], .8);
    near(scaled_result.posterior_std[0], 200);
    auto preprocessing_package =
        model::Package::from_onnx(std::filesystem::path(IBEAMLAB_FIXTURES) / "forward.onnx",
                                  metadata(model::Direction::Forward));
    auto &pre = std::get<model::ForwardModelMetadata>(preprocessing_package.metadata.model);
    pre.input_transform.steps = {model::MinMaxScale{{0}, {200}}};
    pre.output_transform = {{"RBS", {{model::MultiplyScale{{1000}}}}}};
    preprocessing_package.write(root / "preprocessing.ibm");
    inference::ForwardModel scaled_forward(model::Package::open(root / "preprocessing.ibm"));
    near(scaled_forward.predict(x).spectra[0].counts[0], 500);
    pre.input_transform.steps = {model::CustomTransform{"missing.forward", 1, {}}};
    rejects([&] { inference::ForwardModel unavailable_custom(preprocessing_package); });
    processing::register_custom_transform(
        "test.forward", 1, [](std::span<float> values, const model::CustomParameters &parameters) {
            for (auto &v : values)
                v *= static_cast<float>(parameters.at("factor")[0]);
        });
    pre.input_transform.steps = {model::CustomTransform{"test.forward", 1, {{"factor", {.5}}}}};
    pre.output_preparation.clear();
    pre.output_transform.clear();
    preprocessing_package.write(root / "custom.ibm");
    inference::ForwardModel custom_forward(model::Package::open(root / "custom.ibm"));
    near(custom_forward.predict(x).spectra[0].counts[0], 50);
    auto multi = metadata(model::Direction::Forward);
    auto &mf = std::get<model::ForwardModelMetadata>(multi.model);
    auto nra = mf.reference_setup.measurements[0];
    nra.label = "NRA";
    nra.beam.energy = 3000;
    nra.acquisition.particlesSr = 2000;
    mf.reference_setup.measurements.push_back(nra);
    mf.output_spectra.spectra.push_back({"NRA", 3});
    mf.output_preparation = {
        {"RBS", {model::ExposureScale{}, model::Rebin{}}},
        {"NRA", {model::ExposureScale{}, model::Rebin{}}}};
    mf.output_transform = {{"NRA", {{model::MultiplyScale{{.5}}}}}};
    require(mf.output_spectra.width() == 7 && mf.output_spectra.offset("NRA") == 4,
            "multi-spectrum graph mapping");
    auto multi_package = model::Package::from_onnx(
        std::filesystem::path(IBEAMLAB_FIXTURES) / "forward_multi.onnx", multi);
    multi_package.write(root / "multi-forward.ibm");
    inference::ForwardModel multi_forward(model::Package::open(root / "multi-forward.ibm"));
    auto requested = reference(mf);
    requested.setup.measurements[0].acquisition.particlesSr = 2000;
    requested.setup.measurements[1].acquisition.particlesSr = 1000;
    std::reverse(requested.setup.measurements.begin(), requested.setup.measurements.end());
    auto multi_output = multi_forward.predict(requested);
    require(multi_output.spectra[0].measurement == "RBS" &&
                multi_output.spectra[1].measurement == "NRA",
            "output measurement ordering");
    require(multi_output.spectra[0].counts.size() == 4 &&
                multi_output.spectra[1].counts.size() == 3,
            "different detector channel counts");
    near(multi_output.spectra[0].counts[0], 200);
    near(multi_output.spectra[1].counts[0], 125);
    std::reverse(requested.setup.measurements.begin(), requested.setup.measurements.end());
    auto reordered_output = multi_forward.predict(requested);
    require(reordered_output.spectra[0].counts == multi_output.spectra[0].counts &&
                reordered_output.spectra[1].counts == multi_output.spectra[1].counts,
            "measurement mapping refreshed between calls");
    auto duplicate = requested;
    duplicate.setup.measurements[1] = duplicate.setup.measurements[0];
    rejects([&] { multi_forward.predict(duplicate); });
    auto multi_inverse_metadata = metadata(model::Direction::Inverse);
    auto &mi = std::get<model::InverseModelMetadata>(multi_inverse_metadata.model);
    mi.reference_setup = mf.reference_setup;
    mi.input_spectra = mf.output_spectra;
    mi.input_preparation = {{"RBS", {model::ExposureScale{}, model::Rebin{}}},
                            {"NRA", {model::ExposureScale{}, model::Rebin{}}}};
    auto multi_inverse_package = model::Package::from_onnx(
        std::filesystem::path(IBEAMLAB_FIXTURES) / "inverse_multi.onnx", multi_inverse_metadata);
    multi_inverse_package.write(root / "multi-inverse.ibm");
    inference::InverseModel multi_inverse(model::Package::open(root / "multi-inverse.ibm"));
    inference::InverseInput multi_input{
        requested.setup, {{"NRA", {.5, 1, 1.5}}, {"RBS", {2, 4, 6, 8}}}, false};
    auto multi_prediction = multi_inverse.predict(multi_input);
    near(multi_prediction.values[0], 70);
    near(multi_prediction.values[1], 140);
    inference::Prediction multi_prepared;
    std::vector<core::Spectrum> reference_spectra{{"NRA", {1, 2, 3}}, {"RBS", {1, 2, 3, 4}}};
    multi_inverse.predict_reference(reference_spectra, multi_prepared);
    near(multi_prepared.values[0], 70);
    multi_input.spectra.pop_back();
    rejects([&] { multi_inverse.predict(multi_input); });
    auto bad_multi = multi;
    std::get<model::ForwardModelMetadata>(bad_multi.model).output_spectra.spectra[1].measurement =
        "RBS";
    rejects([&] { bad_multi.validate(); });
    auto malformed = inverse_package;
    std::get<model::InverseModelMetadata>(malformed.metadata.model)
        .input_spectra.spectra[0]
        .channels = 5;
    rejects([&] { inference::InverseModel bad(malformed); });
#else
    rejects([&] { inference::ForwardModel unavailable(package); });
#endif
}
int main(int argc, char **argv) {
    try {
        const std::string suite = argc == 1 ? "all" : argv[1];
        if (argc > 2 || (suite != "all" && suite != "processing" && suite != "generation" &&
                         suite != "packages"))
            throw std::invalid_argument(
                "usage: ibeamlab_workflow_tests [processing|generation|packages]");
        if (suite == "all" || suite == "processing")
            processing_tests();
        if (suite == "all" || suite == "generation") {
            Temporary root;
            generation_tests(root.path);
        }
        if (suite == "all" || suite == "packages") {
            Temporary root;
            package_tests(root.path);
        }
        std::cout << suite << " tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
