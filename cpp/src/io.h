#pragma once
#include <ibeamlab/core/isotopes.h>
#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <ibeamlab/core/parameter.h>
#include <ibeamlab/core/spectrum.h>
#include <limits>
namespace ibeamlab::detail {
inline constexpr std::uint64_t max_payload = 512ULL * 1024 * 1024;
inline std::uint64_t checksum(std::span<const std::byte> bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (auto b : bytes) {
        hash ^= std::to_integer<unsigned char>(b);
        hash *= 1099511628211ULL;
    }
    return hash;
}
struct Encoder {
    std::vector<std::byte> bytes;
    void integer(std::uint64_t value) {
        for (int i = 0; i < 8; ++i)
            bytes.push_back(std::byte((value >> (8 * i)) & 255));
    }
    void real(double value) { integer(std::bit_cast<std::uint64_t>(value)); }
    void text(std::string_view value) {
        integer(value.size());
        for (unsigned char c : value)
            bytes.push_back(std::byte(c));
    }
    void reals(std::span<const double> values) {
        integer(values.size());
        for (double v : values)
            real(v);
    }
};
struct Decoder {
    std::span<const std::byte> bytes;
    std::size_t offset = 0;
    std::uint64_t integer() {
        if (bytes.size() - offset < 8)
            throw std::runtime_error("truncated binary data");
        std::uint64_t value = 0;
        for (int i = 0; i < 8; ++i)
            value |= std::uint64_t(std::to_integer<unsigned char>(bytes[offset++])) << (8 * i);
        return value;
    }
    std::size_t count(std::size_t limit = 1000000) {
        auto value = integer();
        if (value > limit)
            throw std::runtime_error("binary count exceeds limit");
        return static_cast<std::size_t>(value);
    }
    double real() { return std::bit_cast<double>(integer()); }
    std::string text() {
        auto n = count(1024 * 1024);
        if (n > bytes.size() - offset)
            throw std::runtime_error("truncated text");
        std::string s(reinterpret_cast<const char *>(bytes.data() + offset), n);
        offset += n;
        return s;
    }
    void reals_into(core::Counts &values) {
        auto n = count(max_payload / 8);
        if (n > (bytes.size() - offset) / 8)
            throw std::runtime_error("truncated array");
        values.resize(n);
        for (auto &v : values)
            v = real();
    }
    core::Counts reals() {
        core::Counts values;
        reals_into(values);
        return values;
    }
    void finish() {
        if (offset != bytes.size())
            throw std::runtime_error("unexpected trailing data");
    }
};
inline void write_envelope(const std::filesystem::path &path, std::string_view magic,
                           const Encoder &encoder) {
    if (magic.size() != 8 || encoder.bytes.size() > max_payload)
        throw std::runtime_error("invalid envelope size");
    if (std::filesystem::exists(path))
        throw std::runtime_error("refusing to overwrite: " + path.string());
    Encoder header;
    header.integer(encoder.bytes.size());
    header.integer(checksum(encoder.bytes));
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(magic.data(), 8);
    f.write(reinterpret_cast<const char *>(header.bytes.data()), 16);
    f.write(reinterpret_cast<const char *>(encoder.bytes.data()),
            static_cast<std::streamsize>(encoder.bytes.size()));
    f.close();
    if (!f)
        throw std::runtime_error("cannot write envelope");
}
inline std::vector<std::byte> read_envelope(const std::filesystem::path &path,
                                            std::string_view magic) {
    auto size = std::filesystem::file_size(path);
    if (size < 24 || size > max_payload + 24)
        throw std::runtime_error("invalid envelope length");
    std::ifstream f(path, std::ios::binary);
    std::array<char, 8> tag{};
    f.read(tag.data(), 8);
    if (std::string_view(tag.data(), 8) != magic)
        throw std::runtime_error("unsupported format or version");
    std::array<std::byte, 16> h{};
    f.read(reinterpret_cast<char *>(h.data()), 16);
    Decoder header{h};
    auto n = header.integer(), hash = header.integer();
    if (n != size - 24)
        throw std::runtime_error("envelope length mismatch");
    std::vector<std::byte> bytes(static_cast<std::size_t>(n));
    f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(n));
    if (!f || checksum(bytes) != hash)
        throw std::runtime_error("envelope checksum mismatch");
    return bytes;
}
inline void encode(Encoder &e, const core::ExperimentalSetup &setup) {
    e.integer(setup.measurements.size());
    for (const auto &m : setup.measurements) {
        e.text(m.label);
        e.text(m.beam.particle);
        e.real(m.beam.energy);
        e.real(m.beam.spread);
        e.real(m.detector.calibrationLinear);
        e.real(m.detector.calibrationOffset);
        e.real(m.detector.calibrationQuadratic);
        e.real(m.detector.resolution);
        e.text(m.detector.info);
        e.real(m.acquisition.particlesSr);
        e.real(m.acquisition.realTime);
        e.real(m.acquisition.liveTime);
    }
}
inline void encode(Encoder &e, const core::Sample &x) {
    e.integer(x.layers.size());
    for (const auto &l : x.layers) {
        e.real(l.thickness);
        e.real(l.roughness);
        e.real(l.porosityFraction);
        e.real(l.poreDiameter);
        e.integer(l.species.size());
        for (const auto &s : l.species) {
            e.text(s.element);
            e.real(s.concentration);
            const auto composition = core::resolved_isotopes(s);
            e.integer(composition.size());
            for (const auto &i : composition) {
                e.integer(i.massNumber);
                e.real(0.0); // Reserved mass slot; identity is element plus mass number.
                e.real(i.fraction);
            }
        }
    }
}
inline void encode(Encoder &e, const core::Experiment &x) {
    encode(e, x.sample);
    encode(e, x.setup);
}
inline core::ExperimentalSetup setup(Decoder &d) {
    core::ExperimentalSetup result;
    result.measurements.resize(d.count(1024));
    for (auto &m : result.measurements) {
        m.label = d.text();
        m.beam.particle = d.text();
        m.beam.energy = d.real();
        m.beam.spread = d.real();
        m.detector.calibrationLinear = d.real();
        m.detector.calibrationOffset = d.real();
        m.detector.calibrationQuadratic = d.real();
        m.detector.resolution = d.real();
        m.detector.info = d.text();
        m.acquisition.particlesSr = d.real();
        m.acquisition.realTime = d.real();
        m.acquisition.liveTime = d.real();
    }
    result.validate();
    return result;
}
inline core::Sample sample(Decoder &d) {
    core::Sample x;
    auto layers = d.count(10000);
    x.layers.resize(layers);
    for (auto &l : x.layers) {
        l.thickness = d.real();
        l.roughness = d.real();
        l.porosityFraction = d.real();
        l.poreDiameter = d.real();
        l.species.resize(d.count(256));
        for (auto &s : l.species) {
            s.element = d.text();
            s.concentration = d.real();
            s.isotopes.resize(d.count(512));
            for (auto &i : s.isotopes) {
                auto mass = d.integer();
                if (mass > 10000)
                    throw std::runtime_error("invalid isotope mass");
                i.massNumber = static_cast<std::int32_t>(mass);
                (void)d.real(); // Reserved mass slot.
                i.fraction = d.real();
            }
        }
    }
    x.validate();
    return x;
}
inline core::Experiment experiment(Decoder &d) {
    core::Experiment x;
    x.sample = sample(d);
    x.setup = setup(d);
    return x;
}
inline void encode(Encoder &e, const core::ParameterTarget &t) {
    e.integer(t.index());
    std::visit(
        [&](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, core::LayerThickness>)
                e.integer(v.layer);
            else if constexpr (std::is_same_v<T, core::SpeciesConcentration>) {
                e.integer(v.layer);
                e.text(v.element);
            } else
                e.text(v.measurement);
        },
        t);
}
inline core::ParameterTarget target(Decoder &d) {
    switch (d.integer()) {
    case 0:
        return core::LayerThickness{d.count(10000)};
    case 1: {
        auto n = d.count(10000);
        return core::SpeciesConcentration{n, d.text()};
    }
    case 2:
        return core::BeamEnergy{d.text()};
    case 3:
        return core::BeamSpread{d.text()};
    case 4:
        return core::CalibrationLinear{d.text()};
    case 5:
        return core::CalibrationOffset{d.text()};
    case 6:
        return core::CalibrationQuadratic{d.text()};
    case 7:
        return core::DetectorResolution{d.text()};
    case 8:
        return core::ParticlesSr{d.text()};
    default:
        throw std::runtime_error("unknown parameter target");
    }
}
inline void encode(Encoder &e, const core::Spectrum &s) {
    e.text(s.measurement);
    e.reals(s.counts);
}
inline void spectrum_into(Decoder &d, core::Spectrum &s) {
    s.measurement = d.text();
    d.reals_into(s.counts);
    core::validate_counts(s.counts);
}
} // namespace ibeamlab::detail
