#include <fstream>
#include <ibeamlab/io/parser.h>
#include <sstream>
#include <stdexcept>
namespace ibeamlab::io {
namespace {
std::optional<double> time(const std::vector<std::string> &lines, const TimeField &field) {
    if (!std::isfinite(field.seconds_per_unit) || field.seconds_per_unit <= 0 ||
        (field.fallback && (!std::isfinite(*field.fallback) || *field.fallback < 0)))
        throw std::invalid_argument("invalid acquisition time settings");
    if (!field.marker.empty())
        for (const auto &line : lines) {
            auto at = line.find(field.marker);
            if (at == std::string::npos)
                continue;
            std::string tail = line.substr(at + field.marker.size());
            for (auto &c : tail)
                if (c == ':' || c == '=')
                    c = ' ';
            std::istringstream fields(tail);
            double value;
            if (!(fields >> value) || !std::isfinite(value) || value < 0 ||
                !std::isfinite(value * field.seconds_per_unit))
                throw std::runtime_error("invalid header time: " + field.marker);
            return value * field.seconds_per_unit;
        }
    return field.fallback;
}
} // namespace
ImportedSpectrum read_spectrum(const std::filesystem::path &path, std::string label,
                               const Parser &p) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open spectrum: " + path.string());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);
    if (in.bad())
        throw std::runtime_error("cannot read spectrum: " + path.string());
    auto options = p.records;
    std::size_t start = options.skip_rows, stop = lines.size();
    if (!p.start_marker.empty()) {
        bool found = false;
        for (std::size_t j = start; j < lines.size(); ++j)
            if (lines[j].find(p.start_marker) != std::string::npos) {
                start = j + 1;
                found = true;
                break;
            }
        if (!found)
            throw std::runtime_error("missing data start marker: " + p.start_marker);
    }
    if (!p.end_marker.empty()) {
        bool found = false;
        for (std::size_t j = start; j < lines.size(); ++j)
            if (lines[j].find(p.end_marker) != std::string::npos) {
                stop = j;
                found = true;
                break;
            }
        if (!found)
            throw std::runtime_error("missing data end marker: " + p.end_marker);
    }
    options.skip_rows = start;
    options.stop_row = stop;
    ImportedSpectrum out;
    out.real_time = time(lines, p.real_time);
    out.live_time = time(lines, p.live_time);
    if (out.real_time && out.live_time && *out.live_time > *out.real_time)
        throw std::invalid_argument("live time exceeds real time");
    out.spectrum = read_spectrum(path, std::move(label), options);
    if (p.missing_channels_are_zero)
        out.spectrum.valid.clear();
    return out;
}
} // namespace ibeamlab::io
