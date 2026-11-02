#include <algorithm>
#include <charconv>
#include <fstream>
#include <ibeamlab/io/reader.h>
#include <sstream>
namespace ibeamlab::io {
namespace {
std::string trim(std::string value) {
    auto first = value.find_first_not_of(" \t\r\n");
    return first == std::string::npos
               ? std::string()
               : value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}
std::vector<std::string> fields(const std::string &line, char delimiter, char comment) {
    std::vector<std::string> result;
    std::string value;
    bool quoted = false, after_quote = false;
    auto emit = [&] {
        result.push_back(trim(value));
        value.clear();
        after_quote = false;
    };
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    value += '"';
                    ++i;
                } else {
                    quoted = false;
                    after_quote = true;
                }
            } else
                value += c;
        } else if (comment && c == comment)
            break;
        else if (c == '"') {
            if (!trim(value).empty() || after_quote)
                throw std::invalid_argument("unexpected quote");
            value.clear();
            quoted = true;
        } else if (delimiter ? c == delimiter : (c == ' ' || c == '\t')) {
            if (delimiter || !value.empty() || after_quote)
                emit();
        } else {
            if (after_quote && c != ' ' && c != '\t' && c != '\r')
                throw std::invalid_argument("characters after quoted field");
            if (!after_quote)
                value += c;
        }
    }
    if (quoted)
        throw std::invalid_argument("unterminated quoted field");
    if (!value.empty() || after_quote || (delimiter && !result.empty()))
        emit();
    return result;
}
double number(const std::string &text) {
    if (text.empty())
        throw std::invalid_argument("empty numeric field");
    // from_chars is locale independent; accept an explicit leading plus as well.
    const char *first = text.data(), *last = first + text.size();
    if (*first == '+')
        ++first;
    double value = 0;
    auto parsed = std::from_chars(first, last, value);
    if (parsed.ec != std::errc() || parsed.ptr != last || !std::isfinite(value))
        throw std::invalid_argument("invalid finite number: " + text);
    return value;
}
} // namespace
core::MeasuredSpectrum read_spectrum(const std::filesystem::path &path, std::string measurement,
                                     ReadOptions options) {
    if (!options.maximum_channels || options.channel_base > 1)
        throw std::invalid_argument("invalid reader channel limit/base");
    if (options.format != SpectrumFormat::Counts &&
        options.format != SpectrumFormat::ChannelCounts && options.format != SpectrumFormat::Csv)
        throw std::invalid_argument("unknown spectrum format");
    if (options.format == SpectrumFormat::Counts) {
        options.counts_column = 0;
        options.channel_column.reset();
    }
    if (options.format == SpectrumFormat::Csv && !options.delimiter)
        options.delimiter = ',';
    if (options.comment && options.delimiter == options.comment)
        throw std::invalid_argument("delimiter and comment characters coincide");
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open spectrum: " + path.string());
    core::MeasuredSpectrum output;
    output.measurement = std::move(measurement);
    output.source = path.string();
    std::string line;
    std::size_t line_number = 0, records = 0;
    bool header = options.header;
    while (std::getline(input, line)) {
        ++line_number;
        if (options.stop_row && line_number > *options.stop_row)
            break;
        if (line_number == 1 && line.compare(0, 3, "\xef\xbb\xbf") == 0)
            line.erase(0, 3);
        if (line_number <= options.skip_rows)
            continue;
        try {
            const auto stripped = trim(line);
            if (stripped.empty() || (options.comment && stripped.front() == options.comment))
                continue;
            auto values = fields(line, options.delimiter, options.comment);
            if (values.empty())
                continue;
            if (header) {
                header = false;
                continue;
            }
            auto get = [&](std::size_t column) {
                if (column >= values.size())
                    throw std::invalid_argument("missing requested column " +
                                                std::to_string(column));
                return number(values[column]);
            };
            std::size_t channel = records;
            if (options.channel_column) {
                const auto raw = get(*options.channel_column);
                if (raw < options.channel_base || std::floor(raw) != raw ||
                    raw - options.channel_base >= options.maximum_channels)
                    throw std::invalid_argument(
                        "channel must be an integer within the configured range");
                channel = static_cast<std::size_t>(raw) - options.channel_base;
            }
            if (channel >= options.maximum_channels)
                throw std::invalid_argument("maximum channel count exceeded");
            const auto count = get(options.counts_column);
            const double variance = options.variance_column ? get(*options.variance_column) : 0;
            if (count < 0 || variance < 0)
                throw std::invalid_argument("negative counts or variance");
            if (channel >= output.counts.size()) {
                output.counts.resize(channel + 1);
                output.valid.resize(channel + 1);
                if (options.variance_column)
                    output.variances.resize(channel + 1);
            }
            if (output.valid[channel])
                throw std::invalid_argument("duplicate channel " + std::to_string(channel));
            output.counts[channel] = count;
            output.valid[channel] = 1;
            if (options.variance_column)
                output.variances[channel] = variance;
            ++records;
        } catch (const std::exception &error) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " +
                                     error.what());
        }
    }
    if (input.bad())
        throw std::runtime_error("error reading spectrum: " + path.string());
    if (!records)
        throw std::runtime_error("no spectrum records: " + path.string());
    if (records == output.counts.size())
        output.valid.clear();
    output.validate();
    return output;
}
} // namespace ibeamlab::io
