#include <fstream>
#include <ibeamlab/model_package.h>
#include <iomanip>
#include <memory>
#include <miniz.h>
#include <sstream>
#include <stdexcept>
#include <toml++/toml.hpp>

namespace ibeamlab::model {
namespace {
constexpr std::uint64_t MaxManifestSize = 4ULL * 1024 * 1024,
                        MaxModelSize = 2ULL * 1024 * 1024 * 1024;
std::vector<std::byte> readFile(const std::filesystem::path &path, std::uint64_t limit) {
    const auto size = std::filesystem::file_size(path);
    if (size > limit)
        throw std::runtime_error("package entry exceeds safety limit: " + path.string());
    std::vector<std::byte> out(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(size));
    if (!input)
        throw std::runtime_error("failed to read package entry: " + path.string());
    return out;
}
struct Archive {
    mz_zip_archive zip{};
    explicit Archive(const std::filesystem::path &path) {
        if (!mz_zip_reader_init_file(&zip, path.string().c_str(), 0))
            throw std::runtime_error("cannot open model ZIP");
    }
    ~Archive() { mz_zip_reader_end(&zip); }
    std::vector<std::byte> read(const std::string &name, std::uint64_t limit) {
        if (name.empty() || name.find('/') != std::string::npos ||
            name.find('\\') != std::string::npos || name == "." || name == "..")
            throw std::runtime_error("unsafe model ZIP entry name: " + name);
        const int index =
            mz_zip_reader_locate_file(&zip, name.c_str(), nullptr, MZ_ZIP_FLAG_CASE_SENSITIVE);
        if (index < 0)
            throw std::runtime_error("model ZIP is missing " + name);
        mz_zip_archive_file_stat stat{};
        if (!mz_zip_reader_file_stat(&zip, index, &stat) || stat.m_is_directory ||
            stat.m_uncomp_size > limit)
            throw std::runtime_error("invalid or oversized model ZIP entry: " + name);
        std::vector<std::byte> out(static_cast<std::size_t>(stat.m_uncomp_size));
        if (!mz_zip_reader_extract_to_mem(&zip, index, out.data(), out.size(), 0))
            throw std::runtime_error("failed CRC validation while extracting: " + name);
        return out;
    }
};
std::vector<std::string> strings(const toml::array *a) {
    std::vector<std::string> out;
    if (a)
        for (const auto &n : *a) {
            auto v = n.value<std::string>();
            if (!v)
                throw std::runtime_error("non-string list value in package");
            out.push_back(*v);
        }
    return out;
}
std::vector<float> floats(const toml::array *a) {
    std::vector<float> out;
    if (a)
        for (const auto &n : *a) {
            auto v = n.value<double>();
            if (!v)
                throw std::runtime_error("non-number list value in package");
            out.push_back(static_cast<float>(*v));
        }
    return out;
}
std::vector<std::size_t> sizes(const toml::array *a) {
    std::vector<std::size_t> out;
    if (a)
        for (const auto &n : *a) {
            auto v = n.value<std::int64_t>();
            if (!v || *v <= 0)
                throw std::runtime_error("invalid spectrum length");
            out.push_back(static_cast<std::size_t>(*v));
        }
    return out;
}
TransformSpec transform(const toml::table *t) {
    TransformSpec o;
    if (!t)
        return o;
    o.type = (*t)["type"].value_or<std::string>("identity");
    o.inputDimension = (*t)["input_dimension"].value_or<std::size_t>(0);
    o.factor = (*t)["factor"].value_or(1.0F);
    o.offset = (*t)["offset"].value_or(1.0F);
    o.low = (*t)["low"].value_or(0.0F);
    o.high = (*t)["high"].value_or(1.0F);
    o.minimum = floats((*t)["minimum"].as_array());
    o.scale = floats((*t)["scale"].as_array());
    o.mean = floats((*t)["mean"].as_array());
    o.deviation = floats((*t)["deviation"].as_array());
    if (auto a = (*t)["transforms"].as_array())
        for (const auto &n : *a) {
            if (!n.is_table())
                throw std::runtime_error("pipeline transform must be a TOML table");
            o.transforms.push_back(transform(n.as_table()));
        }
    return o;
}
std::string crc32(const std::vector<std::byte> &bytes) {
    const auto value = mz_crc32(
        MZ_CRC32_INIT, reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(8) << value;
    return out.str();
}
} // namespace
ModelPackage ModelPackage::open(const std::filesystem::path &path) {
    std::vector<std::byte> manifest;
    std::unique_ptr<Archive> archive;
    if (std::filesystem::is_directory(path))
        manifest = readFile(path / "package.toml", MaxManifestSize);
    else {
        archive = std::make_unique<Archive>(path);
        manifest = archive->read("package.toml", MaxManifestSize);
    }
    const std::string text(reinterpret_cast<const char *>(manifest.data()), manifest.size());
    const auto root = toml::parse(text);
    if (root["format"].value_or<std::string>("") != "ibeamlab.onnx-package")
        throw std::runtime_error("unsupported model package format");
    ModelPackage package;
    auto &m = package.metadata_;
    m.formatVersion = root["format_version"].value_or<std::uint32_t>(0);
    if (m.formatVersion != 1)
        throw std::runtime_error("unsupported model package major version");
    m.createdUtc = root["created_utc"].value_or<std::string>("");
    const auto *model = root["model"].as_table();
    if (!model)
        throw std::runtime_error("model package has no [model] table");
    m.task = (*model)["task"].value_or<std::string>("");
    m.className = (*model)["class_name"].value_or<std::string>("");
    m.inputName = (*model)["input_name"].value_or<std::string>("inputs");
    m.outputName = (*model)["output_name"].value_or<std::string>("outputs");
    m.inputDimension = (*model)["input_dimension"].value_or<std::size_t>(0);
    m.outputDimension = (*model)["output_dimension"].value_or<std::size_t>(0);
    m.opsetVersion = (*model)["opset_version"].value_or(0);
    m.modelSize = (*model)["size"].value_or<std::uint64_t>(0);
    m.modelChecksum = (*model)["crc32"].value_or<std::string>("");
    m.methodNames = strings(root["methods"].as_array());
    m.spectrumLengths = sizes(root["spectrum_lengths"].as_array());
    m.inputFeatures = strings(root["input_features"].as_array());
    m.outputFeatures = strings(root["output_features"].as_array());
    m.outputUnits = strings(root["output_units"].as_array());
    if (m.methodNames.empty() || m.inputDimension == 0 || m.outputDimension == 0)
        throw std::runtime_error("package is missing required dimensions or methods");
    if (!m.spectrumLengths.empty()) {
        if (m.spectrumLengths.size() != m.methodNames.size())
            throw std::runtime_error("spectrum_lengths and methods differ in size");
        std::size_t total = 0;
        for (auto n : m.spectrumLengths)
            total += n;
        if (total != m.inputDimension)
            throw std::runtime_error("spectrum lengths do not match input dimension");
    }
    if (m.outputFeatures.size() != m.outputDimension)
        throw std::runtime_error("output features do not match output dimension");
    if (!m.outputUnits.empty() && m.outputUnits.size() != m.outputDimension)
        throw std::runtime_error("output units do not match output dimension");
    if (auto p = root["preprocessing"].as_table()) {
        m.inputTransform = transform((*p)["input"].as_table());
        m.outputTransform = transform((*p)["output"].as_table());
    }
    const auto filename = (*model)["onnx_file"].value_or<std::string>("model.onnx");
    package.modelBytes_ =
        archive ? archive->read(filename, MaxModelSize) : readFile(path / filename, MaxModelSize);
    if (m.modelSize && m.modelSize != package.modelBytes_.size())
        throw std::runtime_error("model size does not match package.toml");
    if (!m.modelChecksum.empty() && m.modelChecksum != crc32(package.modelBytes_))
        throw std::runtime_error("model CRC32 checksum does not match package.toml");
    return package;
}
} // namespace ibeamlab::model
