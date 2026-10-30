#include "io.h"
#include <algorithm>
#include <ibeamlab/datasets/dataset.h>
#include <unordered_set>
namespace ibeamlab::datasets {
namespace {
void encode_metadata(detail::Encoder &e, const Metadata &m) {
    detail::encode(e, m.reference);
    e.integer(m.parameters.size());
    for (auto &p : m.parameters) {
        e.text(p.name);
        detail::encode(e, p.target);
        e.text(p.unit);
    }
    e.text(m.provenance);
    e.integer(m.requested);
    e.integer(m.accepted);
    e.integer(m.failed);
    e.integer(m.discarded);
}
Metadata decode_metadata(detail::Decoder &d) {
    Metadata m;
    m.reference = detail::experiment(d);
    m.parameters.resize(d.count(100000));
    for (auto &p : m.parameters) {
        p.name = d.text();
        p.target = detail::target(d);
        p.unit = d.text();
    }
    m.provenance = d.text();
    m.requested = d.integer();
    m.accepted = d.integer();
    m.failed = d.integer();
    m.discarded = d.integer();
    m.validate();
    return m;
}
void record_valid(const Record &r, const Metadata &m) {
    if (r.sample_id.empty() || r.sample_index >= m.requested ||
        r.parameters.size() != m.parameters.size())
        throw std::invalid_argument("invalid dataset record identity or parameter dimensions");
    for (double v : r.parameters)
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite dataset parameter");
    if (r.result.failure)
        return;
    if (r.result.spectra.size() != m.reference.setup.measurements.size())
        throw std::invalid_argument("dataset measurement count mismatch");
    for (std::size_t i = 0; i < r.result.spectra.size(); ++i) {
        const auto &s = r.result.spectra[i];
        if (s.measurement != m.reference.setup.measurements[i].label || s.counts.empty())
            throw std::invalid_argument("dataset measurement order mismatch");
        core::validate_counts(s.counts);
    }
}
std::string shard_name(std::size_t n) { return "shard-" + std::to_string(n) + ".ibd"; }
} // namespace
void Metadata::validate() const {
    reference.validate();
    core::validate_parameters(reference, parameters);
    if (accepted > requested || failed > requested - accepted ||
        discarded > requested - accepted - failed)
        throw std::invalid_argument("dataset counters exceed requested count");
}
struct Writer::Impl {
    std::filesystem::path path;
    Metadata metadata;
    std::size_t shard_limit, shards = 0, pending = 0;
    detail::Encoder payload;
    bool finished = false;
    std::optional<std::uint64_t> last;
    void flush() {
        if (!pending)
            return;
        detail::Encoder e;
        e.integer(pending);
        e.bytes.insert(e.bytes.end(), payload.bytes.begin(), payload.bytes.end());
        auto dest = path / shard_name(shards);
        auto temp = dest;
        temp += ".tmp";
        detail::write_envelope(temp, "IBSHRD01", e);
        std::filesystem::rename(temp, dest);
        ++shards;
        pending = 0;
        payload.bytes.clear();
    }
};
Writer::Writer(const std::filesystem::path &path, Metadata metadata, std::size_t limit)
    : impl_(std::make_unique<Impl>()) {
    metadata.validate();
    if (!limit || limit > 100000 || metadata.accepted || metadata.failed || metadata.discarded)
        throw std::invalid_argument("invalid writer settings");
    if (!std::filesystem::create_directory(path))
        throw std::runtime_error("dataset directory already exists");
    impl_->path = path;
    impl_->metadata = std::move(metadata);
    impl_->shard_limit = limit;
    detail::Encoder e;
    encode_metadata(e, impl_->metadata);
    detail::write_envelope(path / "incomplete.ibd", "IBDATA01", e);
}
Writer::~Writer() = default;
const Metadata &Writer::metadata() const { return impl_->metadata; }
void Writer::append(const Record &r) {
    auto &w = *impl_;
    if (w.finished)
        throw std::logic_error("dataset already finished");
    record_valid(r, w.metadata);
    if (w.last && r.sample_index <= *w.last)
        throw std::invalid_argument("sample indices must increase");
    if (w.metadata.accepted + w.metadata.failed + w.metadata.discarded >= w.metadata.requested)
        throw std::invalid_argument("too many dataset records");
    detail::Encoder e;
    e.integer(r.sample_index);
    e.text(r.sample_id);
    e.reals(r.parameters);
    e.integer(r.result.failure ? 1 : 0);
    if (r.result.failure) {
        e.text(r.result.failure->measurement);
        e.text(r.result.failure->message);
    } else {
        e.integer(r.result.spectra.size());
        for (auto &s : r.result.spectra)
            detail::encode(e, s);
    }
    if (e.bytes.size() > detail::max_payload - 8)
        throw std::invalid_argument("record too large");
    if (w.payload.bytes.size() + e.bytes.size() > detail::max_payload - 8)
        w.flush();
    w.payload.bytes.insert(w.payload.bytes.end(), e.bytes.begin(), e.bytes.end());
    ++w.pending;
    w.last = r.sample_index;
    if (r.result.failure)
        ++w.metadata.failed;
    else
        ++w.metadata.accepted;
    if (w.pending >= w.shard_limit)
        w.flush();
}
void Writer::discard() {
    auto &w = *impl_;
    if (w.finished ||
        w.metadata.accepted + w.metadata.failed + w.metadata.discarded >= w.metadata.requested)
        throw std::logic_error("cannot discard record");
    ++w.metadata.discarded;
}
void Writer::finish() {
    auto &w = *impl_;
    if (w.finished)
        return;
    if (w.metadata.accepted + w.metadata.failed + w.metadata.discarded != w.metadata.requested)
        throw std::logic_error("cannot finish incomplete dataset");
    w.flush();
    detail::Encoder e;
    encode_metadata(e, w.metadata);
    e.integer(w.shards);
    detail::write_envelope(w.path / "manifest.tmp", "IBDATA01", e);
    std::filesystem::rename(w.path / "manifest.tmp", w.path / "manifest.ibd");
    w.finished = true;
    std::filesystem::remove(w.path / "incomplete.ibd");
}
struct Reader::Impl {
    std::filesystem::path path;
    Metadata metadata;
    std::size_t shards = 0, index = 0, remaining = 0;
    std::vector<std::byte> bytes;
    detail::Decoder decoder{{}};
    std::uint64_t accepted = 0, failed = 0;
    std::optional<std::uint64_t> last;
};
Reader::Reader(const std::filesystem::path &path) : impl_(std::make_unique<Impl>()) {
    auto &r = *impl_;
    r.path = path;
    if (!std::filesystem::exists(path / "manifest.ibd"))
        throw std::runtime_error("incomplete dataset: no committed manifest");
    auto bytes = detail::read_envelope(path / "manifest.ibd", "IBDATA01");
    detail::Decoder d{bytes};
    r.metadata = decode_metadata(d);
    r.shards = d.count(10000000);
    d.finish();
    if (r.metadata.accepted + r.metadata.failed + r.metadata.discarded != r.metadata.requested)
        throw std::runtime_error("incomplete dataset counters");
}
Reader::~Reader() = default;
const Metadata &Reader::metadata() const { return impl_->metadata; }
std::map<std::string, std::size_t> Reader::maximum_spectrum_lengths() const {
    Reader cursor(impl_->path);
    Record record;
    std::map<std::string, std::size_t> lengths;
    while (cursor.next(record)) {
        if (record.result.failure) continue;
        for (const auto &spectrum : record.result.spectra) {
            auto &length = lengths[spectrum.measurement];
            length = std::max(length, spectrum.counts.size());
        }
    }
    return lengths;
}
void Reader::rewind() {
    auto &r = *impl_;
    r.index = r.remaining = 0;
    r.bytes.clear();
    r.decoder = {{}};
    r.accepted = r.failed = 0;
    r.last.reset();
}
bool Reader::next(Record &out) {
    auto &r = *impl_;
    if (!r.remaining) {
        if (r.index) {
            r.decoder.finish();
        }
        if (r.index == r.shards) {
            if (r.accepted != r.metadata.accepted || r.failed != r.metadata.failed)
                throw std::runtime_error("dataset record counters mismatch");
            return false;
        }
        r.bytes = detail::read_envelope(r.path / shard_name(r.index++), "IBSHRD01");
        r.decoder = {r.bytes};
        r.remaining = r.decoder.count(100000);
        if (!r.remaining)
            throw std::runtime_error("empty shard");
    }
    auto &d = r.decoder;
    out.sample_index = d.integer();
    out.sample_id = d.text();
    d.reals_into(out.parameters);
    out.result.failure.reset();
    auto status = d.integer();
    if (status == 1) {
        out.result.spectra.clear();
        auto label = d.text();
        auto message = d.text();
        out.result.failure = simulation::Failure{label, message};
        ++r.failed;
    } else if (status == 0) {
        out.result.spectra.resize(d.count(1024));
        for (auto &s : out.result.spectra)
            detail::spectrum_into(d, s);
        ++r.accepted;
    } else
        throw std::runtime_error("unknown record status");
    record_valid(out, r.metadata);
    if (r.last && out.sample_index <= *r.last)
        throw std::runtime_error("unordered record index");
    r.last = out.sample_index;
    --r.remaining;
    return true;
}
} // namespace ibeamlab::datasets
