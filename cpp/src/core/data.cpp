#include <ibeamlab/core/data.h>
#include <unordered_set>
namespace ibeamlab::core {
void MeasuredSpectrum::validate() const {
    if (measurement.empty() || counts.empty())
        throw std::invalid_argument("measured spectrum requires label and counts");
    if ((!valid.empty() && valid.size() != counts.size()) ||
        (!variances.empty() && variances.size() != counts.size()))
        throw std::invalid_argument("measured mask/variance dimensions differ from counts");
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (!std::isfinite(counts[i]) || counts[i] < 0)
            throw std::invalid_argument("measured counts must be finite and nonnegative");
        if (!valid.empty() && valid[i] > 1)
            throw std::invalid_argument("validity mask must contain zero or one");
        if (!variances.empty() && (!std::isfinite(variances[i]) || variances[i] < 0))
            throw std::invalid_argument("variances must be finite and nonnegative");
    }
}
void MeasuredData::validate() const {
    setup.validate();
    if (spectra.size() != setup.measurements.size())
        throw std::invalid_argument("observations must match setup labels");
    std::unordered_set<std::string> labels;
    for (const auto &s : spectra) {
        s.validate();
        find_measurement(setup, s.measurement);
        if (!labels.insert(s.measurement).second)
            throw std::invalid_argument("duplicate measured spectrum label");
    }
}
} // namespace ibeamlab::core
