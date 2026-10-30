#include <ibeamlab/core/isotopes.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace ibeamlab::core {
namespace {
struct Element { std::string_view symbol; std::size_t offset, count, naturalOffset, naturalCount; };
#include "isotope_data.inc"
const Element &find(std::string_view symbol) {
    for (const auto &element : elements) if (element.symbol == symbol) return element;
    throw std::invalid_argument("unknown element: " + std::string(symbol));
}
}
std::span<const IsotopeData> isotope_data(std::string_view element) {
    const auto &e = find(element); return {isotope_records + e.offset, e.count};
}
std::span<const Isotope> natural_isotopes(std::string_view element) {
    const auto &e = find(element); return {natural_records + e.naturalOffset, e.naturalCount};
}
std::span<const Isotope> resolved_isotopes(const Species &species) {
    if (!species.isotopes.empty()) return species.isotopes;
    auto result = natural_isotopes(species.element);
    if (result.empty()) throw std::invalid_argument("explicit isotope composition required for " + species.element);
    return result;
}
void validate_isotopes(std::string_view element, std::span<const Isotope> composition) {
    const auto known = isotope_data(element);
    if (composition.empty()) {
        if (natural_isotopes(element).empty())
            throw std::invalid_argument("explicit isotope composition required for " + std::string(element));
        return;
    }
    double sum = 0;
    for (std::size_t i = 0; i < composition.size(); ++i) {
        const auto &isotope = composition[i];
        if (std::none_of(known.begin(), known.end(), [&](auto &k) {return k.massNumber == isotope.massNumber;}))
            throw std::invalid_argument("unsupported isotope: " + std::string(element) + "-" + std::to_string(isotope.massNumber));
        for (std::size_t j = 0; j < i; ++j)
            if (composition[j].massNumber == isotope.massNumber)
                throw std::invalid_argument("duplicate isotope for " + std::string(element));
        if (!std::isfinite(isotope.fraction) || isotope.fraction < 0)
            throw std::invalid_argument("isotope fractions must be finite and nonnegative");
        sum += isotope.fraction;
    }
    if (!std::isfinite(sum) || std::abs(sum - 1) > 1e-6)
        throw std::invalid_argument("isotope fractions must sum to one");
}
bool same_isotopes(std::span<const Isotope> a, std::span<const Isotope> b, double tolerance) noexcept {
    auto fraction = [](auto list, int mass) {
        for (const auto &i : list) if (i.massNumber == mass) return i.fraction;
        return 0.;
    };
    for (const auto &i : a) if (std::abs(i.fraction - fraction(b, i.massNumber)) > tolerance) return false;
    for (const auto &i : b) if (std::abs(i.fraction - fraction(a, i.massNumber)) > tolerance) return false;
    return true;
}
} // namespace ibeamlab::core
