#include <ibeamlab/fitting/parameterization.h>
#include <set>
#include <stdexcept>
#include <tuple>
namespace ibeamlab::fitting {
Parameterization::Parameterization(profile::Model m, std::vector<ProfileParameter> p)
    : model_(std::move(m)), discretizer_(model_), parameters_(std::move(p)) {
    std::set<std::tuple<std::size_t, std::size_t, std::size_t>> used;
    for (const auto &v : parameters_) {
        v.bounds.validate();
        if (!used.emplace(v.layer, v.species, v.coefficient).second ||
            v.layer >= model_.layers.size() ||
            v.species >= model_.layers[v.layer].functions.size() ||
            v.coefficient >= model_.layers[v.layer].functions[v.species].coefficients.size() ||
            !v.bounds.contains(
                model_.layers[v.layer].functions[v.species].coefficients[v.coefficient]))
            throw std::invalid_argument(
                "invalid/duplicate profile coefficient target or initial bounds");
        double initial = model_.layers[v.layer].functions[v.species].coefficients[v.coefficient];
        for (const auto &link : v.links) {
            if (!std::isfinite(link.ratio) ||
                !used.emplace(link.layer, link.species, link.coefficient).second ||
                link.layer >= model_.layers.size() ||
                link.species >= model_.layers[link.layer].functions.size() ||
                link.coefficient >=
                    model_.layers[link.layer].functions[link.species].coefficients.size())
                throw std::invalid_argument("invalid or duplicate linked profile coefficient");
            double value =
                model_.layers[link.layer].functions[link.species].coefficients[link.coefficient];
            if (std::abs(value - initial * link.ratio) > 1e-10 * std::max(1., std::abs(value)))
                throw std::invalid_argument("linked profile initial value differs from ratio");
        }
    }
}
void Parameterization::apply(std::span<const double> c, const core::Sample &macro,
                             core::Sample &out) {
    if (c.size() != parameters_.size() || macro.layers.size() != model_.layers.size())
        throw std::invalid_argument("profile parameter dimensions differ");
    for (std::size_t i = 0; i < c.size(); ++i) {
        const auto &p = parameters_[i];
        if (!p.bounds.contains(c[i]))
            throw std::invalid_argument("profile coefficient outside bounds");
        model_.layers[p.layer].functions[p.species].coefficients[p.coefficient] = c[i];
        for (const auto &link : p.links)
            model_.layers[link.layer].functions[link.species].coefficients[link.coefficient] =
                c[i] * link.ratio;
    }
    for (std::size_t i = 0; i < macro.layers.size(); ++i)
        model_.layers[i].reference = macro.layers[i];
    discretizer_.apply(model_, out);
}
} // namespace ibeamlab::fitting
