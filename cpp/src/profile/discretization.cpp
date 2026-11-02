#include <ibeamlab/core/isotopes.h>
#include <ibeamlab/profile/discretization.h>
#include <stdexcept>
namespace ibeamlab::profile {
Discretizer::Discretizer(const Model &m) {
    m.validate();
    identity_ = m.reference_sample();
    for (const auto &l : m.layers) {
        core::Counts e(l.sublayers + 1);
        for (std::size_t j = 0; j < e.size(); ++j) {
            double y = double(j) / double(l.sublayers);
            e[j] = l.spacing == Spacing::Linear ? y : std::pow(4., -3 + 3 * y) - std::pow(4., -3);
        }
        e.back() = 1;
        edges_.push_back(std::move(e));
        species_.push_back(l.functions.size());
        spacing_.push_back(l.spacing);
    }
}
void Discretizer::apply(const Model &m, core::Sample &out) const {
    if (m.layers.size() != edges_.size())
        throw std::invalid_argument("profile topology changed");
    std::size_t total = 0;
    for (std::size_t i = 0; i < m.layers.size(); ++i) {
        const auto &l = m.layers[i];
        if (l.sublayers + 1 != edges_[i].size() || l.functions.size() != species_[i] ||
            l.reference.species.size() != species_[i] || l.spacing != spacing_[i])
            throw std::invalid_argument("profile topology changed");
        for (const auto &f : l.functions)
            f.validate();
        const auto &r = l.reference;
        if (!std::isfinite(r.roughness) || r.roughness < 0 || !std::isfinite(r.porosityFraction) ||
            r.porosityFraction < 0 || r.porosityFraction > 1 || !std::isfinite(r.poreDiameter) ||
            r.poreDiameter < 0)
            throw std::invalid_argument("invalid profile roughness/porosity");
        for (std::size_t k = 0; k < r.species.size(); ++k) {
            const auto &s = r.species[k];
            const auto &expected = identity_.layers[i].species[k];
            if (s.element != expected.element)
                throw std::invalid_argument("profile species identity changed");
            core::validate_isotopes(s.element, s.isotopes);
            if (!core::same_isotopes(core::resolved_isotopes(s), core::resolved_isotopes(expected)))
                throw std::invalid_argument("profile isotope identity changed");
        }
        if (!std::isfinite(l.reference.thickness) || l.reference.thickness < 0)
            throw std::invalid_argument("invalid profile thickness");
        total += l.sublayers;
    }
    out.layers.resize(total);
    std::size_t index = 0;
    for (std::size_t i = 0; i < m.layers.size(); ++i) {
        const auto &l = m.layers[i];
        const auto &e = edges_[i];
        for (std::size_t j = 0; j < l.sublayers; ++j) {
            auto &dest = out.layers[index++];
            dest.roughness = l.reference.roughness;
            dest.porosityFraction = l.reference.porosityFraction;
            dest.poreDiameter = l.reference.poreDiameter;
            dest.thickness = (e[j + 1] - e[j]) * l.reference.thickness;
            dest.species.resize(l.reference.species.size());
            double sum = 0;
            for (std::size_t k = 0; k < dest.species.size(); ++k) {
                auto &s = dest.species[k];
                s = l.reference.species[k];
                s.concentration = l.functions[k].normalized((e[j] + e[j + 1]) * .5);
                sum += s.concentration;
            }
            if (!(sum > 0) || !std::isfinite(sum))
                throw std::invalid_argument("profile has zero/nonfinite composition total");
            for (auto &s : dest.species)
                s.concentration /= sum;
        }
    }
}
core::Sample discretize(const Model &m) {
    Discretizer d(m);
    core::Sample out;
    d.apply(m, out);
    return out;
}
} // namespace ibeamlab::profile
