#include <ibeamlab/profile/model.h>
#include <stdexcept>
namespace ibeamlab::profile {
core::Sample Model::reference_sample() const {
    core::Sample out;
    for (const auto &l : layers)
        out.layers.push_back(l.reference);
    return out;
}
void Model::validate() const {
    reference_sample().validate();
    std::size_t total = 0;
    for (const auto &l : layers) {
        if (!l.sublayers || l.sublayers > 1000000 || total > 1000000 - l.sublayers ||
            l.functions.size() != l.reference.species.size() ||
            (l.spacing != Spacing::Linear && l.spacing != Spacing::Logarithmic))
            throw std::invalid_argument("invalid profile topology, functions or spacing");
        total += l.sublayers;
        for (const auto &f : l.functions)
            f.validate();
    }
}
} // namespace ibeamlab::profile
