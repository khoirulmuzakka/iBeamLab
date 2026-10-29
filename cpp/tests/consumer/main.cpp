#include <ibeamlab/core/parameter.h>
#include <ibeamlab/model/metadata.h>
#include <ibeamlab/processing/rebin.h>
int main() {
    ibeamlab::model::SpectraSchema spectra{{{"RBS", 4}, {"NRA", 3}}};
    if (spectra.width() != 7 || spectra.offset("NRA") != 4)
        return 1;
    ibeamlab::core::Counts counts{2, 4}, edges{0, 1, 2};
    auto result = ibeamlab::processing::rebin(counts, edges, edges);
    return result == counts ? 0 : 1;
}
