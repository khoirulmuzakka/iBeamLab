#include <cmath>
#include <ibeamlab/fitting/constraints.h>
#include <stdexcept>
namespace ibeamlab::fitting {
void validate_ratio(double ratio) {
    if (!std::isfinite(ratio))
        throw std::invalid_argument("nonfinite linked parameter ratio");
}
} // namespace ibeamlab::fitting
