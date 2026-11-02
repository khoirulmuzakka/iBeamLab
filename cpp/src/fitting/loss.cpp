#include <algorithm>
#include <cmath>
#include <ibeamlab/fitting/loss.h>
#include <stdexcept>
namespace ibeamlab::fitting {
double loss_value(Loss loss, double d, double p, double variance, double floor) {
    const double delta = d - p;
    switch (loss) {
    case Loss::DataChiSquare:
        return delta * delta / (d + floor);
    case Loss::VarianceChiSquare:
        return delta * delta / variance;
    case Loss::Neyman:
        return delta * delta / std::max(d, 1.);
    case Loss::Pearson:
        return delta * delta / std::max(p, 1.);
    case Loss::Mse:
        return delta * delta;
    case Loss::Poisson:
        return 2 * (p - d + d * std::log(std::max(d, 1.) / std::max(p, 1.)));
    case Loss::Gaussian: {
        double mp = std::sqrt(.25 + d * d) - .5, m = std::max(mp, 1.), t = std::max(p, 1.);
        return delta * delta / t + std::log(t / m) - (mp - d) * (mp - d) / m;
    }
    }
    throw std::invalid_argument("unknown fitting loss");
}
} // namespace ibeamlab::fitting
