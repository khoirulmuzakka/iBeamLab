#include <ibeamlab/profile/functions.h>
#include <stdexcept>
namespace ibeamlab::profile {
void Function::validate() const {
    if (kind != FunctionKind::Constant && kind != FunctionKind::Polynomial &&
        kind != FunctionKind::ErfPolynomial)
        throw std::invalid_argument("unknown profile function");
    if (coefficients.empty() || (kind == FunctionKind::Constant && coefficients.size() != 1) ||
        (kind == FunctionKind::ErfPolynomial && (coefficients.size() < 4 || coefficients[3] <= 0)))
        throw std::invalid_argument("invalid profile coefficients or erf transition width");
    for (auto v : coefficients)
        if (!std::isfinite(v))
            throw std::invalid_argument("nonfinite profile coefficient");
}
double Function::normalized(double y) const {
    double v = 0;
    if (kind == FunctionKind::ErfPolynomial) {
        double poly = 0;
        for (std::size_t i = coefficients.size(); i > 4; --i)
            poly = (poly + coefficients[i - 1]) * y;
        v = coefficients[0] *
            (1 + coefficients[1] * std::erf((y - coefficients[2]) / coefficients[3])) * (1 + poly);
    } else
        for (auto i = coefficients.rbegin(); i != coefficients.rend(); ++i)
            v = v * y + *i;
    if (!std::isfinite(v))
        throw std::invalid_argument("nonfinite evaluated profile");
    return std::abs(v);
}
double Function::evaluate(double depth, double shift, double width) const {
    validate();
    if (!std::isfinite(depth) || !std::isfinite(shift) || !std::isfinite(width) || width <= 0)
        throw std::invalid_argument("profile depth/width must be finite; width positive");
    return depth < shift || depth > shift + width ? 0 : normalized((depth - shift) / width);
}
} // namespace ibeamlab::profile
