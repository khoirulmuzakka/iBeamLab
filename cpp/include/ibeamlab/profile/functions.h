#pragma once
#include <ibeamlab/core/spectrum.h>
namespace ibeamlab::profile {
/** Profile in normalized local depth y=(depth-shift)/width. Coefficients are
 * ascending polynomial powers; ErfPolynomial uses [A,a,b,c,p1,...]. Values
 * outside the layer are zero; absolute magnitude matches the profile convention.
 * Erf transition c must be positive. Width and depth use 1e15 atoms/cm2. */
enum class FunctionKind { Constant, Polynomial, ErfPolynomial };
struct IBEAMLAB_API Function {
    FunctionKind kind = FunctionKind::Constant;
    core::Counts coefficients{1};
    void validate() const;
    double evaluate(double depth, double shift, double width) const;
    double normalized(double y) const;
};
} // namespace ibeamlab::profile
